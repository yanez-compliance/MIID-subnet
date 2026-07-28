"""Drop-in replacement for bt.Axon / bt.Dendrite, removed entirely in v11.

Per the v11 migration guide ("Axon, Dendrite, and Synapse are gone"), there
is no first-party miner/validator networking stack anymore — subnets are
expected to bring their own HTTP server/client and authenticate with
``bittensor.http_auth``. This module implements that: a FastAPI-based
``CompatAxon`` (server) and an httpx-based ``CompatDendrite`` (client), kept
API-compatible with the old ``bt.Axon`` / ``bt.Dendrite`` surface that the
rest of this codebase (``forward.py``, ``base/miner.py``, ``base/validator.py``,
``mock.py``) already calls.
"""

import asyncio
import inspect
import threading
import time
from typing import Callable, Optional

import bittensor as bt
import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

from MIID.compat.synapse import Synapse, TerminalInfo


class NotVerifiedException(Exception):
    """Raised by verify_fn / default_verify to reject a request (-> HTTP 401).

    Replaces bittensor.core.errors.NotVerifiedException, which no longer
    exists now that bt.Axon (and its request-verification pipeline) is gone.
    """


def _synapse_class_from_handler(fn: Callable) -> type:
    """Infer the Synapse subclass a forward/blacklist/priority fn expects.

    Mirrors how the old bt.Axon.attach() inferred the synapse type from the
    handler's type-annotated first parameter.
    """
    sig = inspect.signature(fn)
    params = [p for p in sig.parameters.values() if p.name != "self"]
    if not params or params[0].annotation is inspect.Parameter.empty:
        return Synapse
    annotation = params[0].annotation
    if isinstance(annotation, type):
        return annotation
    return Synapse


async def _maybe_await(value):
    if inspect.isawaitable(value):
        return await value
    return value


class CompatAxon:
    """Replacement for bt.Axon: a small FastAPI server signing/verifying with http_auth."""

    def __init__(self, wallet, config=None, port=None, ip=None, external_ip=None, external_port=None):
        self.wallet = wallet
        axon_cfg = getattr(config, "axon", None) if config is not None else None

        self.port = port or getattr(axon_cfg, "port", None) or 8091
        self.ip = ip or getattr(axon_cfg, "ip", None) or "0.0.0.0"
        self.external_ip = external_ip or getattr(axon_cfg, "external_ip", None)
        self.external_port = external_port or getattr(axon_cfg, "external_port", None) or self.port

        self.forward_fns: dict = {}
        self.blacklist_fns: dict = {}
        self.priority_fns: dict = {}
        # Public dict: miners may register a custom verify_fn per synapse
        # class name directly, e.g. self.axon.verify_fns["IdentitySynapse"] = fn
        self.verify_fns: dict = {}

        self._app: Optional[FastAPI] = None
        self._server: Optional[uvicorn.Server] = None
        self._thread: Optional[threading.Thread] = None
        self.started = False

    # -- registration ----------------------------------------------------
    def attach(self, forward_fn, blacklist_fn=None, priority_fn=None, verify_fn=None):
        cls = _synapse_class_from_handler(forward_fn)
        name = cls.__name__
        self.forward_fns[name] = forward_fn
        if blacklist_fn is not None:
            self.blacklist_fns[name] = blacklist_fn
        if priority_fn is not None:
            self.priority_fns[name] = priority_fn
        if verify_fn is not None:
            self.verify_fns[name] = verify_fn
        return self

    # -- verification ------------------------------------------------------
    async def default_verify(self, synapse: Synapse):
        """Cryptographically verify the request that produced `synapse`.

        Uses bittensor.http_auth.verify against the raw request context
        stashed on the synapse by the route handler before any verify_fn ran.
        """
        ctx = getattr(synapse, "_transport_ctx", None)
        if not ctx:
            raise NotVerifiedException("Missing transport context; cannot verify request.")
        try:
            caller = bt.http_auth.verify(
                ctx["headers"],
                ctx["body"],
                method=ctx["method"],
                path=ctx["path"],
                self_hotkey_ss58=self.wallet.hotkey.ss58_address,
            )
        except bt.http_auth.AuthError as e:
            raise NotVerifiedException(str(e)) from e
        synapse.dendrite.hotkey = caller.hotkey_ss58
        synapse.dendrite.nonce = caller.nonce_ns
        synapse.dendrite.uuid = str(caller.nonce_ns)
        return caller

    # -- serving -----------------------------------------------------------
    def serve(self, netuid: int, subtensor):
        from MIID.compat.chain import serve_axon

        ip = self.external_ip or _detect_external_ip()
        port = self.external_port or self.port
        return serve_axon(subtensor, self.wallet, netuid, ip, port)

    def _build_app(self) -> FastAPI:
        app = FastAPI()

        @app.get("/{synapse_name}")
        async def _get_probe(synapse_name: str):
            # Mirrors the old Axon's behavior of responding to a bare GET
            # (used by connectivity checks like MIID/miner/active_miner_check).
            return JSONResponse(status_code=200, content={"message": f"{synapse_name} endpoint is alive"})

        @app.post("/{synapse_name}")
        async def _handle(synapse_name: str, request: Request):
            return await self._handle_request(synapse_name, request)

        return app

    async def _handle_request(self, synapse_name: str, request: Request) -> Response:
        start = time.time()
        cls = None
        for registered_name, fn in self.forward_fns.items():
            if registered_name == synapse_name:
                cls = _synapse_class_from_handler(fn)
                break
        if cls is None:
            return JSONResponse(status_code=404, content={"message": f"Unknown synapse: {synapse_name}"})

        body = await request.body()
        try:
            synapse = cls.model_validate_json(body) if body else cls()
        except Exception as e:
            return JSONResponse(status_code=400, content={"message": f"Malformed request body: {e}"})

        client_host = request.client.host if request.client else None
        client_port = request.client.port if request.client else None
        synapse.dendrite = TerminalInfo(
            hotkey=request.headers.get(bt.http_auth.HEADER_HOTKEY),
            nonce=_safe_int(request.headers.get(bt.http_auth.HEADER_NONCE)),
            ip=client_host,
            port=client_port,
        )
        synapse.axon = TerminalInfo(hotkey=self.wallet.hotkey.ss58_address, ip=self.ip, port=self.port)
        synapse._transport_ctx = {
            "headers": request.headers,
            "body": body,
            "method": "POST",
            "path": request.url.path,
        }

        verify_fn = self.verify_fns.get(synapse_name, self.default_verify)
        try:
            await _maybe_await(verify_fn(synapse))
        except NotVerifiedException as e:
            return JSONResponse(status_code=401, content={"message": str(e)})
        except bt.http_auth.AuthError as e:
            return JSONResponse(status_code=401, content={"message": str(e)})
        except Exception as e:
            return JSONResponse(status_code=401, content={"message": f"Verification failed: {e}"})

        blacklist_fn = self.blacklist_fns.get(synapse_name)
        if blacklist_fn is not None:
            try:
                is_blacklisted, reason = await _maybe_await(blacklist_fn(synapse))
            except Exception as e:
                is_blacklisted, reason = True, f"blacklist_fn raised: {e}"
            if is_blacklisted:
                return JSONResponse(status_code=403, content={"message": reason})

        priority_fn = self.priority_fns.get(synapse_name)
        if priority_fn is not None:
            try:
                await _maybe_await(priority_fn(synapse))
            except Exception:
                pass

        forward_fn = self.forward_fns[synapse_name]
        try:
            result = await _maybe_await(forward_fn(synapse))
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"message": f"Internal Server Error: {e}"},
                headers={"process-time": str(time.time() - start)},
            )

        if result is None:
            result = synapse
        return Response(
            content=result.model_dump_json(),
            media_type="application/json",
            headers={"process-time": str(time.time() - start)},
        )

    def start(self):
        if self.started:
            return self
        self._app = self._build_app()
        config = uvicorn.Config(self._app, host=self.ip, port=self.port, log_level="warning", loop="asyncio")
        self._server = uvicorn.Server(config)

        def _run():
            asyncio.set_event_loop(asyncio.new_event_loop())
            asyncio.get_event_loop().run_until_complete(self._server.serve())

        self._thread = threading.Thread(target=_run, daemon=True)
        self._thread.start()
        # Give uvicorn a moment to bind before returning, mirroring the old
        # Axon.start()'s synchronous feel.
        for _ in range(50):
            if getattr(self._server, "started", False):
                break
            time.sleep(0.1)
        self.started = True
        return self

    def stop(self):
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
        self.started = False
        return self

    def __repr__(self):
        return f"CompatAxon({self.ip}:{self.port}, hotkey={self.wallet.hotkey.ss58_address})"

    __str__ = __repr__


def _safe_int(value):
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _detect_external_ip() -> str:
    try:
        import requests

        resp = requests.get("https://checkip.amazonaws.com", timeout=5)
        return resp.text.strip()
    except Exception:
        bt.logging.warning("Could not auto-detect external IP; defaulting to 0.0.0.0.")
        return "0.0.0.0"


class CompatDendrite:
    """Replacement for bt.Dendrite: an httpx client signing with http_auth."""

    def __init__(self, wallet):
        self.wallet = wallet
        # `.keypair` mirrors the old bt.Dendrite attribute used by MockDendrite.__str__.
        self.keypair = wallet.hotkey

    async def __call__(self, axons=None, synapse=None, deserialize: bool = True, timeout: float = 12, **kwargs):
        return await self.forward(axons=axons, synapse=synapse, deserialize=deserialize, timeout=timeout, **kwargs)

    async def forward(
        self,
        axons,
        synapse: Optional[Synapse] = None,
        timeout: float = 12,
        deserialize: bool = True,
        run_async: bool = True,
        streaming: bool = False,
        **kwargs,
    ):
        if streaming:
            raise NotImplementedError("Streaming is not supported by the v11 compat transport.")
        if synapse is None:
            synapse = Synapse()

        async with httpx.AsyncClient() as client:
            results = await asyncio.gather(
                *(self._call_single(client, axon, synapse, timeout) for axon in axons)
            )

        if deserialize:
            return [r.deserialize() for r in results]
        return list(results)

    def preprocess_synapse_for_request(self, axon, synapse: Synapse, timeout: float = 12) -> Synapse:
        """Attach sender/receiver terminal info before sending, like the old Dendrite did."""
        s = synapse.model_copy(deep=True)
        s.timeout = timeout
        s.dendrite = TerminalInfo(hotkey=self.wallet.hotkey.ss58_address, nonce=time.time_ns())
        s.axon = TerminalInfo(ip=getattr(axon, "ip", None), port=getattr(axon, "port", None), hotkey=getattr(axon, "hotkey", None))
        return s

    async def _call_single(self, client: httpx.AsyncClient, axon, synapse: Synapse, timeout: float) -> Synapse:
        cls = type(synapse)
        name = cls.__name__
        path = f"/{name}"
        url = f"http://{axon.ip}:{axon.port}{path}"
        body = synapse.model_dump_json(exclude={"dendrite", "axon"}).encode("utf-8")
        receiver_hotkey = getattr(axon, "hotkey", None)

        start = time.time()
        result = synapse.model_copy(deep=True)
        try:
            headers = bt.http_auth.sign(
                self.wallet, method="POST", path=path, body=body, receiver_ss58=receiver_hotkey
            )
            resp = await client.post(url, content=body, headers=headers, timeout=timeout)
            elapsed = time.time() - start
            if resp.status_code == 200:
                result = cls.model_validate_json(resp.content)
                result.dendrite = TerminalInfo(
                    status_code=200, status_message="OK", process_time=elapsed, hotkey=receiver_hotkey
                )
            else:
                result.dendrite = TerminalInfo(
                    status_code=resp.status_code,
                    status_message=resp.text[:500],
                    process_time=elapsed,
                    hotkey=receiver_hotkey,
                )
        except httpx.TimeoutException:
            elapsed = time.time() - start
            result.dendrite = TerminalInfo(
                status_code=408, status_message="Timeout", process_time=elapsed, hotkey=receiver_hotkey
            )
        except Exception as e:
            elapsed = time.time() - start
            result.dendrite = TerminalInfo(
                status_code=422, status_message=str(e)[:500], process_time=elapsed, hotkey=receiver_hotkey
            )
        return result

    def __repr__(self):
        return f"CompatDendrite({self.wallet.hotkey.ss58_address})"

    __str__ = __repr__
