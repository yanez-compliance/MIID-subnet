"""Drop-in replacement for the ``bt.Synapse`` base class removed in v11.

v11's migration guide is explicit that Synapse's shared request schema has
no replacement — "your own request/response models — signing covers raw
bytes, so any schema works" (see the migration guide's
"Axon, Dendrite, and Synapse are gone" section). This module recreates just
enough of the old Synapse surface (``.dendrite`` / ``.axon`` terminal info,
``.deserialize()``, ``.name``) that the rest of this codebase — which reads
``response.dendrite.status_code`` etc. everywhere — keeps working unchanged
against our own FastAPI/httpx transport (``MIID/compat/transport.py``).
"""

from typing import List, Optional

from pydantic import BaseModel, ConfigDict, PrivateAttr


class TerminalInfo(BaseModel):
    """Equivalent of the old bt.TerminalInfo attached as `.axon` / `.dendrite`."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    status_code: Optional[int] = None
    status_message: Optional[str] = None
    process_time: Optional[float] = None
    ip: Optional[str] = None
    port: Optional[int] = None
    version: Optional[int] = None
    nonce: Optional[int] = None
    uuid: Optional[str] = None
    hotkey: Optional[str] = None
    signature: Optional[str] = None


class Synapse(BaseModel):
    """Replacement base class for bt.Synapse.

    Subclasses (e.g. MIID.protocol.IdentitySynapse) add their own request /
    response fields on top of this, exactly as they did with the real
    bt.Synapse in bittensor <11.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: Optional[str] = None
    timeout: float = 12.0
    total_size: Optional[int] = None
    header_size: Optional[int] = None
    dendrite: Optional[TerminalInfo] = None
    axon: Optional[TerminalInfo] = None
    computed_body_hash: str = ""
    required_hash_fields: List[str] = []

    # Not part of the wire schema: the raw request context (headers, body,
    # method, path) stashed by CompatAxon so a custom verify_fn can still
    # call `axon.default_verify(synapse)` for the real crypto check.
    _transport_ctx: dict = PrivateAttr(default_factory=dict)

    def __init__(self, **data):
        super().__init__(**data)
        if self.name is None:
            self.name = type(self).__name__
        if self.dendrite is None:
            self.dendrite = TerminalInfo()
        if self.axon is None:
            self.axon = TerminalInfo()

    def deserialize(self):
        """Default deserialization: return self, same as old bt.Synapse."""
        return self
