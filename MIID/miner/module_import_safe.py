import importlib, importlib.util, sys, os, hashlib, threading

_probe_lock = threading.Lock()

def get_module_version(qualified: str):
    """
    Get the version of the module.
    """
    spec = importlib.util.find_spec(qualified)
    if spec is None or not spec.origin or not os.path.exists(spec.origin):
        return None
    with open(spec.origin, "r") as f:
        data = json.load(f)
        return data.get("version", None)

def probe_fresh_import(qualified: str):
    """
    Try to import `qualified` (e.g. 'MIID.miner.generate_name_variations_v2')
    bypassing module cache. Returns (ok: bool, info: dict|None, err: str|None).
    """
    importlib.invalidate_caches()
    spec = importlib.util.find_spec(qualified)
    if spec is None or not spec.origin or not os.path.exists(spec.origin):
        return False, None, f"Spec/origin not found for {qualified}"

    path = spec.origin
    # Use file hash to mint a unique module name and bypass caching
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    digest = h.hexdigest()
    tmp_name = f"_probe_{qualified.replace('.', '_')}_{digest[:12]}"

    spec2 = importlib.util.spec_from_file_location(tmp_name, path)
    if spec2 is None or spec2.loader is None:
        return False, {"path": path, "digest": digest}, "No loader for module"

    module = importlib.util.module_from_spec(spec2)

    # Some loaders expect presence in sys.modules during exec; add then remove.
    with _probe_lock:
        sys.modules[tmp_name] = module
        try:
            spec2.loader.exec_module(module)  # runs top-level code; catches import-time errors
            ok, err = True, None
        except Exception as e:
            ok, err = False, f"{type(e).__name__}: {e}"
        finally:
            sys.modules.pop(tmp_name, None)

    return ok, {"path": path, "digest": digest}, err
