"""Drop-in replacement for the ``bt.Config`` / argparse machinery removed in v11.

v11 deleted ``bt.Config`` and every ``add_args`` classmethod on ``bt.Wallet``
/ ``bt.Subtensor`` / ``bt.Axon`` — the SDK no longer touches ``sys.argv`` at
all. This module reimplements the small piece of that machinery MIID
actually relies on: turning a flat argparse namespace whose dests contain
dots (``--wallet.name`` -> dest ``wallet.name``) into a nested,
attribute-accessible config object, exactly like the old ``bt.Config``.
"""

import argparse
import copy


class DotDict(dict):
    """A dict that also supports attribute access, recursively."""

    def __getattr__(self, item):
        try:
            value = self[item]
        except KeyError as e:
            raise AttributeError(item) from e
        if isinstance(value, dict) and not isinstance(value, DotDict):
            value = DotDict(value)
            self[item] = value
        return value

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        try:
            del self[item]
        except KeyError as e:
            raise AttributeError(item) from e

    def copy(self):
        return copy.deepcopy(self)


class Config(DotDict):
    """Nested config object built from an argparse parser, replacing bt.Config.

    Usage mirrors the old bittensor API: ``Config(parser)`` parses
    ``sys.argv`` (or ``args=``) and nests any dotted destinations
    (``wallet.name``) into ``config.wallet.name``.
    """

    def __init__(self, parser: "argparse.ArgumentParser" = None, args=None):
        super().__init__()
        if parser is None:
            return

        namespace, _ = parser.parse_known_args(args=args)
        flat = vars(namespace)

        for dotted_key, value in flat.items():
            parts = dotted_key.split(".")
            cursor = self
            for part in parts[:-1]:
                nxt = cursor.get(part)
                if not isinstance(nxt, DotDict):
                    nxt = DotDict()
                    cursor[part] = nxt
                cursor = nxt
            cursor[parts[-1]] = value

    def merge(self, other):
        """Recursively merge another config's non-None values into this one."""
        if other is None:
            return

        def _merge(dst: dict, src: dict):
            for k, v in src.items():
                if isinstance(v, dict):
                    nxt = dst.get(k)
                    if not isinstance(nxt, dict):
                        nxt = DotDict()
                        dst[k] = nxt
                    _merge(nxt, v)
                elif v is not None:
                    dst[k] = v

        _merge(self, other)
