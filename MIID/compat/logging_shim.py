"""Drop-in replacement for the ``bittensor.logging`` object removed in v11.

bittensor v11 deleted ``bt.logging`` entirely in favor of the standard
``logging`` module. Rather than touching every one of the hundreds of
``bt.logging.info(...)`` call sites across this codebase, this module
provides an object with the same surface area, backed by the stdlib
``logging`` module, that gets installed as ``bittensor.logging`` at import
time (see ``MIID/compat/__init__.py``).
"""

import logging
import sys

_LOGGER_NAME = "MIID"
_logger = logging.getLogger(_LOGGER_NAME)

# EVENT sits between INFO and WARNING, mirroring the old bittensor level.
EVENT_LEVEL_NUM = 25
if not hasattr(logging, "EVENT"):
    logging.addLevelName(EVENT_LEVEL_NUM, "EVENT")


class _BtLoggingShim:
    """Mimics the old ``bittensor.logging`` module-level API."""

    def __init__(self, logger: logging.Logger):
        self._logger = logger
        self._configured = False
        self._trace_enabled = False
        self._debug_enabled = False

    # -- configuration -----------------------------------------------
    def _ensure_handler(self):
        if self._configured:
            return
        self._configured = True
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)8s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        self._logger.addHandler(handler)
        self._logger.setLevel(logging.INFO)
        self._logger.propagate = False

    def set_config(self, config=None):
        self._ensure_handler()
        if config is None:
            return
        try:
            if getattr(config, "trace", False):
                self.set_trace(True)
            elif getattr(config, "debug", False):
                self.set_debug(True)
        except Exception:
            pass

    def check_config(self, config):
        # v10's bt.logging.check_config validated the logging dir; the
        # neuron code that calls this does its own directory creation, so
        # this is a no-op kept for interface compatibility.
        return True

    def add_args(self, parser):
        parser.add_argument(
            "--logging.debug",
            action="store_true",
            default=False,
            help="Turn on bittensor debugging information.",
        )
        parser.add_argument(
            "--logging.trace",
            action="store_true",
            default=False,
            help="Turn on bittensor trace level information.",
        )
        parser.add_argument(
            "--logging.logging_dir",
            type=str,
            default="~/.bittensor/miners",
            help="Logging directory.",
        )

    def register_primary_logger(self, name):
        # Old bittensor routed EVENT-level records to an extra logger; we
        # just no-op here since events.log is written directly by
        # setup_events_logger in MIID/utils/logging.py.
        return None

    def set_debug(self, on: bool = True):
        self._ensure_handler()
        self._debug_enabled = on
        self._logger.setLevel(logging.DEBUG if on else logging.INFO)

    def set_trace(self, on: bool = True):
        self._ensure_handler()
        self._trace_enabled = on
        if on:
            self._debug_enabled = True
            self._logger.setLevel(logging.DEBUG)

    # -- logging methods -----------------------------------------------
    @staticmethod
    def _format(msg, args):
        if args:
            return " ".join(str(a) for a in (msg, *args))
        return str(msg)

    def trace(self, msg="", *args, **kwargs):
        self._ensure_handler()
        if self._trace_enabled:
            self._logger.debug(self._format(msg, args))

    def debug(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.debug(self._format(msg, args))

    def info(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.info(self._format(msg, args))

    def success(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.info(self._format(msg, args))

    def warning(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.warning(self._format(msg, args))

    def error(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.error(self._format(msg, args))

    def critical(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.critical(self._format(msg, args))

    def exception(self, msg="", *args, **kwargs):
        self._ensure_handler()
        self._logger.exception(self._format(msg, args))


bt_logging = _BtLoggingShim(_logger)
