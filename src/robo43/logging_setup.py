"""The one logging configuration for the whole pipeline.

The legacy scripts each carried their own ``setup_logging`` that called
``addHandler`` unconditionally, so importing two of them duplicated every line,
and gated the level on ``--verbose`` so ``--loglevel DEBUG`` alone did nothing.
This module replaces all of them: it is idempotent, it configures the single
``robo43`` logger that every module logs through, and it can route records from
worker processes back to the parent over a queue.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import multiprocessing.queues
    from collections.abc import Mapping

__all__ = [
    "ROOT_LOGGER_NAME",
    "JsonFormatter",
    "log_listener",
    "setup_logging",
    "worker_log_config",
]

ROOT_LOGGER_NAME = "robo43"

_TEXT_FORMAT = "%(asctime)s [%(levelname)s] @%(name)s.%(funcName)s() %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

#: Handlers this module installed are tagged so a second call can remove
#: exactly those and leave anything the embedding application added alone.
_OWNED = "_robo43_owned"

_JSON_SKIP = frozenset(
    {
        "args", "created", "exc_info", "exc_text", "filename", "funcName",
        "levelname", "levelno", "lineno", "module", "msecs", "msg", "name",
        "pathname", "process", "processName", "relativeCreated", "stack_info",
        "taskName", "thread", "threadName",
    }
)


class JsonFormatter(logging.Formatter):
    """One JSON object per record, for service mode (phase 8)."""

    def format(self, record: logging.LogRecord) -> str:
        import json

        payload: dict[str, Any] = {
            "ts": self.formatTime(record, _DATE_FORMAT),
            "level": record.levelname,
            "logger": record.name,
            "func": record.funcName,
            "msg": record.getMessage(),
            "pid": record.process,
        }
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        for key, value in record.__dict__.items():
            if key not in _JSON_SKIP and not key.startswith("_"):
                payload.setdefault(key, value)
        return json.dumps(payload, default=str)


def _coerce_level(level: str | int) -> int:
    if isinstance(level, int):
        return level
    resolved = logging.getLevelNamesMapping().get(level.upper())
    if resolved is None:
        msg = f"unknown log level {level!r}"
        raise ValueError(msg)
    return resolved


def _clear_owned(logger: logging.Logger) -> None:
    for handler in [h for h in logger.handlers if getattr(h, _OWNED, False)]:
        logger.removeHandler(handler)
        handler.close()


def setup_logging(
    level: str | int = "INFO",
    logfile: str | os.PathLike[str] | None = None,
    *,
    json_format: bool = False,
    queue: multiprocessing.queues.Queue[Any] | None = None,
) -> logging.Logger:
    """Configure and return the ``robo43`` logger.

    Calling this twice replaces the previous configuration rather than adding a
    second copy of every handler.  ``queue``, when given, sends records to a
    :class:`~logging.handlers.QueueHandler` instead of writing them directly,
    which is how worker processes log safely under ``multiprocessing``.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    numeric = _coerce_level(level)
    logger.setLevel(numeric)
    # Records reach our handlers only; the application's root logger is not ours
    # to write to.
    logger.propagate = False
    _clear_owned(logger)

    formatter: logging.Formatter = (
        JsonFormatter() if json_format else logging.Formatter(_TEXT_FORMAT, _DATE_FORMAT)
    )

    handlers: list[logging.Handler] = []
    if queue is not None:
        handlers.append(logging.handlers.QueueHandler(queue))
    else:
        handlers.append(logging.StreamHandler(stream=sys.stderr))
        if logfile is not None:
            handlers.append(logging.FileHandler(os.fspath(logfile), encoding="utf-8"))

    for handler in handlers:
        handler.setLevel(numeric)
        handler.setFormatter(formatter)
        handler.__dict__[_OWNED] = True
        logger.addHandler(handler)

    return logger


def worker_log_config(
    queue: multiprocessing.queues.Queue[Any],
    level: str | int = "INFO",
) -> None:
    """Point this process's ``robo43`` logger at the parent's queue.

    Intended as (part of) a ``multiprocessing.Pool(initializer=...)`` callable.
    """
    setup_logging(level, queue=queue)


def log_listener(
    queue: multiprocessing.queues.Queue[Any],
) -> logging.handlers.QueueListener:
    """A started listener that re-emits queued worker records in the parent.

    The caller owns it and must call ``.stop()``; the handlers it forwards to
    are whatever :func:`setup_logging` installed in the parent.
    """
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    listener = logging.handlers.QueueListener(
        queue, *logger.handlers, respect_handler_level=True
    )
    listener.start()
    return listener


def describe_handlers() -> Mapping[str, Any]:
    """Introspection helper used by the tests to assert idempotency."""
    logger = logging.getLogger(ROOT_LOGGER_NAME)
    return {
        "level": logging.getLevelName(logger.level),
        "n_handlers": len(logger.handlers),
        "owned": sum(1 for h in logger.handlers if getattr(h, _OWNED, False)),
    }
