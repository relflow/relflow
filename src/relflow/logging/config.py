from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from typing import Any, Self, TextIO, cast

CONTEXT = "relflow_context"
LOG_LEVEL = os.getenv("RELFLOW_LOG_LEVEL", "DEBUG").upper()


class Formatter(logging.Formatter):
    """Append structured context without changing standard logging behavior."""

    def formatMessage(self, record: logging.LogRecord) -> str:
        message = super().formatMessage(record)
        context = getattr(record, CONTEXT, {})
        return f"{message} {dict(context)!r}" if isinstance(context, Mapping) and context else message


FORMATTER = Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")


class Logger(logging.LoggerAdapter):
    """Standard logger with immutable structured context binding."""

    def bind(self, **context: object) -> Self:
        """Return an adapter combining the configured mapping with new context."""
        return type(self)(self.logger, {**cast(Mapping[str, object], self.extra), **context})

    def process(
        self,
        msg: object,
        kwargs: MutableMapping[str, Any],
    ) -> tuple[object, MutableMapping[str, Any]]:
        extra = dict(kwargs.get("extra") or {})
        supplied = extra.pop(CONTEXT, {})
        if not isinstance(supplied, Mapping):
            raise TypeError(f"logging {CONTEXT} must be a mapping, got {type(supplied).__name__}")
        extra[CONTEXT] = {**cast(Mapping[str, object], self.extra), **supplied}
        kwargs["extra"] = extra
        return msg, kwargs


def configure(*, level: str | int | None = None, output: TextIO | None = None) -> Logger:
    """Install one plain stream handler, retaining handlers owned by callers."""

    selected = LOG_LEVEL if level is None else level
    if isinstance(selected, str):
        name = selected.upper()
        levels = {
            "DEBUG": logging.DEBUG,
            "INFO": logging.INFO,
            "WARNING": logging.WARNING,
            "ERROR": logging.ERROR,
            "CRITICAL": logging.CRITICAL,
        }
        if name not in levels:
            choices = ", ".join(levels)
            raise ValueError(f"RelFlow log level must be one of {choices}, got {selected!r}")
        selected = levels[name]
    elif not isinstance(selected, int) or isinstance(selected, bool):
        raise TypeError(f"RelFlow log level must be a string or integer, got {type(selected).__name__}")

    core = logging.getLogger("relflow")
    core.setLevel(selected)
    core.propagate = False
    for existing in tuple(core.handlers):
        if existing.formatter is FORMATTER:
            core.removeHandler(existing)
            existing.close()

    handler = logging.StreamHandler(output)
    handler.setLevel(selected)
    handler.setFormatter(FORMATTER)
    core.addHandler(handler)
    return Logger(core, {})


logger = configure()

__all__ = ["CONTEXT", "LOG_LEVEL", "Logger", "configure", "logger"]
