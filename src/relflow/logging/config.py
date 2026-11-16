from __future__ import annotations

import logging
import os
from collections.abc import Mapping, MutableMapping
from typing import Any, Self, TextIO, cast

CONTEXT = "relflow_context"
# Uvicorn's vocabulary extends the standard names with ``trace``, its finest level.
LEVELS: dict[str, int] = {
    "critical": logging.CRITICAL,
    "error": logging.ERROR,
    "warning": logging.WARNING,
    "info": logging.INFO,
    "debug": logging.DEBUG,
    "trace": logging.DEBUG,
}


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


def severity(level: str | int) -> int:
    """Translate a level name from RelFlow's or Uvicorn's vocabulary into a standard level."""

    if isinstance(level, bool) or not isinstance(level, (str, int)):
        raise TypeError(f"RelFlow log level must be a string or integer, got {type(level).__name__}")
    if isinstance(level, int):
        return level
    try:
        return LEVELS[level.strip().lower()]
    except KeyError:
        choices = ", ".join(name.upper() for name in LEVELS)
        raise ValueError(f"RelFlow log level must be one of {choices}, got {level!r}") from None


def configure(*, level: str | int | None = None, output: TextIO | None = None) -> Logger:
    """Install one plain stream handler, retaining handlers owned by callers.

    An explicit ``level`` is validated strictly. Otherwise ``RELFLOW_LOG_LEVEL``
    selects the level, and an unknown value falls back to INFO with a warning
    so that importing the package never fails on environment configuration.
    """

    unknown: str | None = None
    if level is None:
        requested = os.getenv("RELFLOW_LOG_LEVEL", "DEBUG")
        try:
            selected = severity(requested)
        except ValueError:
            selected, unknown = logging.INFO, requested
    else:
        selected = severity(level)

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
    configured = Logger(core, {})
    if unknown is not None:
        choices = ", ".join(name.upper() for name in LEVELS)
        configured.bind(component="logging").warning(
            "RELFLOW_LOG_LEVEL=%r is not one of %s; logging at INFO", unknown, choices
        )
    return configured


logger = configure()

__all__ = ["CONTEXT", "LEVELS", "Logger", "configure", "logger", "severity"]
