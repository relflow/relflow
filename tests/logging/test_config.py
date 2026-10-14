import io
import logging
import re
from collections.abc import Iterator
from logging.handlers import BufferingHandler

import pytest

from relflow.logging.config import CONTEXT, Logger, configure


@pytest.fixture
def package_logger() -> Iterator[logging.Logger]:
    core = logging.getLogger("relflow")
    handlers, level, propagate = core.handlers[:], core.level, core.propagate
    core.handlers.clear()
    try:
        yield core
    finally:
        for handler in core.handlers[:]:
            core.removeHandler(handler)
            handler.close()
        core.handlers[:] = handlers
        core.setLevel(level)
        core.propagate = propagate


def test_logger_preserves_bound_context_on_standard_records():
    core = logging.Logger("relflow-test", level=logging.DEBUG)
    records = BufferingHandler(capacity=10)
    core.addHandler(records)
    logger = Logger(core, {}).bind(component="model", rank=2)

    logger.bind(epoch=4).info("started training")

    record = records.buffer[0]
    assert record.getMessage() == "started training"
    assert getattr(record, CONTEXT) == {"component": "model", "rank": 2, "epoch": 4}


def test_logger_merges_call_context_without_mutating_bound_context():
    core = logging.Logger("relflow-test", level=logging.DEBUG)
    records = BufferingHandler(capacity=10)
    core.addHandler(records)
    logger = Logger(core, {}).bind(component="model", rank=2)

    logger.info("batch %d", 4, extra={CONTEXT: {"rank": 3}, "request_id": "abc"})
    logger.info("next batch")

    first, second = records.buffer
    assert first.getMessage() == "batch 4"
    assert first.request_id == "abc"
    assert getattr(first, CONTEXT) == {"component": "model", "rank": 3}
    assert getattr(second, CONTEXT) == {"component": "model", "rank": 2}


def test_configure_writes_plain_messages_and_python_context(package_logger: logging.Logger):
    stream = io.StringIO()
    logger = configure(output=stream)

    logger.bind(component="tensorfield", address="/amount", count=10000, trainable=True, value=None).warning(
        "[red]values exceeded %s[/red]", "the safe range"
    )

    timestamp, _, rendered = stream.getvalue().partition(" ")
    assert re.fullmatch(r"\d{2}:\d{2}:\d{2}", timestamp)
    assert rendered == (
        "WARNING [red]values exceeded the safe range[/red] "
        "{'component': 'tensorfield', 'address': '/amount', 'count': 10000, 'trainable': True, 'value': None}\n"
    )
    assert "\x1b" not in rendered


def test_configure_preserves_standard_exception_tracebacks(package_logger: logging.Logger):
    stream = io.StringIO()
    logger = configure(output=stream).bind(component="model")

    try:
        raise ValueError("invalid observations")
    except ValueError:
        logger.exception("failed batch %d", 4)

    rendered = stream.getvalue().partition(" ")[2]
    assert rendered.startswith("ERROR failed batch 4 {'component': 'model'}\nTraceback (most recent call last):\n")
    assert rendered.endswith("ValueError: invalid observations\n")
    assert "\x1b" not in rendered


def test_configure_replaces_only_its_own_handler(package_logger: logging.Logger):
    records = BufferingHandler(capacity=10)
    package_logger.addHandler(records)
    previous, current = io.StringIO(), io.StringIO()
    configure(level="DEBUG", output=previous)

    logger = configure(level="WARNING", output=current)
    logger.info("filtered")
    logger.warning("logged once")

    assert package_logger.handlers[0] is records
    assert len(package_logger.handlers) == 2
    assert previous.getvalue() == ""
    assert current.getvalue().partition(" ")[2] == "WARNING logged once\n"
    assert [record.getMessage() for record in records.buffer] == ["logged once"]


def test_configure_uses_current_stderr(package_logger: logging.Logger, monkeypatch: pytest.MonkeyPatch):
    stream = io.StringIO()
    monkeypatch.setattr("sys.stderr", stream)

    configure().info("current stream")

    assert stream.getvalue().partition(" ")[2] == "INFO current stream\n"


def test_logger_rejects_non_mapping_context():
    logger = Logger(logging.Logger("relflow-test", level=logging.DEBUG), {})
    with pytest.raises(TypeError, match="relflow_context must be a mapping, got str"):
        logger.info("invalid context", extra={CONTEXT: "invalid"})


@pytest.mark.parametrize("level", ["verbose", "WARN", "NOTSET", True, object()])
def test_configure_rejects_invalid_levels(level: object):
    with pytest.raises((TypeError, ValueError), match="RelFlow log level"):
        configure(level=level)
