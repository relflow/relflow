import pydantic
import pytest

import relflow as rf
from relflow.structs.enums import Overflow
from relflow.structs.experiment import Schema
from relflow.tensorfields.base import TENSORFIELDS

ADDRESS = "/items/signal"


def build_schema(**options) -> Schema:
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        items=rf.Branch(length=2, signal=rf.Series(**options)),
    )


def test_series_request_defaults():
    request = build_schema(channels=2).requests[ADDRESS]
    assert (request.type, request.length, request.overflow, request.horizon, request.n_blocks) == (
        "series",
        None,
        None,
        None,
        6,
    )


def test_series_capped_length_defaults_overflow_to_head():
    assert build_schema(channels=1, length=8).requests[ADDRESS].overflow == Overflow.head


def test_series_uncapped_rejects_overflow():
    with pytest.raises(ValueError, match="cannot configure overflow"):
        rf.Series(channels=1, overflow="tail")


@pytest.mark.parametrize(
    "option",
    [{"channels": 0}, {"channels": 1, "length": 0}, {"channels": 1, "horizon": 0}, {"channels": 1, "n_blocks": 0}],
)
def test_series_rejects_non_positive_sizes(option):
    with pytest.raises(pydantic.ValidationError):
        rf.Series(**option)


def test_series_is_exported():
    assert rf.Series is TENSORFIELDS["series"].Request
