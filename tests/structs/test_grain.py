import pydantic
import pytest

import relflow as rf
from relflow.structs.grain import Steps


def test_grain_settings_are_frozen_tagged_values():
    assert rf.Coordinates().model_dump() == {"type": "coordinates"}
    assert Steps().model_dump() == {"type": "steps", "pooling": "query"}
    with pytest.raises(pydantic.ValidationError):
        Steps(pooling="max")
    with pytest.raises(pydantic.ValidationError, match="frozen"):
        Steps().pooling = "mean"


def test_step_pooling_uses_exported_string_enum():
    assert rf.Pooling.query.value == "query"
    assert Steps().pooling is rf.Pooling.query
    assert Steps(pooling="mean").pooling is rf.Pooling.mean
