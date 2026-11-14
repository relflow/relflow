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
