import pydantic
import pytest

import relflow as rf
from relflow.structs.mixer import Conv


def test_mixer_settings_are_frozen_tagged_values():
    assert rf.Transformer().model_dump() == {"type": "transformer"}
    assert Conv().model_dump() == {"type": "conv", "n_blocks": 6}
    for invalid in (0, True, 1.5):
        with pytest.raises(pydantic.ValidationError):
            Conv(n_blocks=invalid)
