"""Retained-column selections follow one contract across data and prediction."""

import pyarrow as pa
import pytest
from tensordict import TensorDict

import relflow as rf
from relflow.data.arrow import Encoded


@pytest.mark.parametrize(
    "retain,error,message",
    [
        ("id", TypeError, "tuple of column names"),
        (["id"], TypeError, "tuple of column names"),
        (("",), TypeError, "non-empty strings"),
        ((1,), TypeError, "non-empty strings"),
        (([],), TypeError, "non-empty strings"),
        (("id", "id"), ValueError, "must be unique"),
    ],
)
def test_invalid_retention_is_rejected_at_each_boundary(retain, error, message):
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, amount=rf.Number, label=rf.Boolean(mask=True))
    source = pa.table({"id": pa.array([], type=pa.int64()), "amount": pa.array([], type=pa.float64())})

    with pytest.raises(error, match=f"Encoded retain.*{message}"):
        Encoded(tensors=TensorDict({}, batch_size=[0]), source=source, retain=retain)
    with pytest.raises(error, match=f"retain for validate.*{message}"):
        rf.ArrowDataModule(model=model, train=source, validate=source, retain={"train": (), "validate": retain})
    with pytest.raises(error, match=f"retain.*{message}"):
        model.write([], source=source, retain=retain)
