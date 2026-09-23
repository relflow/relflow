"""Source precision stays on the host while percentile models use accelerators."""

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.structs.enums import TensorKey
from relflow.tensorfields.extensions.quantile import loss


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_quantile_accelerator_keeps_raw_doubles_off_device(device, monkeypatch):
    available = torch.cuda.is_available() if device == "cuda" else torch.backends.mps.is_available()
    if not available:
        pytest.skip(f"{device} is unavailable")
    model = rf.Model.xs(x=rf.Quantile, y=rf.Quantile(mask=True))
    source = pa.table({"x": 1e12 + np.arange(32), "y": 1e12 + np.arange(32) * 3})
    encoded = model.encode(source, strata="train")
    model.to(device)
    batch = encoded.to(device)
    monkeypatch.setattr(model, "track", lambda names, value: value)

    [prediction] = model(batch, strata="train")
    content = prediction.payload[TensorKey.content]
    assert content.device.type == "cpu"
    assert content.dtype == torch.float64
    assert torch.isfinite(content).all()
    assert content.min() >= 1e12
    assert content.max() <= 1e12 + 31 * 3
    objective = loss(model, prediction, batch["/y"], rf.Strata.train)
    assert torch.isfinite(objective)
    objective.backward()
    assert torch.isfinite(model.nodes["/y"].decoder.regression.weight.grad).all()
