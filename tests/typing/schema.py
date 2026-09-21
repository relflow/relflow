"""Constructor inputs and normalized schema attributes remain distinct contracts."""

from typing import Literal, assert_type

import pyarrow as pa
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import AttentionMode, Overflow
from relflow.tensorfields.base import Loss, Observe, OutputSchema, Write
from relflow.tensorfields.extensions.dateparts import DatePart
from relflow.tensorfields.extensions.number import Objective


def schema_api() -> None:
    number = rf.Number(mask=[rf.Mask(rate=0.2)], objective="mse", pooling="mean")
    assert_type(number.mask, tuple[rf.Mask, ...])
    assert_type(number.objective, Objective)
    assert_type(number.n_bands, int)
    quantile = rf.Quantile(compression=200, n_bands=4, objective="mse", mask=True)
    assert_type(quantile.compression, int)
    assert_type(quantile.objective, Literal["mae", "mse", "huber"])
    assert_type(quantile.mask, tuple[rf.Mask, ...])
    assert_type(rf.Category(topk=[2, 3]).topk, list[int] | None)
    assert_type(rf.Cluster(bounds=4).n_clusters, tuple[int, int])
    assert_type(rf.Cluster(n_clusters=(2, 8)).n_clusters, tuple[int, int])
    assert_type(rf.DateParts(dateparts=["day_of_week", DatePart.hour_of_day]).dateparts, list[DatePart])
    assert_type(rf.Vector(n_dim=8, objective="l1").n_dim, int)
    rf.Text(encoder_pooling="mean", objective="l2", max_length=64)
    rf.Set(threshold=0.5)
    rf.Hash(n_hashes=2, n_buckets=8)
    rf.Boolean(threshold=[0.3, 0.7], mask=True)
    branch = rf.Branch(
        amount=number,
        length=32,
        overflow="tail",
        attention="gqa",
        reduction=rf.Attention(n_outputs=2),
        kind=rf.Category,
        selected=rf.Boolean(mask=rf.Mask(query="selected", reconstruct=True)),
    )
    assert_type(branch.attention, AttentionMode | None)
    assert_type(branch.overflow, Overflow)
    assert_type(branch.mask, tuple[rf.Mask, ...])
    rf.Branch(fields={"length": number, "description": rf.Category}, reduction=rf.Mean())
    rf.Model(d_model=32, n_heads=4, n_layers=1, items=branch, label=rf.Boolean(mask=True))
    rf.Model.xs(value=rf.Quantile, target=quantile)
    rf.Model(d_model=32, n_heads=4, n_layers=1, attention=None, items=rf.Branch(attention=None, value=rf.Number))


def third_party_schema() -> None:
    extension = rf.Extension(name="typing_custom", types=(str,))

    @extension.register
    class Request(rf.RequestBase):
        """Extension-owned options do not require edits to shared field hints."""

        type: Literal["typing_custom"] = "typing_custom"
        width: int = 4

    request = Request(width=8, mask=True)
    assert_type(request, Request)
    assert_type(request.width, int)
    assert_type(request.mask, tuple[rf.Mask, ...])
    rf.Branch(custom=request)
    assert_type(extension.loss, Loss)
    assert_type(extension.observe, Observe)
    assert_type(extension.output, OutputSchema)
    assert_type(extension.write, Write)


def field_content(tensor: rf.TensorFieldBase[torch.Tensor], nested: rf.TensorFieldBase[TensorDict]) -> None:
    assert_type(tensor.content, torch.Tensor)
    assert_type(nested.content, TensorDict)


def arrow_normalization(extension: rf.Extension, values: pa.Array, chunks: pa.ChunkedArray) -> None:
    assert_type(extension.normalize(values, pa.float32()), pa.Array)
    assert_type(extension.normalize(chunks, pa.float32()), pa.ChunkedArray)
