"""Constructor inputs and normalized schema attributes remain distinct contracts."""

from typing import Literal, assert_type

import pyarrow as pa
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.structs.enums import AttentionMode, Overflow
from relflow.tensorfields.base import Loss, Observe, OutputSchema, Write
from relflow.tensorfields.extensions.number import Objective
from relflow.tensorfields.extensions.period import Part


def schema_api() -> None:
    number = rf.Number(hide=("eligible", 0.2), objective="mse", pooling="mean")
    assert_type(number.predict, bool)
    assert_type(number.hide, rf.HideInput)
    assert_type(number.task, rf.Task | tuple[rf.Task, ...] | None)
    assert_type(rf.reconstruct(fields=4, rate=0.15, ablate=True), rf.Reconstruction)
    assert_type(rf.reconstruct(fields=4).fields, int | None)
    assert_type(number.objective, Objective)
    assert_type(number.n_bands, int)
    quantile = rf.Quantile(compression=200, n_bands=4, objective="mse", predict=True)
    assert_type(quantile.compression, int)
    assert_type(quantile.objective, Literal["mae", "mse", "huber"])
    assert_type(quantile.predict, bool)
    assert_type(rf.Category(topk=[2, 3]).topk, list[int] | None)
    enum = rf.Enum(values=("standard", "express", "pickup"), topk=[2], predict=True)
    assert_type(enum.values, tuple[bool | int | float | str | bytes, ...])
    assert_type(enum.topk, list[int] | None)
    assert_type(enum.predict, bool)
    positional_enum = rf.Enum("standard", "express", "pickup", topk=[2], predict=True)
    assert_type(positional_enum.values, tuple[bool | int | float | str | bytes, ...])
    labels: tuple[str, ...] = ("standard", "express", "pickup")
    rf.Enum(*labels, embed=True)
    rf.Enum(1, 2, 3, predict=True)
    rf.Enum(False, True)
    rf.Enum(b"a", b"b")
    ordinal = rf.Ordinal(values=("low", "medium", "high"), predict=True)
    assert_type(ordinal.values, tuple[bool | int | float | str | bytes, ...] | None)
    assert_type(ordinal.predict, bool)
    rf.Model(preset="small", rank=rf.Ordinal, rating=ordinal)
    assert_type(rf.Group(n_groups=4).n_groups, int)
    assert_type(rf.Group(n_groups=8).n_groups, int)
    assert_type(rf.Period(parts=["day_of_week", Part.hour_of_day]).parts, list[Part])
    assert_type(rf.Period("day_of_week", Part.hour_of_day).parts, list[Part])
    assert_type(rf.Period(parts=("day_of_week", Part.hour_of_day)).parts, list[Part])
    parts: tuple[str | Part, ...] = ("day_of_week", Part.hour_of_day)
    rf.Period(*parts, pattern="%Y-%m-%d", jitter=rf.Jitter(add=0.1), predict=True)
    assert_type(rf.Vector(n_dim=8, objective="l1").n_dim, int)
    rf.Text(encoder_pooling="mean", objective="l2", max_length=64)
    rf.Set(threshold=0.5)
    rf.Hash(n_hashes=2, n_buckets=8)
    rf.Boolean(threshold=[0.3, 0.7], predict=True)
    branch = rf.Branch(
        amount=number,
        ablate=0.25,
        predict=True,
        hide=(0.1, "redacted"),
        ignore="selected",
        task=(rf.reconstruct(rate=0.2, span=3, ignore=("selected",)), rf.reconstruct(query="holdout")),
        length=32,
        overflow="tail",
        attention="gqa",
        reduction=rf.Attention(n_outputs=2),
        kind=rf.Category,
        selected=rf.Boolean(task=rf.reconstruct(query="selected")),
    )
    assert_type(branch.attention, AttentionMode | None)
    assert_type(branch.overflow, Overflow | None)
    assert_type(branch.ablate, float)
    assert_type(branch.predict, bool)
    assert_type(rf.Number(ablate=0.1).ablate, float)
    assert_type(branch.task, rf.Task | tuple[rf.Task, ...] | None)
    rf.Branch(length=3, overflow="random", value=rf.Number)
    rf.Branch(fields={"length": number, "description": rf.Category}, reduction=rf.Mean())
    rf.Model(d_model=32, n_heads=4, n_layers=1, items=branch, label=rf.Boolean(predict=True))
    rf.Model(preset="small", value=rf.Quantile, target=quantile)
    rf.Model(preset="small", target=enum)
    candidate = rf.Model(preset="small", jacobian=True, first=rf.Boolean(predict=True), second=rf.Number(predict=True))
    assert_type(candidate.jacobian, bool)
    rf.Model(d_model=32, n_heads=4, n_layers=1, jacobian=True, label=rf.Boolean(predict=True))
    rf.Model(d_model=32, n_heads=4, n_layers=1, attention=None, items=rf.Branch(attention=None, value=rf.Number))


def group_assignments(model: rf.Model) -> None:
    assert_type(rf.Group.assign(model, "/group", "token", 2), int)
    with rf.Group.override(model, "/group", {"token": 1}):
        pass


def third_party_schema() -> None:
    extension = rf.Extension(name="typing_custom", types=(str,))

    @extension.register(rf.Component.Request)
    class Request(rf.RequestBase):
        """Extension-owned options do not require edits to shared field hints."""

        type: Literal["typing_custom"] = "typing_custom"
        width: int = 4

    request = Request(width=8, predict=True)
    assert_type(request, Request)
    assert_type(request.width, int)
    assert_type(request.predict, bool)
    rf.Branch(custom=request)
    assert_type(extension.loss, Loss)
    assert_type(extension.observe, Observe)
    assert_type(extension.output, OutputSchema)
    assert_type(extension.write, Write)


def field_content(tensor: rf.TensorFieldBase[torch.Tensor], nested: rf.TensorFieldBase[TensorDict]) -> None:
    assert_type(tensor.content, torch.Tensor)
    assert_type(nested.content, TensorDict)


def registration(extension: rf.Extension, writer: Write, objective: Loss) -> None:
    assert_type(extension.register(rf.Component.write, writer), Write)
    assert_type(extension.register("loss", objective), Loss)
    assert_type(extension.register(rf.Component.output, None), None)


def arrow_normalization(extension: rf.Extension, values: pa.Array, chunks: pa.ChunkedArray) -> None:
    assert_type(extension.normalize(values, pa.float32()), pa.Array)
    assert_type(extension.normalize(chunks, pa.float32()), pa.ChunkedArray)
