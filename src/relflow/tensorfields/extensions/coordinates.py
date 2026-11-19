"""Geographic coordinates represented and scored on a spherical Earth."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import TYPE_CHECKING, ClassVar, Literal, Self, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from tensordict import TensorDict, tensorclass

from relflow.data.query import Projection
from relflow.data.ragged import RaggedField, selected
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import Context, Extension, TensorFieldBase, TensorInput
from relflow.tensorfields.extensions import number, vector
from relflow.tensorfields.output import array, fixed, struct
from relflow.tensorfields.shared.jitter import Jitter

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.ragged import Sources
    from relflow.structs.experiment import Schema

EARTH_RADIUS = 6_371_009.0  # Mean Earth radius in meters.
coordinates = Extension.inherit(vector.Request, name="coordinates", queries=(1, 2))


@coordinates.register(Component.Request)
class Request(vector.Request):
    """Latitude and longitude in degrees, modeled with spherical distance.

    Supply one ``[latitude, longitude]`` pair or use
    ``query=(latitude_path, longitude_path)`` for separate scalars. Latitude
    ranges from -90 to 90 and longitude from -180 to 180. A null component
    makes the whole position null; NaN or infinity makes it ``other``.

    Reconstruction minimizes great-circle distance scaled by the observed
    target spread. Reported MAE and RMSE are great-circle distances in meters,
    using a mean spherical Earth radius; they do not account for roads, terrain,
    or elevation.
    """

    # A geographic position fixes the parent's width and objective.
    type: Literal["coordinates"] = "coordinates"  # pyrefly: ignore [bad-override-mutable-attribute]
    n_dim: Literal[2] = 2  # pyrefly: ignore [bad-override-mutable-attribute]
    objective: Literal["great_circle"] = "great_circle"  # pyrefly: ignore [bad-override-mutable-attribute]

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            jitter: Jitter = ...,
            type: Literal["coordinates"] = "coordinates",
            **options: Unpack[FieldOptions],
        ) -> None: ...


def values(inputs: tuple[Projection[pa.Array | pa.ChunkedArray], ...], *, rows: int, address: Address) -> Projection:
    """Validate and combine sources before masking can hide invalid coordinates."""
    sources = [selected(item.values, item.present, rows=rows, address=address) for item in inputs]
    present: pa.Array | pa.ChunkedArray = pa.repeat(pa.scalar(True), rows)
    masked: list[pa.Array | pa.ChunkedArray] = []
    for source in sources:
        available = source.present if source.present.type == pa.bool_() else pc.is_valid(source.present)
        present = pc.and_(present, available)
        value = source.values
        if pa.types.is_dictionary(value.type):
            value = pc.dictionary_decode(value)  # pyrefly: ignore [missing-attribute]
        masked.append(pc.if_else(available, value, None))
    masked = [pc.if_else(present, value, None) for value in masked]

    if len(masked) == 1:
        pairs = masked[0]
        if pa.types.is_null(pairs.type):
            lat = lon = pc.cast(pairs, pa.float64())
        else:
            if not (
                pa.types.is_list(pairs.type)
                or pa.types.is_large_list(pairs.type)
                or pa.types.is_fixed_size_list(pairs.type)
            ):
                raise ValueError(f"Coordinates field at '{address}' expects [latitude, longitude], got {pairs.type}")
            lengths = pc.list_value_length(pairs)
            if pc.any(pc.not_equal(lengths, 2)).as_py():  # pyrefly: ignore [no-matching-overload]
                raise ValueError(f"Coordinates field at '{address}' expects pairs of [latitude, longitude]")
            lat, lon = pc.list_element(pairs, 0), pc.list_element(pairs, 1)
    else:
        lat, lon = masked

    result: list[pa.Array] = []
    for name, column, limit in (("latitude", lat, 90), ("longitude", lon, 180)):
        if pa.types.is_dictionary(column.type):
            column = pc.dictionary_decode(column)  # pyrefly: ignore [missing-attribute]
        if not (pa.types.is_null(column.type) or pa.types.is_integer(column.type) or pa.types.is_floating(column.type)):
            raise ValueError(f"Coordinates field at '{address}' expects numeric {name}, got {column.type}")
        try:
            column = pc.cast(column, pa.float64(), safe=True)
        except pa.ArrowException as error:
            raise ValueError(f"Coordinates field at '{address}' cannot convert {name} to float64") from error
        invalid = pc.and_(pc.is_finite(column), pc.greater(pc.abs(column), pa.scalar(limit)))
        if pc.any(invalid).as_py():
            bad = pc.filter(column, invalid)[0].as_py()
            raise ValueError(
                f"Coordinates field at '{address}' has {name} {bad}; expected degrees in [-{limit}, {limit}]"
            )
        result.append(column.combine_chunks() if isinstance(column, pa.ChunkedArray) else column)

    lat_array, lon_array = result
    valid = pc.and_(pc.is_valid(lat_array), pc.is_valid(lon_array))
    lat_filled = cast(pa.Array, pc.fill_null(lat_array, pa.scalar(0.0)))
    lon_filled = cast(pa.Array, pc.fill_null(lon_array, pa.scalar(0.0)))
    dense = np.column_stack((lat_filled.to_numpy(), lon_filled.to_numpy())).reshape(-1)
    pairs = pa.FixedSizeListArray.from_arrays(pa.array(dense, type=pa.float64()), 2, mask=pc.invert(valid))
    return Projection(pairs, present)


@coordinates.register(Component.TensorField)
@tensorclass
class TensorField(vector.TensorField):
    dtype: ClassVar[type[np.float32] | type[np.float64]] = np.float64

    @classmethod
    def new(
        cls,
        sources: Sources,
        *,
        schema: Schema,
        context: Context,
    ) -> tuple[Self, TensorDict | None, tuple[object, ...]]:
        if not sources.vacant:
            selected = values(sources.inputs, rows=sources.rows, address=sources.address)
            sources = replace(sources, inputs=(selected,))
        return super().new(sources, schema=schema, context=context)


def unit(position: torch.Tensor) -> torch.Tensor:
    """Map degrees to a unit vector, making the date line continuous."""
    latitude, longitude = (position * (math.pi / 180)).unbind(dim=-1)
    return torch.stack((latitude.cos() * longitude.cos(), latitude.cos() * longitude.sin(), latitude.sin()), dim=-1)


def angle(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """Great-circle angle with finite gradients at coincident points and antipodes."""
    predicted, observed = unit(predicted), unit(observed)
    cross = torch.linalg.vector_norm(torch.linalg.cross(predicted, observed, dim=-1), dim=-1)
    dot = (predicted * observed).sum(dim=-1)
    return torch.atan2(cross, dot)


@coordinates.register(Component.observe)
def observe(
    field: RaggedField, *, address: Address, schema: Schema, state: object | None, learn: bool
) -> TensorDict | None:
    """Summarize unmasked positions in Cartesian space, including across the date line."""
    if not learn:
        return None
    pairs = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
    if pa.types.is_null(pairs.type):
        points = torch.empty((0, 2), dtype=torch.float64)
    else:
        values = pc.list_flatten(pairs).to_numpy(zero_copy_only=False).reshape(-1, 2).copy()
        points = torch.from_numpy(values)
        points = points[torch.isfinite(points).all(dim=-1)]
    cartesian = unit(points)
    return TensorDict({str(index): number.moments(cartesian[:, index]) for index in range(3)}, batch_size=[])


@coordinates.register(Component.Embedder)
class Embedder(vector.Embedder):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        self.linear[0] = torch.nn.Linear(in_features=3, out_features=schema.d_model)

    def forward(self, inputs: TensorInput) -> Parcel:
        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = cast(torch.Tensor, inputs.content).reshape(-1, 2)
        if self.training:
            content = self.jitter.apply(content, state.eq(Tokens.valued).unsqueeze(-1) & torch.isfinite(content))
        position = unit(content).masked_fill(~state.eq(Tokens.valued).unsqueeze(-1), 0)
        position = position.to(dtype=cast(torch.nn.Linear, self.linear[0]).weight.dtype)
        projected = self.linear(position) + self.embeddings(state)
        return Parcel(
            payload=projected.reshape(N, *dims, -1),
            present=torch.ones(N, dtype=torch.bool, device=projected.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@coordinates.register(Component.Decoder)
class Decoder(vector.Decoder):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)
        self.regression = torch.nn.Linear(in_features=schema.d_model, out_features=3)
        self.normalizers = torch.nn.ModuleList(number.GlobalOnlineNormalizer() for _ in range(3))

    @property
    def spread(self) -> torch.Tensor:
        """A common scale preserves spherical geometry near narrow target clusters."""
        return (
            torch.stack([cast(number.GlobalOnlineNormalizer, item).var for item in self.normalizers])
            .sum()
            .sqrt()
            .clamp_min(1e-9)
        )

    def decode(self, pooled: torch.Tensor) -> TensorDict:
        # Coordinate readout in bf16 loses kilometers of precision even when
        # the rest of the model benefits from mixed-precision computation.
        with number.precise(pooled.device):
            payload = super().decode(pooled.float())
            direction = cast(torch.Tensor, payload[TensorKey.content])
            center = torch.cat([cast(number.GlobalOnlineNormalizer, item).mean for item in self.normalizers])
            center = torch.where(center.norm() < 1e-12, center.new_tensor([1.0, 0.0, 0.0]), center)
            direction = center + self.spread * direction.to(dtype=center.dtype)
            x, y, z = direction.unbind(dim=-1)
            x = torch.where((x == 0) & (y == 0), torch.full_like(x, 1e-15), x)
            latitude = torch.atan2(z, torch.linalg.vector_norm(torch.stack((x, y), dim=-1), dim=-1))
            longitude = torch.atan2(y, x)
            payload[TensorKey.content] = torch.stack((latitude, longitude), dim=-1) * (180 / math.pi)
        return payload


@coordinates.register(Component.learn)
def learn(module: Model, observation: TensorDict, *, address: Address, strata: Strata) -> None:
    if strata != Strata.train:
        raise ValueError(f"Coordinates learner at '{address}' requires train strata, got {strata}")
    if address not in module.schema.objectives:
        return
    decoder = cast(Decoder, module.nodes[address].decoder)
    for index, normalizer in enumerate(decoder.normalizers):
        cast(number.GlobalOnlineNormalizer, normalizer).learn(cast(TensorDict, observation[str(index)]))


@coordinates.register(Component.loss)
def loss(module: Model, prediction: Prediction, batch: TensorFieldBase, strata: Strata) -> torch.Tensor:
    address = prediction.address
    decoder = cast(Decoder, module.nodes[address].decoder)
    trainable = batch.trainable.reshape(-1)
    states = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(-1)
    logits = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))
    selected_states = logits[trainable]
    selected_targets = states[trainable]
    if selected_targets.numel():
        state_loss = torch.nn.functional.cross_entropy(selected_states, selected_targets)
        module.track(
            (address, strata, Metric.accuracy, TensorKey.state),
            value=selected_states.argmax(dim=1).eq(selected_targets).float().mean(),
        )
    else:
        state_loss = logits.sum() * 0
    output = module.track((address, strata, Metric.loss, TensorKey.state), value=state_loss)

    valued = trainable & states.eq(Tokens.valued.value)
    inputs = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(-1, 2)[valued]
    targets = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(-1, 2)[valued]
    with number.precise(inputs.device):
        separation = angle(inputs.to(number.wide(inputs.device)), targets.to(number.wide(inputs.device)))
    if separation.numel():
        output = output + module.track(
            (address, strata, Metric.loss, TensorKey.content), value=separation.mean() / decoder.spread
        ).to(output)
    else:
        output = output + cast(torch.Tensor, prediction.payload[TensorKey.content]).sum() * 0

    distances = separation.detach() * EARTH_RADIUS
    counts = torch.full((1,), distances.numel(), dtype=torch.int64, device=distances.device)
    decoder.metrics.record(module, strata, counts, torch.stack((distances.sum(), distances.square().sum())))
    return output


@coordinates.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    content = pa.list_(pa.field("element", pa.float64()), 2)
    return pa.struct([pa.field(TensorKey.content.name, content, nullable=False)])


@coordinates.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content = cast(torch.Tensor, prediction.payload[TensorKey.content]).detach().clone()
    non_valued = cast(torch.Tensor, prediction.payload[TensorKey.state]).argmax(dim=-1).ne(Tokens.valued.value)
    content[non_valued] = 0.0
    values = fixed(array(content, pa.float64()), 2, datatype=datatype.field(TensorKey.content.name).type)
    return struct({TensorKey.content.name: values}, datatype)
