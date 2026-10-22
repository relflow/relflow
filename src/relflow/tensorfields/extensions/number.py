from __future__ import annotations

import enum
import math
from collections.abc import Iterator
from contextlib import AbstractContextManager, nullcontext
from dataclasses import InitVar
from typing import TYPE_CHECKING, Annotated, Literal, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass
from torchmetrics import Metric as TorchMetric
from torchmetrics.regression import MeanAbsoluteError, MeanSquaredError

from relflow.data.ragged import RaggedField
from relflow.distributed import all_reduce_sum
from relflow.helpers import Jitter
from relflow.logging import logger
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.output import array, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


number: Extension = Extension(name="number", types=(int | float,))
number.callback(CounterUpdateCallback)

FOURIER_SAFE_MAX_ANGLE = float(1e4)

Lane = Literal["raw", "log"]
LANES: tuple[Lane, ...] = ("raw", "log")


def precise(device: torch.device) -> AbstractContextManager[object]:
    """Disable mixed precision for source-value arithmetic; bf16 would quantize lanes and readouts."""
    if device.type in {"cuda", "cpu"}:
        return torch.autocast(device_type=device.type, enabled=False)
    return nullcontext()


class Objective(enum.StrEnum):
    mae = "mae"
    mse = "mse"
    huber = "huber"

    def loss(self, *, input: torch.Tensor, target: torch.Tensor, reduction: str = "none") -> torch.Tensor:
        match self:
            case Objective.mae:
                return torch.nn.functional.l1_loss(input=input, target=target, reduction=reduction)
            case Objective.mse:
                return torch.nn.functional.mse_loss(input=input, target=target, reduction=reduction)
            case Objective.huber:
                return torch.nn.functional.huber_loss(input=input, target=target, reduction=reduction)


@number.register(Component.Request)
class Request(RequestBase):
    """Continuous numeric scalar with raw and signed-log lanes and a standardized readout.

    Inputs embed two standardized lanes: the source value and its signed log
    scale ``sign(x) * ln(1 + |x|)``. Each lane keeps its own online statistics,
    Fourier features, and monotone channel, so additive structure reads from
    the raw lane and multiplicative or heavy-tailed structure from the log lane
    without a per-field choice. The embedder and the readout run in float32
    under mixed precision so bf16 never quantizes source values.

    ``objective`` selects the reconstruction loss. ``"mae"`` trains on the
    signed log scale, where the median-optimal prediction is unchanged, and
    adds the source-unit absolute error so the bulk stays sharp; ``"mse"`` and
    ``"huber"`` train in source units because their optimum is not
    transform-invariant. The readout starts at the training mean of that
    scale and predicts standardized deviations, so targets far from zero or
    with heavy tails need no option. Predictions and error metrics return to
    source units. ``alpha=None`` uses cumulative statistics; a value in
    ``(0, 1)`` uses exponential updates. ``n_bands`` and ``offset`` bound
    Fourier exponents. ``jitter=rf.Jitter(...)`` perturbs training inputs once
    in source units while retaining pristine targets. ``mask=True`` makes the
    scalar a supervised, unembedded target.
    """

    type: Literal["number"] = "number"
    jitter: Jitter = pydantic.Field(default_factory=Jitter)
    n_bands: Annotated[int, pydantic.Field(gt=0, default=8)] = 8
    offset: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    alpha: Annotated[float | None, pydantic.Field(gt=0.0, lt=1.0, default=None)] = None
    objective: Objective = Objective.mae

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            jitter: Jitter = ...,
            n_bands: int = 8,
            offset: int = 4,
            alpha: float | None = None,
            objective: Objective | Literal["mae", "mse", "huber"] = Objective.mae,
            type: Literal["number"] = "number",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @property
    def log(self) -> bool:
        """Whether the target trains on the signed log scale: the median-preserving ``"mae"`` objective does."""
        return self.objective == Objective.mae

    @classmethod
    def normalization(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[str, float | int | None]:
        """Return a CPU-native snapshot of one Number field's source-unit normalizer."""
        from relflow.architecture.root import Model

        if not isinstance(model, Model):
            raise TypeError(f"Number.normalization model must be a Model, got {type(model).__name__}")

        address = Address(str(address))
        if address not in model.nodes:
            raise KeyError(f"no field at address {str(address)!r}")

        embedder = getattr(model.nodes[address], "embedder", None)
        if not isinstance(embedder, Embedder):
            raise TypeError(f"address {str(address)!r} is not a Number field (got {type(embedder).__name__})")

        normalizer = cast(GlobalOnlineNormalizer, embedder.normalizers["raw"])
        mean = float(normalizer.mean.detach().cpu().item())
        variance = float(normalizer.var.detach().cpu().item())
        count = None if normalizer.alpha is not None else int(normalizer.count.detach().cpu().item())

        return {
            "mean": mean,
            "variance": variance,
            "std": math.sqrt(variance + normalizer.epsilon),
            "count": count,
            "alpha": normalizer.alpha,
        }


def compress(values: torch.Tensor) -> torch.Tensor:
    """Signed log scale ``sign(x) * ln(1 + |x|)``: the identity near zero and monotone everywhere."""
    return values.sign() * values.abs().log1p()


def expand(values: torch.Tensor) -> torch.Tensor:
    """Invert ``compress`` back to source units."""
    return values.sign() * values.abs().expm1()


# Log-unit scores beyond this magnitude would overflow float32 source units; clamp before inverting.
LOG_SAFE_MAGNITUDE: float = 80.0


@number.register(Component.observe)
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Summarize complete pristine numeric state and content."""

    if not learn:
        return None
    values = pc.cast(field.values, pa.float32(), safe=False)
    if isinstance(values, pa.ChunkedArray):
        values = values.combine_chunks()
    content = torch.from_numpy(values.to_numpy(zero_copy_only=False).copy())
    # Both scales keep statistics: lanes read their own, and a log target scores on its own.
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: moments(content),
            "log": moments(compress(content)),
        },
        batch_size=[],
    )


@number.register(Component.TensorField)
@tensorclass
class TensorField(TensorFieldBase[torch.Tensor]):
    content: torch.Tensor
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    if TYPE_CHECKING:
        # TensorClass metadata is accepted by its generated initializer.
        batch_size: InitVar[int | torch.Size | list[int] | tuple[int, ...] | None] = None
        device: InitVar[torch.device | str | int | None] = None
        names: InitVar[list[str | None] | None] = None

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorField:
        def encode(field: RaggedField) -> torch.Tensor:
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            if (
                pa.types.is_list(values.type)
                or pa.types.is_large_list(values.type)
                or pa.types.is_fixed_size_list(values.type)
            ):
                raise ValueError(f"number field at '{address}' expects scalar Arrow values, got {values.type}")

            # Integer conversion retains Arrow's existing float64 safety range.
            datatype = pa.float32() if pa.types.is_floating(values.type) else pa.float64()
            encoded = pc.cast(values, datatype, safe=True)
            if isinstance(encoded, pa.ChunkedArray):
                encoded = encoded.combine_chunks()
            data = field.place(encoded.to_numpy(zero_copy_only=False).astype(np.float32, copy=False), fill=0.0)
            np.nan_to_num(data, copy=False, nan=0.0, posinf=np.inf, neginf=-np.inf)
            return torch.from_numpy(data)

        state = torch.from_numpy(input.dense)

        return cls(
            content=encode(input),
            state=state,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: encode(target),
                }
                if address in schema.objectives
                else {},
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


def moments(values: torch.Tensor) -> TensorDict:
    """Summarize finite values with an exact count and centered float32 moments."""

    finite = values.reshape(-1).to(dtype=torch.float32)
    finite = finite.masked_select(torch.isfinite(finite))
    if finite.numel():
        variance, mean = torch.var_mean(finite, correction=0)
    else:
        variance, mean = finite.new_zeros(()), finite.new_zeros(())
    return TensorDict(
        {
            "count": torch.tensor(finite.numel(), dtype=torch.int64, device=values.device),
            "mean": mean,
            "var": variance,
        },
        batch_size=[],
    )


class GlobalOnlineNormalizer(torch.nn.Module):
    def __init__(self, alpha: float | None = None, epsilon: float = 1e-5):
        super().__init__()

        self.epsilon: float = epsilon
        self.alpha: float | None = alpha

        self.register_buffer("mean", torch.zeros(1, dtype=torch.float32))
        self.register_buffer("var", torch.ones(1, dtype=torch.float32))
        self.register_buffer("count", torch.zeros(1, dtype=torch.int64))

    @torch.no_grad()
    def update(self, values: torch.Tensor):
        self.merge(moments(values))

    @torch.no_grad()
    def learn(self, observation: TensorDict) -> None:
        """Synchronize and apply one pristine numeric observation."""

        if not isinstance(observation, TensorDict):
            raise TypeError(f"normalizer observation must be a TensorDict, got {type(observation).__name__}")
        observation = observation.to(device=self.mean.device)
        local_count = cast(torch.Tensor, observation["count"])
        local_mean = cast(torch.Tensor, observation["mean"])
        local_var = cast(torch.Tensor, observation["var"])
        count = all_reduce_sum(local_count.clone())
        if not count:
            return
        weight = local_count.to(dtype=torch.float32) / count
        mean = all_reduce_sum(local_mean * weight)
        # Center on the shared mean before reducing variance. Raw squared sums
        # lose small variances at large offsets in float32; empty ranks contribute zero.
        delta = torch.where(local_count > 0, local_mean - mean, 0.0)
        variance = all_reduce_sum((local_var + delta.square()) * weight)
        self.merge(TensorDict({"count": count, "mean": mean, "var": variance}, batch_size=[]))

    @torch.no_grad()
    def merge(self, observation: TensorDict) -> None:
        """Merge centered moments into running statistics without retaining values."""

        count = cast(torch.Tensor, observation["count"])
        mean = cast(torch.Tensor, observation["mean"])
        variance = cast(torch.Tensor, observation["var"])
        if not all(torch.isfinite(value).all() for value in (count, mean, variance)):
            raise ValueError("normalizer observation must contain only finite values")
        if count < 0:
            raise ValueError("normalizer observation count cannot be negative")
        if variance < 0:
            raise ValueError("normalizer observation variance cannot be negative")

        batch_count = count.to(device=self.count.device, dtype=self.count.dtype)
        if not batch_count:
            return

        batch_mean = mean.to(self.mean)
        batch_var = variance.to(self.var)

        if self.alpha is not None:
            alpha: float = self.alpha
            new_mean = (1 - alpha) * self.mean + alpha * batch_mean
            new_var = (1 - alpha) * self.var + alpha * batch_var

            self.mean = new_mean
            self.var = new_var

            return

        new_count = self.count + batch_count
        weight = batch_count.to(dtype=self.mean.dtype) / new_count
        delta = batch_mean - self.mean
        correction = delta * torch.sqrt((self.count.to(dtype=self.mean.dtype) / new_count) * weight)
        self.mean = torch.lerp(self.mean, batch_mean, weight)
        self.var = torch.lerp(self.var, batch_var, weight) + correction.square()
        self.count = new_count

    def forward(self, inputs: torch.Tensor, mask: torch.Tensor, update=True) -> torch.Tensor:
        finite_mask = mask & torch.isfinite(inputs)
        if self.training and update:
            self.update(inputs[finite_mask])

        std = torch.sqrt(self.var + self.epsilon)
        normalized = (torch.atleast_1d(inputs) - self.mean) / std
        return torch.where(finite_mask, normalized.reshape_as(inputs), inputs)


@number.register(Component.Embedder)
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address

        self.embeddings = torch.nn.Embedding(num_embeddings=len(Tokens), embedding_dim=schema.d_model)
        self.counter = Counter(address=address, size=len(Tokens))

        n_bands = request.n_bands
        offset = request.offset
        if schema.d_model < len(LANES):
            raise ValueError(
                f"number field at '{address}' reserves one payload channel per lane; "
                f"d_model={schema.d_model} cannot hold {len(LANES)} lanes"
            )

        weights = torch.logspace(start=-n_bands, end=offset, steps=n_bands + offset + 1, base=2)
        self.linear = torch.nn.Linear(2 * len(weights) * len(LANES), schema.d_model)
        self.scalar = torch.nn.Linear(len(LANES), schema.d_model, bias=False)
        self.register_buffer("weights", weights.mul(math.pi).unsqueeze(dim=0))
        self.register_buffer("max_fourier_input", torch.tensor(FOURIER_SAFE_MAX_ANGLE) / self.weights.abs().max())

        self.jitter: Jitter = request.jitter
        self.weights: torch.Tensor
        self.max_fourier_input: torch.Tensor

        self.normalizers = torch.nn.ModuleDict({name: GlobalOnlineNormalizer(alpha=request.alpha) for name in LANES})

    def clamp(self, content: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        bound = self.max_fourier_input.to(device=content.device, dtype=content.dtype)
        valued = state.eq(Tokens.valued)
        finite = torch.isfinite(content)
        out_of_range = content.abs().gt(bound)
        unsafe = ~finite | out_of_range

        if unsafe.any():
            unsafe_values = content[unsafe].detach()
            unsafe_abs = unsafe_values.abs()
            finite_abs = unsafe_abs[torch.isfinite(unsafe_abs)]
            if torch.isinf(unsafe_abs).any():
                max_abs = math.inf
            elif finite_abs.numel() > 0:
                max_abs = float(finite_abs.max().cpu().item())
            else:
                max_abs = math.nan

            logger.bind(
                component="tensorfield",
                field_type="number",
                address=str(self.origin),
                count=int(unsafe.sum().cpu().item()),
                valued_count=int((unsafe & valued).sum().cpu().item()),
                nonfinite_count=int((~torch.isfinite(unsafe_values)).sum().cpu().item()),
                max_abs_normalized=max_abs,
                bound=float(bound.detach().cpu().item()),
                safe_max_angle=FOURIER_SAFE_MAX_ANGLE,
            ).warning("number Fourier inputs exceed safe range; clamping normalized values")

        clamped = content.clamp(min=-bound, max=bound)
        return torch.where(torch.isnan(clamped), torch.zeros_like(clamped), clamped)

    def perturb(self, content: torch.Tensor, eligible: torch.Tensor) -> torch.Tensor:
        """Jitter the measurement once, in source units, so every lane sees the same value.

        ``normalize=True`` sizes the perturbation in standardized raw units and
        returns to source units before the lanes derive their own scales.
        """
        if not self.jitter.normalize:
            return self.jitter.apply(content, eligible)
        raw = cast(GlobalOnlineNormalizer, self.normalizers["raw"])
        std = torch.sqrt(raw.var + raw.epsilon).to(dtype=content.dtype)
        mean = raw.mean.to(dtype=content.dtype)
        standardized = self.jitter.apply((content - mean) / std, eligible)
        return torch.where(eligible, standardized * std + mean, content)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        with precise(inputs.state.device):
            return self.project(inputs)

    def project(self, inputs: TensorInput) -> Parcel:
        """Embed both lanes of every value with Fourier features and monotone channels, in float32."""
        N, *dims = inputs.state.shape
        D = math.prod(tuple([N, *dims]))

        state = inputs.state.reshape(D)
        content = cast(torch.Tensor, inputs.content).reshape(D).float()
        valued = state.eq(Tokens.valued)
        eligible = valued & torch.isfinite(content)

        if self.training:
            content = self.perturb(content, eligible)

        lanes: list[torch.Tensor] = []
        for name in LANES:
            values = compress(content) if name == "log" else content
            values = cast(GlobalOnlineNormalizer, self.normalizers[name])(inputs=values, mask=valued, update=False)
            lanes.append(self.clamp(content=values, state=state))
        standardized = torch.stack(lanes, dim=1)

        weighted = standardized.unsqueeze(dim=2).mul(self.weights)
        fourier = torch.cat([torch.sin(weighted), torch.cos(weighted)], dim=2).reshape(D, -1)

        projection = torch.nn.functional.gelu(self.linear(fourier), approximate="tanh").reshape(N, *dims, -1)

        embeddings = self.embeddings(state).reshape(N, *dims, -1).float()

        payload = embeddings + projection + self.scalar(standardized).reshape(N, *dims, -1)
        # Keep one clamped monotone channel per lane alongside the Fourier
        # features. This gives downstream additive and multiplicative
        # interactions direct access to numerical scale instead of forcing them
        # to invert a periodic basis outside the values observed during fitting.
        width = len(LANES)
        monotone = torch.where(
            eligible.reshape(N, *dims, 1),
            standardized.reshape(N, *dims, width),
            embeddings[..., -width:],
        )
        monotone = monotone + payload[..., -width:] * 0.0
        payload = torch.cat((payload[..., :-width], monotone), dim=-1)

        return Parcel(
            payload=payload,
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )


@number.register(Component.learn)
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine numeric counts and normalization moments."""

    if strata != Strata.train:
        raise ValueError(f"number learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    embedder.counter.learn(cast(torch.Tensor, observation[TensorKey.state]))
    cast(GlobalOnlineNormalizer, embedder.normalizers["raw"]).learn(cast(TensorDict, observation[TensorKey.content]))
    cast(GlobalOnlineNormalizer, embedder.normalizers["log"]).learn(cast(TensorDict, observation["log"]))
    if address in module.schema.objectives:
        decoder: Decoder = cast(Decoder, module.nodes[address].decoder)
        decoder.normalizer.learn(cast(TensorDict, observation["log" if decoder.log else TensorKey.content]))


@number.register(Component.Decoder)
class Decoder(DecoderBase):
    coordinatewise = True

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.classification = torch.nn.Linear(in_features=schema.d_model, out_features=len(Tokens))
        self.regression = torch.nn.Linear(in_features=schema.d_model, out_features=1)
        self.log: bool = request.log
        # Target statistics in the training scale: the readout starts at their
        # mean and predicts standardized deviations.
        self.normalizer: GlobalOnlineNormalizer = GlobalOnlineNormalizer(alpha=request.alpha)
        self.metrics = torch.nn.ModuleDict(
            {
                f"{strata.value}_metrics": torch.nn.ModuleDict(
                    {Metric.mae.value: MeanAbsoluteError(), Metric.rmse.value: MeanSquaredError(squared=False)}
                )
                for strata in (Strata.train, Strata.validate, Strata.test)
            }
        )

    def content_metrics(self, strata: Strata) -> Iterator[tuple[str, TorchMetric]]:
        for name, metric in cast(torch.nn.ModuleDict, self.metrics[f"{strata.value}_metrics"]).items():
            yield name, cast(TorchMetric, metric)

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        with precise(pooled.device):
            deviation = self.regression(pooled.float())
            std = torch.sqrt(self.normalizer.var + self.normalizer.epsilon)
            scaled = self.normalizer.mean + std * deviation
            content = scaled
            if self.log:
                # Content stays in source units for every consumer; the loss reads the training-scale lane.
                content = expand(scaled.detach().clamp(-LOG_SAFE_MAGNITUDE, LOG_SAFE_MAGNITUDE).to(dtype=torch.float64))
                content = content.to(dtype=scaled.dtype)
        return TensorDict(
            source={
                TensorKey.state: self.classification(pooled),
                TensorKey.content: content,
                "scaled": scaled,
            }
        )


# The source-unit term of the ``"mae"`` loss ignores predictions this many
# training-scale deviations past the mean, so its exponential gradient stays bounded.
SOURCE_TERM_SPREAD: float = 6.0


@number.register(Component.loss)
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    address: Address = prediction.address
    request: Request = cast(Request, module.schema.requests[prediction.address])

    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    decoder: Decoder = cast(Decoder, module.nodes[address].decoder)

    N: int = cast(torch.Tensor, batch.targets[TensorKey.state]).numel()

    trainable: torch.Tensor = batch.trainable.reshape(N)
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(N)

    loss: torch.Tensor = module.track(
        (address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(N, -1),
                target=state_targets,
                weight=embedder.counter.weight,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    target: torch.Tensor = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(N)
    scaled: torch.Tensor = cast(torch.Tensor, prediction.payload["scaled"]).reshape(N)
    valued = trainable & state_targets.eq(Tokens.valued.value)
    actual = scaled.masked_select(valued).float()
    observed = target.masked_select(valued).float()
    if actual.numel():
        expected = compress(observed) if request.log else observed
        std = decoder.normalizer.var.sqrt().clamp_min(decoder.normalizer.epsilon)
        value = request.objective.loss(
            input=(actual - expected) / std,
            target=torch.zeros_like(actual),
            reduction="mean",
        )
        if request.log:
            # The median is invariant to the log scale, so the source-unit error
            # sharpens the bulk without moving the optimum.
            raw = cast(GlobalOnlineNormalizer, embedder.normalizers["raw"])
            spread = SOURCE_TERM_SPREAD * std
            bounded = actual.clamp(min=decoder.normalizer.mean - spread, max=decoder.normalizer.mean + spread)
            value = value + (expand(bounded) - observed).abs().div(raw.var.sqrt().clamp_min(raw.epsilon)).mean()
        loss += module.track((address, strata, Metric.loss, TensorKey.content), value=value)
    else:
        loss += scaled.sum() * 0.0

    # Error metrics report source units whatever the training scale.
    content = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(N)
    predicted = content.masked_select(valued).float()
    for name, metric in decoder.content_metrics(strata):
        if strata == Strata.train:
            metric.reset()
            value = metric(predicted, observed) if predicted.numel() else scaled.new_full((), float("nan"))
            module.track((address, strata, name, TensorKey.content), value=cast(torch.Tensor, value))
        else:
            # Every rank registers the same sum/count metrics, including ranks
            # with no valued targets. Global RMSE is not a mean of batch RMSEs.
            metric.update(predicted, observed)
            module.track((address, strata, name, TensorKey.content), value=metric)

    return loss


@number.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    return pa.struct([pa.field(TensorKey.content.name, pa.float64(), nullable=False)])


@number.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content = array(cast(torch.Tensor, prediction.payload[TensorKey.content]), pa.float64())
    return struct({TensorKey.content.name: content}, datatype)
