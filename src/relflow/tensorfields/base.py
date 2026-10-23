"""Tensorfield extension base classes and registry."""

from __future__ import annotations

import importlib.util
import math
import re
import shlex
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from functools import partial
from types import MappingProxyType, UnionType
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Generic,
    Protocol,
    Self,
    TypeAlias,
    TypeVar,
    cast,
    get_args,
    overload,
    runtime_checkable,
)

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import torch
from lightning.pytorch import Callback
from tensordict import TensorClass, TensorDict, TensorDictBase
from typing_extensions import TypeVar as DefaultTypeVar

from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.architecture.rotary import coordinates, flatten, signal
from relflow.data.arrow import Geometry, variants
from relflow.structs.enums import Component, Strata, TensorKey
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, Leaf

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.ragged import RaggedField
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch
    from relflow.tensorfields.shared.resize import Resize

RequestBase: TypeAlias = Leaf
CallbackFactory: TypeAlias = type[Callback] | Callable[[], Callback]
ComponentValue: TypeAlias = Callable[..., Any] | type[Any]
RegisterT = TypeVar("RegisterT", bound=ComponentValue)
OMITTED = object()
ValueTypeFamily: TypeAlias = type[Any] | UnionType
ArrowMatcher: TypeAlias = Callable[[pa.DataType], bool]
ContentT = DefaultTypeVar(
    "ContentT", bound=torch.Tensor | TensorDict, default=torch.Tensor | TensorDict, covariant=True
)


class Observe(Protocol):
    """Summarize pristine Arrow values before masking or augmentation."""

    def __call__(
        self, field: RaggedField, *, address: Address, schema: Schema, state: object | None, learn: bool
    ) -> TensorDict | None: ...


class Learn(Protocol):
    """Apply one field-owned observation to the model's training state."""

    def __call__(self, module: Model, observation: TensorDict, *, address: Address, strata: Strata) -> None: ...


@runtime_checkable
class BatchContext(Protocol):
    """Isolate worker encoding and carry an extension-owned binding to the model.

    The returned pair contains the context used to encode this batch and the
    metadata supplied to the extension's ``bind`` component before forward.
    """

    def batch(self, values: pa.Array | pa.ChunkedArray) -> tuple[object, object]: ...


class Bind(Protocol):
    """Resolve worker-local tensors against model resources before forward."""

    def __call__(
        self,
        module: Model,
        field: TensorFieldBase,
        observation: TensorDict | None,
        binding: object,
        *,
        address: Address,
        strata: Strata,
        resize: Resize,
    ) -> None: ...


class Loss(Protocol):
    """Return the scalar objective for selected targets of one field."""

    def __call__(
        self, module: Model, prediction: Prediction, batch: TensorFieldBase, strata: Strata
    ) -> torch.Tensor: ...


class OutputSchema(Protocol):
    """Declare the extension-owned Arrow fields written at one address."""

    def __call__(self, module: Model, address: Address) -> pa.StructType | None: ...


class Write(Protocol):
    """Materialize prediction coordinates with the declared Arrow schema."""

    def __call__(
        self, module: Model, prediction: Prediction, datatype: pa.StructType | None
    ) -> pa.StructArray | None: ...


MATCHERS: Mapping[type[Any], ArrowMatcher] = MappingProxyType(
    {
        object: lambda datatype: True,
        bool: pa.types.is_boolean,
        int: pa.types.is_integer,
        float: pa.types.is_floating,
        str: lambda datatype: pa.types.is_string(datatype) or pa.types.is_large_string(datatype),
        bytes: lambda datatype: (
            pa.types.is_binary(datatype)
            or pa.types.is_large_binary(datatype)
            or pa.types.is_fixed_size_binary(datatype)
        ),
        datetime: pa.types.is_timestamp,
        date: pa.types.is_date,
        time: pa.types.is_time,
        timedelta: pa.types.is_duration,
        dict: lambda datatype: pa.types.is_struct(datatype) or pa.types.is_map(datatype),
    }
)


def default_output(module: "Model", address: Address) -> None:
    return None


def default_write(
    module: "Model",
    prediction: Prediction,
    datatype: pa.StructType | None,
) -> None:
    return None


def default_observe(
    field: "RaggedField",
    *,
    address: Address,
    schema: "Schema",
    state: object | None,
    learn: bool,
) -> None:
    return None


def default_learn(
    module: "Model",
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    return None


@dataclass(frozen=True, slots=True)
class Context:
    """Per-field resources supplied uniformly during tensorization."""

    state: object | None = None
    salt: int = 0


class TensorInput(TensorClass):
    """Compact model input exposed to a tensorfield embedder."""

    state: torch.Tensor
    content: torch.Tensor | TensorDict


class EmbedderBase(torch.nn.Module):
    """Base class for tensorfield embedders."""

    def __init__(self, schema: Schema, address: Address) -> None:
        super().__init__()

        request = schema.requests[address]
        self.address = address
        self.destination = cast("Branch", request.parent).address
        self.d_model = schema.d_model
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

    @property
    def context(self) -> object | None:
        """Return the field-owned state shared with encoding workers."""

        return None

    def embed(self, field: "TensorFieldBase") -> Parcel:
        """Embed present coordinates and restore their fixed schema geometry."""

        shape = tuple(field.state.shape)
        present = field.present.reshape(-1)
        indices = present.nonzero(as_tuple=False).reshape(-1)
        count = indices.numel()

        if count:
            complete = count == present.numel()
            compact = self(field.take(None if complete else indices))
            if not isinstance(compact, Parcel):
                raise TypeError(f"embedder for '{self.address}' must return a Parcel, got {type(compact).__name__}")
            expected = (count, self.d_model)
            if tuple(compact.payload.shape) != expected:
                raise ValueError(
                    f"embedder for '{self.address}' must return payload shape {expected}, "
                    f"got {tuple(compact.payload.shape)}"
                )
            if compact.present.dtype != torch.bool or tuple(compact.present.shape) != (count,):
                raise ValueError(
                    f"embedder for '{self.address}' must return bool presence shape {(count,)}, "
                    f"got {tuple(compact.present.shape)} with dtype {compact.present.dtype}"
                )
            if compact.origin != self.address or compact.destination != self.destination:
                raise ValueError(
                    f"compact parcel from '{self.address}' must route from '{self.address}' "
                    f"to '{self.destination}', got '{compact.origin}' to '{compact.destination}'"
                )

            compact_payload = compact.payload.masked_fill(~compact.present.unsqueeze(-1), 0.0)
            if complete:
                payload = compact_payload
                restored = compact.present.clone()
            else:
                payload = compact.payload.new_zeros((math.prod(shape), self.d_model))
                payload = payload.index_copy(0, indices, compact_payload)
                restored = torch.zeros(math.prod(shape), dtype=torch.bool, device=compact.present.device)
                restored = restored.index_copy(0, indices, compact.present)
        else:
            payload = self.anchor.new_zeros((math.prod(shape), self.d_model))
            if torch.is_grad_enabled():
                for parameter in self.parameters():
                    payload = payload + parameter.sum() * 0.0
            restored = field.present.reshape(-1)

        return Parcel(
            payload=payload.reshape(*shape, self.d_model),
            present=restored.reshape(shape),
            origin=self.address,
            destination=self.destination,
            batch_size=shape[0],
        )


class DecoderBase(torch.nn.Module):
    """Decode field coordinates with extension-owned output heads.

    A concrete decoder may declare ``coordinatewise = True`` when ``decode``
    treats every target coordinate independently. Objective execution can then
    omit unselected queries while restoring the original tensor-tree geometry.
    Custom forward/context methods, hooks, and active dropout keep dense shapes
    and their original random-number schedule.
    """

    coordinatewise: ClassVar[bool] = False

    def __init__(self, schema: Schema, address: Address, *, conditioned: bool = True) -> None:
        super().__init__()

        self.address: Address = address
        self.register_buffer("anchor", torch.zeros(()), persistent=False)

        request = schema.requests[address]
        self.shape = schema.shapes[address]
        self.axes = tuple(node.address for node in request.path if getattr(node, "type", None) == "branch")
        n_context = math.prod(dimension if dimension is not None else 1 for dimension in self.shape)
        self.n_context = n_context
        self.d_model = schema.d_model
        self.memory_coordinates: dict[Address, tuple[tuple[Address, ...], tuple[int | None, ...], int | None]] = {}
        for origin in request.heritage:
            node = schema.requests[origin] if origin in schema.requests else schema.branches[origin]
            path = tuple(cast("Branch", node) for node in node.path if getattr(node, "type", None) == "branch")
            if isinstance(node, Leaf):
                outer = path[:-1]
                capacity = path[-1].length
            else:
                outer = path[:-2]
                capacity = schema.branch_outputs[origin]
                if len(path) > 1:
                    capacity = None if path[-2].length is None or capacity is None else path[-2].length * capacity
            self.memory_coordinates[origin] = (
                tuple(node.address for node in outer),
                tuple(node.length for node in outer),
                capacity,
            )

        def visible(node: Branch | Leaf) -> list[Address]:
            if isinstance(node, Leaf):
                return [node.address] if node.active and node.address != address else []

            if schema.branch_outputs[node.address] == 0:
                return []
            addresses = [node.address]
            if getattr(node, "type", None) == "branch" and node.reduction is None:
                for child in node.fields:
                    addresses.extend(visible(child))
            return addresses

        context_addresses: list[Address] = []
        if conditioned:
            for child in cast("Branch", request.parent).fields:
                if child.address == address:
                    continue
                context_addresses.extend(visible(child))
        self.context_addresses = tuple(dict.fromkeys(context_addresses))

        def aligns(context_address: Address) -> bool:
            if context_address in schema.requests:
                shape = schema.shapes[context_address]
                if any(dimension is None for dimension in (*self.shape, *shape)):
                    axes = tuple(
                        node.address
                        for node in schema.requests[context_address].path
                        if getattr(node, "type", None) == "branch"
                    )
                    return axes == self.axes
                return math.prod(cast(tuple[int, ...], shape)) == n_context

            branch = schema.branches[context_address]
            lengths = tuple(node.length for node in branch.path if getattr(node, "type", None) == "branch")
            output = schema.branch_outputs[context_address]
            if any(dimension is None for dimension in (*self.shape, *lengths, output)):
                axes = tuple(node.address for node in branch.path if getattr(node, "type", None) == "branch")
                return axes[:-1] == self.axes and output == 1
            return math.prod(cast(tuple[int, ...], (*lengths[1:-1], output))) == n_context

        self.aligned_addresses = frozenset(item for item in self.context_addresses if aligns(item))

        self.context_projection = (
            torch.nn.Sequential(
                torch.nn.RMSNorm(schema.d_model, eps=1e-5),
                torch.nn.Linear(schema.d_model, schema.d_model * request.ffnx),
                torch.nn.GELU(approximate="tanh"),
                torch.nn.Linear(schema.d_model * request.ffnx, schema.d_model),
            )
            if self.aligned_addresses and request.pooling == "query"
            else None
        )
        match request.pooling:
            case "query":
                self.pool = LearnedQueryCrossAttention(
                    ffnx=request.ffnx,
                    n_context=n_context,
                    d_model=schema.d_model,
                    nhead=request.n_heads,
                    dropout=float(request.dropout or 0.0),
                    n_linear=request.n_linear,
                    # By default scalar answers do not acquire an ordering over
                    # an otherwise unordered evidence set, while repeated
                    # targets retain position as one routing signal. Ordered
                    # scalar tasks can opt in explicitly.
                    position=(n_context > 1 or None in self.shape)
                    if request.decoder_position is None
                    else request.decoder_position,
                )
            case "mean":
                self.pool = MeanPool(n_context=n_context)
            case _:
                raise ValueError(f"unsupported decoder pooling: {request.pooling}")

    def attach(self, context: object | None) -> None:
        """Attach live resources owned by this field's embedder after construction."""
        return None

    def decode(self, pooled: torch.Tensor) -> TensorDict:
        raise NotImplementedError("decoder must implement decode(pooled)")

    def objective(
        self,
        parcels: list[Parcel],
        *,
        selection: torch.Tensor,
        contexts: list[Parcel] | None = None,
        batch_size: int,
        device: torch.device,
        embed: bool = False,
        target_shape: tuple[int, ...] | None = None,
        geometry: TensorDict | None = None,
    ) -> Prediction:
        """Decode objectives without imposing selection on custom implementations."""
        from relflow.architecture.compiler import canonical
        from relflow.architecture.packed import customized
        from relflow.architecture.pool import CrossAttentionBlock

        arguments: dict[str, Any] = dict(
            contexts=contexts,
            batch_size=batch_size,
            device=device,
            embed=embed,
            target_shape=target_shape,
            geometry=geometry,
        )
        count = math.prod(target_shape) if target_shape is not None else self.n_context
        # CUDA measurements found launch/synchronization overhead dominant below
        # this query volume. Scalar/all-selected targets retain the dense path.
        substantial = device.type != "cuda" or batch_size * count * self.d_model >= 1_048_576
        original = (
            type(self).__dict__.get("coordinatewise", False)
            and getattr(self.forward, "__func__", None) is DecoderBase.forward
            and getattr(self.context, "__func__", None) is DecoderBase.context
        )
        if not embed and count > 1 and substantial and original:
            pool = self.pool
            canonical_pool = (
                isinstance(pool, LearnedQueryCrossAttention)
                and type(pool).forward is LearnedQueryCrossAttention.forward
                and type(pool).compute is LearnedQueryCrossAttention.compute
                and canonical(pool, pool.compute)
                and pool.mass_projection is None
            ) or (isinstance(pool, MeanPool) and type(pool).forward is MeanPool.forward)
            layers = [pool, *([self.context_projection] if self.context_projection is not None else [])]
            hooks = ("_forward_pre_hooks", "_forward_hooks", "_backward_pre_hooks", "_backward_hooks")
            observed = any(
                "forward" in module.__dict__ or any(getattr(module, name) for name in hooks)
                for module in self.modules()
            )
            stochastic = any(
                isinstance(module, torch.nn.Dropout) and module.training and module.p > 0 for module in self.modules()
            ) or (
                isinstance(pool, LearnedQueryCrossAttention)
                and any(
                    block.attention.training and block.attention.dropout_p > 0
                    for block in cast(Iterator[CrossAttentionBlock], iter(pool.blocks))
                )
            )
            if (
                canonical_pool
                and not observed
                and not stochastic
                and not customized(
                    layers, additional=(LearnedQueryCrossAttention, CrossAttentionBlock, MeanPool, torch.nn.ModuleList)
                )
            ):
                arguments["selection"] = selection
        return self(parcels, **arguments)

    def forward(
        self,
        parcels: list[Parcel],
        *,
        contexts: list[Parcel] | None = None,
        batch_size: int,
        device: torch.device,
        embed: bool = False,
        target_shape: tuple[int, ...] | None = None,
        geometry: TensorDict | None = None,
        selection: torch.Tensor | None = None,
    ) -> Prediction:
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 0:
            raise ValueError(f"decoder batch_size must be a non-negative integer, got {batch_size!r}")

        shape = (
            target_shape
            if target_shape is not None
            else tuple(dimension if dimension is not None else 0 for dimension in self.shape)
        )
        count = math.prod(shape)
        lengths = tuple(cast(Geometry, geometry[axis]).lengths if geometry is not None else None for axis in self.axes)
        positions = coordinates((batch_size, *shape), self.shape, lengths, device=device).reshape(batch_size, count)
        queries = None
        if isinstance(self.pool, LearnedQueryCrossAttention) and (shape != self.shape or None in self.shape):
            indices = torch.zeros(shape, dtype=torch.int64, device=device)
            for axis, capacity in enumerate(self.shape):
                if capacity is not None:
                    axis_shape = [1] * len(shape)
                    axis_shape[axis] = shape[axis]
                    indices = indices * capacity + torch.arange(shape[axis], device=device).reshape(axis_shape)
            queries = self.pool.queries.index_select(0, indices.reshape(-1)).unsqueeze(0).expand(batch_size, -1, -1)
            if None in self.shape:
                queries = queries + signal(positions, self.d_model, dtype=queries.dtype)

        selected_indices = None
        selected_present = None
        original_count = count
        if embed:
            selection = None
        if selection is not None:
            if (
                selection.dtype != torch.bool
                or selection.numel() != batch_size * count
                or selection.device.type != device.type
                or (device.index is not None and selection.device.index != device.index)
            ):
                raise ValueError(f"decoder selection at '{self.address}' requires bool shape {(batch_size, *shape)}")
            selection = selection.reshape(batch_size, count)
            width = int(selection.sum(dim=1).max().item()) if batch_size else 0
            if width * 2 <= count:
                ranks = torch.arange(count, device=device).expand(batch_size, count)
                selected_indices = torch.where(selection, ranks, count).sort(dim=1).values[:, :width]
                selected_present = selected_indices < count
                selected_indices = selected_indices.clamp_max(max(count - 1, 0))
                positions = positions.gather(1, selected_indices)
                if isinstance(self.pool, LearnedQueryCrossAttention):
                    if queries is None:
                        queries = self.pool.queries.unsqueeze(0).expand(batch_size, -1, -1)
                    queries = queries.gather(1, selected_indices.unsqueeze(-1).expand(-1, -1, self.d_model))
                count = width

        if not parcels:
            pooled = torch.zeros(
                (batch_size, count, self.d_model),
                device=device,
                dtype=self.anchor.dtype,
            )
            if torch.is_grad_enabled():
                for parameter in self.pool.parameters():
                    pooled = pooled + parameter.sum() * 0.0
        else:
            for parcel in parcels:
                if parcel.payload.shape[0] != batch_size:
                    raise ValueError(
                        f"decoder parcel from '{parcel.origin}' has batch size {parcel.payload.shape[0]}, "
                        f"expected {batch_size}"
                    )
                if tuple(parcel.present.shape) != tuple(parcel.payload.shape[:-1]):
                    raise ValueError(
                        f"decoder parcel from '{parcel.origin}' presence must have shape "
                        f"{tuple(parcel.payload.shape[:-1])}, got {tuple(parcel.present.shape)}"
                    )

            stacked = torch.cat(
                [
                    parcel.payload.reshape(batch_size, math.prod(parcel.payload.shape[1:-1]), self.d_model)
                    for parcel in parcels
                ],
                dim=1,
            )
            present = torch.cat(
                [parcel.present.reshape(batch_size, math.prod(parcel.present.shape[1:])) for parcel in parcels], dim=1
            )
            memory_positions = []
            offset = torch.zeros(batch_size, dtype=torch.int64, device=device)
            for parcel in parcels:
                size = math.prod(parcel.present.shape[1:])
                if parcel.positions is None or parcel.spans is None:
                    ranks = torch.arange(size, device=device).expand(batch_size, size)
                    span = torch.full((batch_size,), size, device=device, dtype=torch.int64)
                else:
                    axes, capacities, capacity = self.memory_coordinates[parcel.origin]
                    lengths = tuple(
                        cast(Geometry, geometry[axis]).lengths if geometry is not None else None for axis in axes
                    )
                    ranks, span = flatten(
                        parcel.positions.reshape(parcel.present.shape)
                        if parcel.present.ndim > 1
                        else parcel.positions.reshape(batch_size, 1),
                        parcel.spans,
                        capacities=capacities,
                        lengths=lengths,
                        capacity=capacity,
                    )
                memory_positions.append(ranks + offset.unsqueeze(-1))
                offset = offset + span
            memory_positions = torch.cat(memory_positions, dim=1)
            context = (
                self.context(contexts or (), batch_size=batch_size, n_context=original_count)
                if selected_indices is None
                else self.context(
                    contexts or (), batch_size=batch_size, n_context=original_count, selection=selected_indices
                )
            )
            if isinstance(self.pool, LearnedQueryCrossAttention):
                pooled = self.pool(
                    stacked,
                    present=present,
                    context=context,
                    queries=queries,
                    positions=positions,
                    memory_positions=memory_positions,
                )
            else:
                pooled = self.pool(stacked, present=present)
                if count != self.n_context:
                    pooled = pooled[:, :1].expand(batch_size, count, self.d_model)
            if context is not None:
                pooled = pooled + context * 0.0

        payload = self.decode(pooled)
        if embed:
            payload[TensorKey.embedding] = pooled
        if selected_indices is not None:
            rows = torch.arange(batch_size, device=device).unsqueeze(-1)
            destination = (rows * original_count + selected_indices).reshape(-1)
            selected_present = cast(torch.Tensor, selected_present)

            def restore(source: TensorDictBase) -> TensorDict:
                # Extensions may include the target axis in nested batch metadata.
                batch = list(source.batch_size)
                if len(batch) > 1:
                    batch[1] = original_count
                result = TensorDict({}, batch_size=batch, device=source.device, names=source.names)
                for key, value in source.items():
                    if isinstance(value, TensorDictBase):
                        result[key] = restore(value)
                        continue
                    if not isinstance(value, torch.Tensor):
                        raise TypeError(
                            f"decoder payload at '{self.address}' requires tensor leaves, got {type(value).__name__}"
                        )
                    trailing = value.shape[2:]
                    dense = value.new_zeros(batch_size * original_count, *trailing)
                    active = selected_present.to(value.device).reshape(*selected_present.shape, *([1] * len(trailing)))
                    values = value.masked_fill(~active, 0.0).reshape(batch_size * count, *trailing)
                    dense.index_add_(0, destination.to(value.device), values)
                    result[key] = dense.reshape(batch_size, original_count, *trailing)
                return result

            payload = restore(payload)

        return Prediction(
            payload=payload,
            address=self.address,
            batch_size=pooled.shape[0],
        )

    def context(
        self,
        parcels: tuple[Parcel, ...] | list[Parcel],
        *,
        batch_size: int,
        n_context: int | None = None,
        selection: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Build one data-conditioned decoder query per target coordinate."""

        if self.context_projection is None:
            return None

        aligned: list[torch.Tensor] = []
        presence: list[torch.Tensor] = []
        n_context = self.n_context if n_context is None else n_context
        expected = (batch_size, n_context, self.d_model)
        for parcel in parcels:
            if parcel.origin not in self.aligned_addresses:
                continue
            payload = parcel.payload.reshape(batch_size, math.prod(parcel.payload.shape[1:-1]), self.d_model)
            present = parcel.present.reshape(batch_size, math.prod(parcel.present.shape[1:]))
            if tuple(payload.shape) != expected:
                continue
            if selection is not None:
                payload = payload.gather(1, selection.unsqueeze(-1).expand(-1, -1, self.d_model))
                present = present.gather(1, selection)
            aligned.append(payload)
            presence.append(present)

        if not aligned:
            return None

        payload = torch.stack(aligned, dim=2)
        present = torch.stack(presence, dim=2)
        count = present.sum(dim=2, keepdim=True, dtype=payload.dtype)
        combined = payload.masked_fill(~present.unsqueeze(-1), 0.0).sum(dim=2)
        combined = combined / count.clamp_min(1.0)
        combined = combined + self.context_projection(combined)
        return combined.masked_fill(~present.any(dim=2).unsqueeze(-1), 0.0)


class TensorFieldBase(ABC, Generic[ContentT]):
    """Tensorized values and targets with an extension-owned content type.

    Consumers read content through this base interface. Concrete tensorclasses
    declare its storage as either a Tensor or a TensorDict and own replacement.
    """

    if TYPE_CHECKING:

        @property
        def content(self) -> ContentT: ...
    else:
        content: torch.Tensor | TensorDict
    state: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    @classmethod
    @abstractmethod
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
    ) -> Self:
        raise NotImplementedError

    def take(self, indices: torch.Tensor | None) -> TensorInput:
        """Copy selected row-major coordinates, or every coordinate when indices is None."""

        if indices is not None:
            if indices.ndim != 1 or indices.dtype != torch.int64:
                raise TypeError("tensorfield take indices must be a one-dimensional int64 tensor")
            if indices.device != self.state.device:
                raise ValueError(
                    f"tensorfield take indices must use state device {self.state.device}, got {indices.device}"
                )

        size = math.prod(self.state.shape)
        if indices is not None and indices.numel() and ((indices < 0) | (indices >= size)).any():
            raise IndexError(f"tensorfield take indices must be between 0 and {size - 1}")
        count = size if indices is None else indices.numel()

        def gather(value: Any) -> Any:
            if torch.is_tensor(value):
                if value.ndim < self.state.ndim or tuple(value.shape[: self.state.ndim]) != tuple(self.state.shape):
                    raise ValueError(
                        f"tensorfield content must start with state shape {tuple(self.state.shape)}, "
                        f"got {tuple(value.shape)}"
                    )
                trailing = tuple(value.shape[self.state.ndim :])
                flat = value.reshape(size, *trailing)
                if indices is None:
                    return flat.clone(memory_format=torch.contiguous_format)
                return flat.index_select(0, indices)
            if isinstance(value, TensorDict):
                return TensorDict(
                    {key: gather(value[key]) for key in value.keys()},
                    batch_size=[count],
                    device=value.device,
                )
            raise TypeError(f"tensorfield content must contain only tensors or TensorDicts, got {type(value).__name__}")

        state = self.state.reshape(size)
        return TensorInput(
            state=state.clone() if indices is None else state.index_select(0, indices),
            content=gather(self.content),
            batch_size=[count],
        )


TENSORFIELDS: dict[str, "Extension"] = {}


class Extension:
    """Registry object for a tensorfield implementation.

    Register request, tensorfield, embedder, decoder, loss, output, and write
    components with ``@extension.register(Component.Request)`` or another explicit
    role. ``types`` names the Python equivalents
    of accepted canonical Arrow terminal families. Separate tuple entries are
    incompatible; types joined in one PEP 604 union may share one Arrow column.
    ``arrow`` supplies a physical-type matcher for custom Python atoms or
    overrides a standard matcher. ``requires`` maps importable module names to
    their package requirements. ``callbacks`` lists the Lightning callback
    classes or factories every model using the extension attaches; every rank
    must declare the same. Creating an extension with an existing name replaces
    the registry entry and emits a warning.
    """

    def __init__(
        self,
        name: str,
        *,
        types: tuple[ValueTypeFamily, ...],
        arrow: Mapping[type[Any], ArrowMatcher] | None = None,
        requires: Mapping[str, str] | None = None,
        callbacks: Sequence[CallbackFactory] = (),
    ):
        if not isinstance(name, str):
            raise TypeError("Extension name must be a string")

        if not re.match(r"^[a-z0-9_]+$", name):
            raise ValueError("Extension name must consist of lowercase letters, numbers, and underscores only")

        if not isinstance(types, tuple):
            raise TypeError("Extension types must be a tuple of types or PEP 604 unions")
        if not types:
            raise ValueError("Extension types must contain at least one value type family")

        families: list[tuple[type[Any], ...]] = []
        type_to_family: dict[type[Any], int] = {}
        for family_index, family in enumerate(types):
            members = get_args(family) if isinstance(family, UnionType) else (family,)
            if not members or any(member is Any or not isinstance(member, type) for member in members):
                raise TypeError("Extension types must contain only concrete types or PEP 604 unions of concrete types")
            if type(None) in members:
                raise TypeError("Extension types must not include NoneType; null is represented by field state")
            if any(issubclass(member, (list, tuple, np.ndarray, np.generic, Iterator)) for member in members):
                raise TypeError(
                    "Extension types declare canonical terminal Python atoms; "
                    "sequences, iterators, ndarrays, and NumPy scalar classes are prepared structurally"
                )

            for member in members:
                if member in type_to_family:
                    raise ValueError(
                        f"Extension value type {member.__name__} appears in more than one compatibility family"
                    )
                type_to_family[member] = family_index
            families.append(cast(tuple[type[Any], ...], tuple(members)))

        if arrow is not None and not isinstance(arrow, Mapping):
            raise TypeError("Extension arrow must be a mapping from declared value types to Arrow matchers")
        declared = {} if arrow is None else dict(arrow)
        unknown = [member for member in declared if member not in type_to_family]
        if unknown:
            names = ", ".join(getattr(member, "__name__", repr(member)) for member in unknown)
            raise ValueError(f"Extension arrow matcher type(s) are absent from types: {names}")
        invalid = [member for member, matcher in declared.items() if not callable(matcher)]
        if invalid:
            names = ", ".join(member.__name__ for member in invalid)
            raise TypeError(f"Extension Arrow matcher(s) must be callable: {names}")
        missing = [member for member in type_to_family if member not in MATCHERS and member not in declared]
        if missing:
            names = ", ".join(member.__name__ for member in missing)
            raise TypeError(f"Extension custom value type(s) require arrow matchers: {names}")

        if requires is not None and not isinstance(requires, Mapping):
            raise TypeError("Extension requires must be a mapping from import names to pip requirements")
        dependencies = {} if requires is None else dict(requires)
        invalid_imports = [
            module
            for module in dependencies
            if not isinstance(module, str) or not module or any(not part.isidentifier() for part in module.split("."))
        ]
        if invalid_imports:
            names = ", ".join(repr(module) for module in invalid_imports)
            raise ValueError(f"Extension requires contains invalid Python import name(s): {names}")
        invalid_requirements = [
            module
            for module, requirement in dependencies.items()
            if not isinstance(requirement, str) or not requirement.strip()
        ]
        if invalid_requirements:
            names = ", ".join(repr(module) for module in invalid_requirements)
            raise ValueError(f"Extension requires contains empty or non-string pip requirement(s) for: {names}")

        if isinstance(callbacks, (str, bytes)) or not isinstance(callbacks, Sequence):
            raise TypeError("Extension callbacks must be a sequence of Lightning Callback classes or factories")
        for factory in callbacks:
            if not callable(factory) or not isinstance(factory(), Callback):
                raise TypeError(f"Extension callback factory for '{name}' must produce a Lightning Callback")

        self.name: str = name
        self.value_types: tuple[ValueTypeFamily, ...] = types
        self.families: tuple[tuple[type[Any], ...], ...] = tuple(families)
        self.family_by_type: Mapping[type[Any], int] = MappingProxyType(type_to_family)
        self.matchers: Mapping[type[Any], ArrowMatcher] = MappingProxyType(
            {member: declared[member] if member in declared else MATCHERS[member] for member in type_to_family}
        )
        self.requires: Mapping[str, str] = MappingProxyType(
            {module: requirement.strip() for module, requirement in dependencies.items()}
        )
        self.components: dict[Component, ComponentValue | None] = {}
        self.callback_factories: tuple[CallbackFactory, ...] = tuple(callbacks)

        if name in TENSORFIELDS:
            warnings.warn(
                f"Extension '{name}' already registered; overriding existing tensorfield extension",
                UserWarning,
                stacklevel=2,
            )

        TENSORFIELDS[name] = self

    @property
    def types(self) -> tuple[ValueTypeFamily, ...]:
        """Registered raw atom compatibility families."""
        return self.value_types

    def require(self, *, address: Address) -> None:
        """Fail before graph construction when optional runtime packages are absent."""

        missing: list[str] = []
        for module in self.requires:
            if module in sys.modules:
                available = sys.modules[module] is not None
            else:
                try:
                    available = importlib.util.find_spec(module) is not None
                except (AttributeError, ImportError, ValueError):
                    available = False
            if not available:
                missing.append(module)

        if not missing:
            return

        requirements = dict.fromkeys(self.requires[module] for module in missing)
        command = " ".join(shlex.quote(requirement) for requirement in requirements)
        modules = ", ".join(repr(module) for module in missing)
        noun = "package" if len(missing) == 1 else "packages"
        raise ModuleNotFoundError(
            f"model field '{address}' uses extension '{self.name}', which requires uninstalled {noun}: "
            f"{modules}; install with `uv add {command}`",
            name=missing[0],
        )

    def terminals(self, datatype: pa.DataType) -> tuple[pa.DataType, ...]:
        """Return terminal Arrow atom types beneath structural containers."""

        if pa.types.is_dictionary(datatype):
            return self.terminals(datatype.value_type)
        if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
            return self.terminals(datatype.value_type)
        if pa.types.is_union(datatype):
            return tuple(child for field in datatype for child in self.terminals(field.type))
        return (datatype,)

    def matches(self, member: type[Any], datatype: pa.DataType) -> bool:
        """Match one registered Python atom to one Arrow terminal type."""

        if pa.types.is_null(datatype):
            return True
        matcher = self.matchers[member]
        if matcher(datatype):
            return True
        if isinstance(datatype, pa.ExtensionType):
            return any(self.matches(member, terminal) for terminal in self.terminals(datatype.storage_type))
        return False

    def accepts(self, datatype: pa.DataType) -> bool:
        """Return whether one Arrow value type belongs to an allowed family."""

        families: set[int] = set()
        for terminal in self.terminals(datatype):
            if pa.types.is_null(terminal):
                continue
            matches = {
                family
                for family, members in enumerate(self.families)
                if any(self.matches(member, terminal) for member in members)
            }
            if not matches:
                return False
            families.update(matches)
        return len(families) <= 1

    def canonical(self, datatype: pa.DataType) -> pa.DataType:
        """Resolve Arrow encodings that wrap an extension's declared atom types."""

        if pa.types.is_dictionary(datatype):
            return self.canonical(datatype.value_type)
        if pa.types.is_union(datatype):
            children = [self.canonical(field.type) for field in datatype]
            try:
                return (
                    pa.unify_schemas(
                        [pa.schema([pa.field("value", child)]) for child in children],
                        promote_options="permissive",
                    )
                    .field("value")
                    .type
                )
            except pa.ArrowException:
                return datatype
        if pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype):
            child = self.canonical(datatype.value_type)
            if child == datatype.value_type:
                return datatype
            field = pa.field(
                datatype.value_field.name,
                child,
                nullable=datatype.value_field.nullable,
                metadata=datatype.value_field.metadata,
            )
            if pa.types.is_large_list(datatype):
                return pa.large_list(field)
            if pa.types.is_fixed_size_list(datatype):
                return pa.list_(field, datatype.list_size)
            return pa.list_(field)
        if isinstance(datatype, pa.ExtensionType):
            direct = any(matcher(datatype) for matcher in self.matchers.values())
            return datatype if direct else self.canonical(datatype.storage_type)
        return datatype

    @overload
    def normalize(self, values: pa.Array, datatype: pa.DataType) -> pa.Array: ...

    @overload
    def normalize(self, values: pa.ChunkedArray, datatype: pa.DataType) -> pa.ChunkedArray: ...

    def normalize(
        self,
        values: pa.Array | pa.ChunkedArray,
        datatype: pa.DataType,
    ) -> pa.Array | pa.ChunkedArray:
        """Normalize Arrow wrappers to one codec-ready physical representation."""

        if values.type == datatype:
            return values
        if isinstance(values, pa.ChunkedArray):
            # Registered Arrow datatypes exceed the stubs' enumerated overloads.
            return pa.chunked_array(  # pyrefly: ignore [no-matching-overload]
                [self.normalize(chunk, datatype) for chunk in values.chunks],
                type=datatype,
            )
        if pa.types.is_dictionary(values.type):
            values = cast(pa.DictionaryArray, values)
            decoded = pc.take(values.dictionary, values.indices)
            return self.normalize(decoded, datatype)
        if isinstance(values, pa.ExtensionArray) and not isinstance(datatype, pa.ExtensionType):
            return self.normalize(values.storage, datatype)
        if pa.types.is_union(values.type) and not pa.types.is_union(datatype):
            values = cast(pa.UnionArray, values)
            codes, offsets = variants(values)
            # nulls accepts any Arrow datatype, including registered extensions.
            result = pa.nulls(len(values), type=cast(pa.DataType, datatype))  # pyrefly: ignore [no-matching-overload]
            for index, code in enumerate(values.type.type_codes):
                selected = codes == code
                positions = np.flatnonzero(selected).astype(np.int64, copy=False)
                if not len(positions):
                    continue
                indices = offsets[selected] if offsets is not None else positions
                child = pc.take(values.field(index), pa.array(indices, type=pa.int64()))
                normalized = self.normalize(child, datatype)
                gather = np.zeros(len(values), dtype=np.int64)
                gather[positions] = np.arange(len(positions))
                placed = pc.take(normalized, pa.array(gather, type=pa.int64(), mask=~selected))
                result = pc.coalesce(result, placed)
            return result
        if (
            pa.types.is_list(values.type)
            or pa.types.is_large_list(values.type)
            or pa.types.is_fixed_size_list(values.type)
        ) and (pa.types.is_list(datatype) or pa.types.is_large_list(datatype) or pa.types.is_fixed_size_list(datatype)):
            mask = pc.is_null(values)
            if pa.types.is_fixed_size_list(values.type):
                values = cast(pa.FixedSizeListArray, values)
                start = values.offset * values.type.list_size
                length = len(values) * values.type.list_size
                child = values.values.slice(start, length)
            else:
                values = cast(pa.ListArray | pa.LargeListArray, values)
                start = values.offsets[0].as_py()
                stop = values.offsets[-1].as_py()
                child = values.values.slice(start, stop - start)
            child = self.normalize(child, datatype.value_type)
            if pa.types.is_fixed_size_list(datatype):
                # The stubs omit Arrow's explicit-type constructor overload.
                return pa.FixedSizeListArray.from_arrays(child, type=datatype, mask=mask)  # pyrefly: ignore [no-matching-overload]
            values = cast(pa.ListArray | pa.LargeListArray, values)
            offsets = pc.subtract(values.offsets, values.offsets[0])
            if pa.types.is_large_list(datatype):
                return pa.LargeListArray.from_arrays(offsets, child, type=datatype, mask=mask)
            return pa.ListArray.from_arrays(pc.cast(offsets, pa.int32()), child, type=datatype, mask=mask)
        return pc.cast(values, datatype, safe=True)

    def prepare(
        self,
        values: pa.Array | pa.ChunkedArray,
        *,
        address: Address,
    ) -> pa.Array | pa.ChunkedArray:
        """Validate one whole Arrow leaf column at the extension boundary."""

        if not isinstance(values, (pa.Array, pa.ChunkedArray)):
            raise TypeError(
                f"extension '{self.name}' at address '{address}' requires an Arrow array, got {type(values).__name__}"
            )
        if not self.accepts(values.type):
            expected = ", ".join(" | ".join(member.__name__ for member in members) for members in self.families)
            raise TypeError(
                f"extension '{self.name}' at address '{address}' does not accept Arrow type {values.type}; "
                f"expected terminal atoms compatible with {expected}; normalize it in a preprocessor"
            )
        datatype = self.canonical(values.type)
        try:
            return self.normalize(values, datatype)
        except pa.ArrowException as error:
            raise TypeError(
                f"extension '{self.name}' at address '{address}' cannot safely normalize Arrow type "
                f"{values.type} to {datatype}; normalize it in a preprocessor"
            ) from error

    @overload
    def register(self, component: Component | str) -> Callable[[RegisterT], RegisterT]: ...

    @overload
    def register(self, component: Component | str, obj: RegisterT) -> RegisterT: ...

    @overload
    def register(self, component: Component | str, obj: None) -> None: ...

    def register(
        self,
        component: Component | str,
        obj: RegisterT | None | object = OMITTED,
    ) -> Callable[[RegisterT], RegisterT] | RegisterT | None:
        """Register an explicit role directly or return its identity decorator.

        Class roles require their corresponding extension base class. Hooks
        follow the declared protocols without constraining callable names,
        annotations, or introspectable signatures. Only output and write may
        explicitly register None, retaining their optional no-op behavior.
        """
        key = Component(component)
        if obj is OMITTED:
            return partial(self.register, key)
        if key in self.components:
            raise ValueError(f"Component '{key}' already registered in extension '{self.name}'")
        if obj is None:
            if key not in {Component.output, Component.write}:
                raise TypeError(
                    f"extension '{self.name}' component '{key}' cannot be None; "
                    "only output and write may be registered as None"
                )
        else:
            base = {
                Component.Request: RequestBase,
                Component.TensorField: TensorFieldBase,
                Component.Embedder: EmbedderBase,
                Component.Decoder: DecoderBase,
            }.get(key)
            if base is not None:
                if not isinstance(obj, type) or not issubclass(obj, base):
                    actual = obj.__name__ if isinstance(obj, type) else type(obj).__name__
                    raise TypeError(
                        f"extension '{self.name}' component '{key}' must be a subclass of {key}Base, got {actual}"
                    )
            elif not callable(obj):
                raise TypeError(f"extension '{self.name}' component '{key}' must be callable, got {type(obj).__name__}")

        value = cast(ComponentValue | None, obj)
        self.components[key] = value
        return cast(RegisterT | None, value)

    @property
    def callbacks(self) -> list[Callback]:
        """Instantiate all registered callback factories."""
        return [factory() for factory in self.callback_factories]

    def component(self, component: Component) -> ComponentValue:
        """Return a registered component, falling back to optional defaults."""
        if component in self.components:
            value = self.components[component]
            if value is not None:
                return value

        if component == Component.output:
            return default_output

        if component == Component.write:
            return default_write

        if component == Component.observe:
            return default_observe

        if component == Component.learn:
            return default_learn

        raise AttributeError(f"Extension '{self.name}' has no component '{component}'")

    @property
    def Request(self) -> type[RequestBase]:
        return cast(type[RequestBase], self.component(Component.Request))

    @property
    def TensorField(self) -> type[TensorFieldBase]:
        return cast(type[TensorFieldBase], self.component(Component.TensorField))

    @property
    def Embedder(self) -> type[EmbedderBase]:
        return cast(type[EmbedderBase], self.component(Component.Embedder))

    @property
    def Decoder(self) -> type[DecoderBase]:
        return cast(type[DecoderBase], self.component(Component.Decoder))

    @property
    def observe(self) -> Observe:
        return cast(Observe, self.component(Component.observe))

    @property
    def learn(self) -> Learn:
        return cast(Learn, self.component(Component.learn))

    @property
    def loss(self) -> Loss:
        return cast(Loss, self.component(Component.loss))

    @property
    def output(self) -> OutputSchema:
        return cast(OutputSchema, self.component(Component.output))

    @property
    def write(self) -> Write:
        return cast(Write, self.component(Component.write))
