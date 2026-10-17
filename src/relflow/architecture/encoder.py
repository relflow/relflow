from __future__ import annotations

import math
from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch
from tensordict import TensorDict

from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.blocks import feed_forward, norm
from relflow.architecture.packed import SDPA_BATCH_MAX, Packed, block, customized, pack, stochastic, unpack
from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.structs.packages import Parcel
from relflow.structs.reduction import Attention, Mean
from relflow.structs.tree import Address, Leaf, Node

if TYPE_CHECKING:
    from relflow.data.arrow import Geometry
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


def locate(parcel: Parcel, *, schema: Schema, geometry: TensorDict | None) -> Parcel:
    """Attach structural leaf coordinates independently of values and masking."""
    branch = cast("Branch", schema.requests[parcel.origin].parent)
    capacity = branch.length
    width = parcel.payload.shape[-2]
    shape = tuple(parcel.payload.shape[:-2])
    layout = cast("Geometry", geometry[branch.address]) if geometry is not None else None
    lengths = layout.lengths if layout is not None else None
    parcel.positions = torch.arange(width, device=parcel.payload.device).expand(parcel.payload.shape[:-1])
    parcel.spans = (
        torch.full(shape, capacity, device=parcel.payload.device, dtype=torch.int64)
        if capacity is not None
        else lengths
        if lengths is not None
        else torch.full(shape, width, device=parcel.payload.device, dtype=torch.int64)
    )
    parcel.slots = (
        parcel.positions < lengths.unsqueeze(-1)
        if lengths is not None
        else torch.ones_like(parcel.present, dtype=torch.bool)
    )
    return parcel


def sequence(
    parcels: list[Parcel], *, order: dict[Address, int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate fields using structural spans rather than padded storage widths."""
    if order is not None:
        parcels = sorted(parcels, key=lambda parcel: order[parcel.origin])
    first = parcels[0]
    spans = torch.zeros(first.payload.shape[:-2], device=first.payload.device, dtype=torch.int64)
    positions, slots = [], []
    for parcel in parcels:
        local = parcel.positions
        if local is None:
            local = torch.arange(parcel.payload.shape[-2], device=parcel.payload.device).expand(
                parcel.payload.shape[:-1]
            )
        positions.append(local + spans.unsqueeze(-1))
        slots.append(torch.ones_like(parcel.present) if parcel.slots is None else parcel.slots)
        spans = spans + (parcel.payload.shape[-2] if parcel.spans is None else parcel.spans)
    return torch.cat(positions, dim=-1), spans, torch.cat(slots, dim=-1)


class RotaryTransformerEncoderLayer(torch.nn.Module):
    """Preserve absent coordinates through attention and feed-forward residuals."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_kv_heads: int,
        dropout: float,
        *,
        ffnx: int = 8,
    ) -> None:
        super().__init__()

        self.attention_norm = norm(d_model)
        self.ffn_norm = norm(d_model)

        self.attention = RotaryMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )

        self.ffn = feed_forward(d_model, dropout, ffnx)

    def forward(
        self,
        inputs: torch.Tensor,
        present: torch.Tensor | None,
        *,
        packing: Packed | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if packing is not None:
            return block(self, inputs, packing)
        if present is not None and (present.dtype != torch.bool or tuple(present.shape) != tuple(inputs.shape[:2])):
            raise ValueError(
                f"encoder presence must have bool shape {tuple(inputs.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )
        if present is not None:
            inputs = inputs.masked_fill(~present.unsqueeze(-1), 0.0)
        normed = self.attention_norm(inputs)
        inputs = inputs + self.attention(
            normed,
            normed,
            normed,
            key_padding_mask=None if present is None else ~present,
            query_positions=positions,
            key_positions=positions,
        )
        inputs = inputs + self.ffn(self.ffn_norm(inputs))
        return inputs if present is None else inputs.masked_fill(~present.unsqueeze(-1), 0.0)

    if TYPE_CHECKING:
        __call__ = forward


class BranchEncoder(torch.nn.Module):
    """Contextualize child parcels and route their reduced branch representation."""

    def __init__(self, schema: Schema, address: Address) -> None:
        super().__init__()

        branch = schema.branches[address]
        dropout = float(branch.dropout or 0.0)

        self.origin: Address = address
        # Schema binding attaches every branch to a node, including the schema root.
        self.destination: Address = cast(Node, branch.parent).address

        layers: list[RotaryTransformerEncoderLayer] = []
        attention = branch.attention
        if attention is not None:
            for _ in range(branch.n_layers):
                layers.append(
                    RotaryTransformerEncoderLayer(
                        d_model=schema.d_model,
                        nhead=branch.n_heads,
                        n_kv_heads=attention.kv_heads(branch.n_heads),
                        dropout=dropout,
                        ffnx=branch.ffnx,
                    )
                )

        self.encoder = torch.nn.ModuleList(layers)

        widths = [
            int(child.active) if isinstance(child, Leaf) else schema.branch_outputs[child.address]
            for child in branch.fields
        ]
        child_width = None if any(width is None for width in widths) else sum(cast(list[int], widths))
        self.input_width = None if branch.length is None or child_width is None else branch.length * child_width
        self.parent_capacity = getattr(branch.parent, "length", 1)
        self.child_order = {child.address: index for index, child in enumerate(branch.fields)}

        self.coordinate_origins = frozenset(
            child.address for child in branch.fields if isinstance(child, Leaf) and child.active
        )
        self.coordinate_encoder: RotaryTransformerEncoderLayer | None = None
        if attention is not None and len(self.coordinate_origins) > 1:
            coordinate_heads = branch.n_heads
            while coordinate_heads > 0 and (
                schema.d_model % coordinate_heads != 0 or schema.d_model // coordinate_heads < 2
            ):
                coordinate_heads -= 1
            if coordinate_heads < 1:
                raise ValueError(
                    f"branch '{self.origin}' cannot allocate coordinate attention heads for d_model={schema.d_model}"
                )
            self.coordinate_encoder = RotaryTransformerEncoderLayer(
                d_model=schema.d_model,
                nhead=coordinate_heads,
                n_kv_heads=attention.kv_heads(coordinate_heads),
                dropout=dropout,
                ffnx=branch.ffnx,
            )

        self.reduction = branch.reduction
        match branch.reduction:
            case None:
                self.n_outputs: int | None = None
                self.pool: torch.nn.Module | None = None
            case Mean():
                self.n_outputs = 1
                self.pool = MeanPool(n_context=1)
            case Attention():
                self.n_outputs = branch.reduction.n_outputs
                self.pool = LearnedQueryCrossAttention(
                    n_context=branch.reduction.n_outputs,
                    d_model=schema.d_model,
                    nhead=branch.reduction.n_heads or branch.n_heads,
                    dropout=float(branch.reduction.dropout if branch.reduction.dropout is not None else dropout),
                    n_linear=branch.reduction.n_layers,
                    position=branch.reduction.position,
                    mass_capacity=max(1, self.input_width or 1),
                    ffnx=branch.ffnx,
                )
            case _:
                raise ValueError(f"unsupported branch reduction: {branch.reduction}")

    def forward(self, parcels: list[Parcel], *, geometry: TensorDict | None = None) -> Parcel:
        from relflow.architecture.compiler import canonical

        if not parcels:
            raise ValueError(f"branch encoder '{self.origin}' requires at least one child parcel")

        unknown = [parcel.origin for parcel in parcels if parcel.origin not in self.child_order]
        if unknown:
            origins = ", ".join(repr(str(origin)) for origin in unknown)
            raise ValueError(f"branch encoder '{self.origin}' received parcel(s) from non-child address(es): {origins}")
        parcels = sorted(parcels, key=lambda parcel: self.child_order[parcel.origin])
        parcels = self.contextualize(parcels)
        positions, spans, slots = sequence(parcels)
        positioned = any(parcel.positions is not None for parcel in parcels)
        payloads = [parcel.payload for parcel in parcels]
        presence = [parcel.present for parcel in parcels]
        for parcel in parcels:
            if parcel.present.dtype != torch.bool or tuple(parcel.present.shape) != tuple(parcel.payload.shape[:-1]):
                raise ValueError(
                    f"parcel from '{parcel.origin}' presence must have bool shape "
                    f"{tuple(parcel.payload.shape[:-1])}, got {tuple(parcel.present.shape)} "
                    f"with dtype {parcel.present.dtype}"
                )

        concatenated: torch.Tensor = torch.cat(payloads, dim=-2)
        present = torch.cat(presence, dim=-1)
        N, *dims, L, C = concatenated.shape
        rows = N * math.prod(dims)
        encoded: torch.Tensor = concatenated.reshape(rows, L, C)
        present = present.reshape(rows, L)
        flat_positions = positions.reshape(rows, L)
        active = present.any(dim=1)
        indices = active.nonzero(as_tuple=False).reshape(-1)

        output_width = L if self.n_outputs is None else self.n_outputs
        count = indices.numel()
        complete = count > 0 and count == encoded.shape[0]
        shape = (encoded.shape[0], output_width, C)
        selected_output = encoded
        selected_output_present = present
        if count:
            selected = encoded if complete else encoded.index_select(0, indices)
            selected_present = present.index_select(0, indices)
            selected_positions = flat_positions if complete else flat_positions.index_select(0, indices)
            selected = selected.masked_fill(~selected_present.unsqueeze(-1), 0.0)
            evidence = selected

            prepared = (
                bool(self.encoder)
                and not customized(self.encoder)
                and not stochastic(cast(Iterable[RotaryTransformerEncoderLayer], self.encoder))
            )
            full = prepared and bool(selected_present.all())
            if prepared and not full:
                values, packing = pack(selected, selected_present, selected_positions if positioned else None)
                values = self.compute(values, selected_present, packing=packing)
                selected = unpack(
                    values,
                    packing,
                    selected,
                    selected_present,
                    cast(Iterable[RotaryTransformerEncoderLayer], self.encoder),
                )
            else:
                # FlashAttention uses a CUDA grid dimension for sequence batches.
                unmasked = (
                    full
                    and selected.is_cuda
                    and selected.shape[0] <= SDPA_BATCH_MAX
                    and type(self) is BranchEncoder
                    and canonical(self, self.compute)
                )
                if positioned and type(self).compute is BranchEncoder.compute and canonical(self, self.compute):
                    selected = self.compute(
                        selected, None if unmasked else selected_present, positions=selected_positions
                    )
                else:
                    selected = self.compute(selected, None if unmasked else selected_present)

            match self.reduction:
                case None:
                    selected_output = selected
                    selected_output_present = selected_present
                case _:
                    assert self.pool is not None
                    if isinstance(self.pool, LearnedQueryCrossAttention):
                        selected_output = self.pool(
                            selected,
                            present=selected_present,
                            evidence=evidence,
                            memory_positions=selected_positions if positioned else None,
                        )
                    else:
                        selected_output = self.pool(selected, present=selected_present)
                    selected_output_present = torch.ones(
                        (indices.numel(), output_width),
                        dtype=torch.bool,
                        device=present.device,
                    )

            complete = complete and (
                selected_output.shape == shape
                and selected_output.dtype == encoded.dtype
                and selected_output.device == encoded.device
                and selected_output_present.shape == shape[:-1]
                and selected_output_present.dtype == present.dtype
                and selected_output_present.device == present.device
            )

        if complete:
            reduced = selected_output.clone(memory_format=torch.contiguous_format)
            reduced_present = selected_output_present.clone(memory_format=torch.contiguous_format)
        else:
            reduced = encoded.new_zeros(shape)
            reduced_present = present.new_zeros(shape[:-1])
            if count:
                reduced = reduced.index_copy(0, indices, selected_output)
                reduced_present = reduced_present.index_copy(0, indices, selected_output_present)
            elif torch.is_grad_enabled():
                for parameter in self.parameters():
                    reduced = reduced + parameter.sum() * 0.0

        reduced = reduced.reshape(N, *dims, output_width, C)
        reduced_present = reduced_present.reshape(N, *dims, output_width)
        if self.n_outputs is not None:
            positions = torch.arange(output_width, device=encoded.device).expand(N, *dims, output_width)
            spans = torch.full((N, *dims), output_width, device=encoded.device, dtype=torch.int64)
            slots = torch.ones((N, *dims, output_width), device=encoded.device, dtype=torch.bool)
        if dims:
            offsets = spans.cumsum(dim=-1) - spans
            positions = (positions + offsets.unsqueeze(-1)).reshape(N, *dims[:-1], dims[-1] * output_width)
            slots = slots.reshape(N, *dims[:-1], dims[-1] * output_width)
            parent = cast("Geometry", geometry[self.destination]) if geometry is not None else None
            if parent is not None:
                retained = torch.arange(dims[-1], device=encoded.device) < parent.lengths.unsqueeze(-1)
                slots = slots.reshape(N, *dims, output_width).masked_fill(~retained.unsqueeze(-1), False)
                slots = slots.reshape(N, *dims[:-1], dims[-1] * output_width)
                spans = spans.masked_fill(~retained, 0)
            capacity = self.input_width if self.n_outputs is None else self.n_outputs
            spans = (
                torch.full((N, *dims[:-1]), self.parent_capacity * capacity, device=encoded.device, dtype=torch.int64)
                if self.parent_capacity is not None and capacity is not None
                else spans.sum(dim=-1)
            )
            routed = reduced.reshape(N, *dims[:-1], dims[-1] * output_width, C)
            routed_present = reduced_present.reshape(N, *dims[:-1], dims[-1] * output_width)
        elif output_width == 1:
            routed = reduced[:, 0]
            routed_present = reduced_present[:, 0]
            positions = positions[:, 0]
            slots = slots[:, 0]
        else:
            routed = reduced
            routed_present = reduced_present

        return Parcel(
            payload=routed,
            present=routed_present,
            origin=self.origin,
            destination=self.destination,
            positions=positions,
            spans=spans,
            slots=slots,
            batch_size=N,
        )

    def compute(
        self,
        inputs: torch.Tensor,
        present: torch.Tensor | None,
        *,
        packing: Packed | None = None,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode prepared sequences; absent presence means every coordinate is live."""
        for layer in cast(Iterable[RotaryTransformerEncoderLayer], self.encoder):
            if packing is None:
                if (
                    positions is not None
                    and type(layer) is RotaryTransformerEncoderLayer
                    and "forward" not in layer.__dict__
                ):
                    inputs = layer(inputs, present=present, positions=positions)
                else:
                    inputs = layer(inputs, present=present)
            else:
                inputs = layer(inputs, present=present, packing=packing)
        return inputs

    def contextualize(self, parcels: list[Parcel]) -> list[Parcel]:
        """Mix direct sibling fields only with fields at the same coordinate."""

        if self.coordinate_encoder is None:
            return parcels

        indices = [index for index, parcel in enumerate(parcels) if parcel.origin in self.coordinate_origins]
        if len(indices) < 2:
            return parcels

        selected = [parcels[index] for index in indices]
        shapes = {tuple(parcel.payload.shape) for parcel in selected}
        presence_shapes = {tuple(parcel.present.shape) for parcel in selected}
        if len(shapes) != 1 or len(presence_shapes) != 1:
            origins = ", ".join(str(parcel.origin) for parcel in selected)
            raise ValueError(f"coordinate fields in branch '{self.origin}' must share one shape: {origins}")

        payload = torch.stack([parcel.payload for parcel in selected], dim=-2)
        present = torch.stack([parcel.present for parcel in selected], dim=-1)
        field_count = payload.shape[-2]
        channel_count = payload.shape[-1]
        rows = math.prod(payload.shape[:-2])
        inputs = payload.reshape(rows, field_count, channel_count)
        presence = present.reshape(rows, field_count)
        prepared = (
            bool(field_count)
            and not customized((self.coordinate_encoder,))
            and not stochastic((self.coordinate_encoder,))
        )
        full = prepared and bool(presence.all())
        if prepared and not full:
            values, packing = pack(inputs, presence)
            if packing.indices.numel():
                values = self.coordinate_encoder(values, present=presence, packing=packing)
            mixed = unpack(values, packing, inputs, presence, (self.coordinate_encoder,))
        else:
            mixed = self.coordinate_encoder(
                inputs,
                present=None if full and inputs.is_cuda and 0 < inputs.shape[0] <= SDPA_BATCH_MAX else presence,
            )
        mixed = mixed.reshape(payload.shape)

        contextualized = list(parcels)
        for field_index, parcel_index in enumerate(indices):
            parcel = parcels[parcel_index]
            contextualized[parcel_index] = Parcel(
                payload=mixed[..., field_index, :],
                present=parcel.present,
                origin=parcel.origin,
                destination=parcel.destination,
                positions=parcel.positions,
                spans=parcel.spans,
                slots=parcel.slots,
                batch_size=parcel.payload.shape[0],
            )
        return contextualized

    if TYPE_CHECKING:
        __call__ = forward
