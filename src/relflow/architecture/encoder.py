from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, cast

import torch

from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.packed import Packed, block, customized, pack, stochastic, unpack
from relflow.architecture.pool import LearnedQueryCrossAttention, MeanPool
from relflow.structs.packages import Parcel
from relflow.structs.reduction import Attention, Mean
from relflow.structs.tree import Address, Leaf, Node

if TYPE_CHECKING:
    from relflow.structs.experiment import Schema


class RotaryTransformerEncoderLayer(torch.nn.Module):
    """Preserve absent coordinates through attention and feed-forward residuals."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        n_kv_heads: int,
        dropout: float,
        ffn_multiplier: int = 4,
    ) -> None:
        super().__init__()

        self.attention_norm = torch.nn.LayerNorm(normalized_shape=d_model)
        self.ffn_norm = torch.nn.LayerNorm(normalized_shape=d_model)

        self.attention = RotaryMultiheadAttention(
            d_model=d_model,
            nhead=nhead,
            n_kv_heads=n_kv_heads,
            dropout=dropout,
        )

        hidden = d_model * ffn_multiplier
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(in_features=d_model, out_features=hidden),
            torch.nn.GELU(),
            torch.nn.Dropout(p=dropout),
            torch.nn.Linear(in_features=hidden, out_features=d_model),
            torch.nn.Dropout(p=dropout),
        )

    def forward(self, inputs: torch.Tensor, present: torch.Tensor, *, packing: Packed | None = None) -> torch.Tensor:
        if packing is not None:
            return block(self, inputs, packing)
        if present.dtype != torch.bool or tuple(present.shape) != tuple(inputs.shape[:2]):
            raise ValueError(
                f"encoder presence must have bool shape {tuple(inputs.shape[:2])}, "
                f"got {tuple(present.shape)} with dtype {present.dtype}"
            )
        inputs = inputs.masked_fill(~present.unsqueeze(-1), 0.0)
        normed = self.attention_norm(inputs)
        inputs = inputs + self.attention(normed, normed, normed, key_padding_mask=~present)
        inputs = inputs + self.ffn(self.ffn_norm(inputs))
        return inputs.masked_fill(~present.unsqueeze(-1), 0.0)

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
                    )
                )

        self.encoder = torch.nn.ModuleList(layers)

        child_width = sum(
            int(child.active) if isinstance(child, Leaf) else schema.branch_outputs[child.address]
            for child in branch.fields
        )
        self.input_width = branch.length * child_width
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
                    mass_capacity=max(1, self.input_width),
                )
            case _:
                raise ValueError(f"unsupported branch reduction: {branch.reduction}")

    def forward(self, parcels: list[Parcel]) -> Parcel:
        if not parcels:
            raise ValueError(f"branch encoder '{self.origin}' requires at least one child parcel")

        unknown = [parcel.origin for parcel in parcels if parcel.origin not in self.child_order]
        if unknown:
            origins = ", ".join(repr(str(origin)) for origin in unknown)
            raise ValueError(f"branch encoder '{self.origin}' received parcel(s) from non-child address(es): {origins}")
        parcels = sorted(parcels, key=lambda parcel: self.child_order[parcel.origin])
        parcels = self.contextualize(parcels)
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
        encoded: torch.Tensor = concatenated.reshape(-1, L, C)
        present = present.reshape(-1, L)
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
            selected = selected.masked_fill(~selected_present.unsqueeze(-1), 0.0)
            evidence = selected

            if (
                self.encoder
                and not customized(self.encoder)
                and not stochastic(cast(Iterable[RotaryTransformerEncoderLayer], self.encoder))
                and not bool(selected_present.all())
            ):
                values, packing = pack(selected, selected_present)
                values = self.compute(values, selected_present, packing=packing)
                selected = unpack(
                    values,
                    packing,
                    selected,
                    selected_present,
                    cast(Iterable[RotaryTransformerEncoderLayer], self.encoder),
                )
            else:
                selected = self.compute(selected, selected_present)

            match self.reduction:
                case None:
                    selected_output = selected
                    selected_output_present = selected_present
                case _:
                    assert self.pool is not None
                    if isinstance(self.pool, LearnedQueryCrossAttention):
                        selected_output = self.pool(selected, present=selected_present, evidence=evidence)
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
        if dims:
            routed = reduced.reshape(N, *dims[:-1], dims[-1] * output_width, C)
            routed_present = reduced_present.reshape(N, *dims[:-1], dims[-1] * output_width)
        elif output_width == 1:
            routed = reduced[:, 0]
            routed_present = reduced_present[:, 0]
        else:
            routed = reduced
            routed_present = reduced_present

        return Parcel(
            payload=routed,
            present=routed_present,
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    def compute(self, inputs: torch.Tensor, present: torch.Tensor, *, packing: Packed | None = None) -> torch.Tensor:
        """Encode prepared sequences without routing or data-dependent selection."""
        for layer in cast(Iterable[RotaryTransformerEncoderLayer], self.encoder):
            if packing is None:
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
        inputs = payload.reshape(-1, field_count, channel_count)
        presence = present.reshape(-1, field_count)
        if (
            field_count
            and not customized((self.coordinate_encoder,))
            and not stochastic((self.coordinate_encoder,))
            and not bool(presence.all())
        ):
            values, packing = pack(inputs, presence)
            if packing.indices.numel():
                values = self.coordinate_encoder(values, present=presence, packing=packing)
            mixed = unpack(values, packing, inputs, presence, (self.coordinate_encoder,))
        else:
            mixed = self.coordinate_encoder(inputs, present=presence)
        mixed = mixed.reshape(payload.shape)

        contextualized = list(parcels)
        for field_index, parcel_index in enumerate(indices):
            parcel = parcels[parcel_index]
            contextualized[parcel_index] = Parcel(
                payload=mixed[..., field_index, :],
                present=parcel.present,
                origin=parcel.origin,
                destination=parcel.destination,
                batch_size=parcel.payload.shape[0],
            )
        return contextualized

    if TYPE_CHECKING:
        __call__ = forward
