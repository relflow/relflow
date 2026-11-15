from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import torch
from tensordict import TensorDict

from relflow.architecture.anchor import anchor
from relflow.architecture.attention import RotaryMultiheadAttention
from relflow.architecture.blocks import feed_forward, norm
from relflow.architecture.contrastive import hierarchical
from relflow.architecture.direction import rules, stepped
from relflow.architecture.elapsed import Elapsed, gaps
from relflow.architecture.grain import extents, grains, rows
from relflow.architecture.mixers import TransformerMixer, mixers
from relflow.architecture.packed import SDPA_BATCH_MAX, Packed, block, customized, pack, unpack
from relflow.architecture.reducers import Reducer, reducers
from relflow.architecture.views import crops, pairs
from relflow.structs.experiment import row_width
from relflow.structs.grain import Coordinates, Steps
from relflow.structs.packages import Parcel
from relflow.structs.tree import Address, Leaf, Node

if TYPE_CHECKING:
    from relflow.data.arrow import Geometry
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


def locate(parcel: Parcel, *, schema: Schema, geometry: TensorDict | None) -> Parcel:
    """Attach structural leaf coordinates independently of values and masking."""
    branch = cast("Branch", schema.requests[parcel.origin].parent)
    capacity = branch.capacity
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
    # Geometry slots exclude null item records, so no routed token stands for a missing item.
    parcel.slots = layout.slots if layout is not None else torch.ones_like(parcel.present, dtype=torch.bool)
    return parcel


def stepwise(parcels: list[Parcel], *, order: dict[Address, int] | None = None) -> torch.Tensor:
    """Number each token by the row of its item, in the token order of ``sequence``.

    Args:
        parcels: Parcels of one branch instance grid.
        order: Field order by origin, as for ``sequence``.

    Returns:
        An int64 tensor of shape ``(*instance dims, L)``; padding tokens take the last row.
    """
    if order is not None:
        parcels = sorted(parcels, key=lambda parcel: order[parcel.origin])
    _, widths = extents(parcels)
    steps = []
    for parcel, width in zip(parcels, widths, strict=True):
        count = parcel.payload.shape[-2]
        tokens = torch.arange(count, device=width.device).expand(*width.shape[:-1], count).contiguous()
        rows = torch.searchsorted(width.cumsum(dim=-1).contiguous(), tokens, right=True)
        # Padding tokens lie past the last item and would index beyond the rows; they join the last row.
        steps.append(rows.clamp_max(max(width.shape[-1] - 1, 0)))
    return torch.cat(steps, dim=-1)


def timeline(*, structural: torch.Tensor, steps: torch.Tensor, times: torch.Tensor | None) -> torch.Tensor:
    """Give every token the time position of its row, or keep the structural positions.

    Args:
        structural: Item-major integer positions of shape ``(rows, L)``.
        steps: Row index of every token, shape ``(rows, L)``.
        times: Float time of every row slot, any shape that flattens to ``(rows, T)``, or ``None``.

    Returns:
        ``structural`` itself without times, else float positions of shape ``(rows, L)``;
        steps outside the time geometry receive zero.
    """
    if times is None:
        return structural
    rows = times.reshape(steps.shape[0], -1)
    if rows.shape[-1] == 0:
        return rows.new_zeros(steps.shape)
    valid = (steps >= 0) & (steps < rows.shape[-1])
    return rows.gather(1, steps.clamp(min=0, max=rows.shape[-1] - 1)).masked_fill(~valid, 0.0)


def sequence(
    parcels: list[Parcel], *, order: dict[Address, int] | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Number tokens item-major within each branch instance.

    Every item owns one contiguous block of positions holding its leaf tokens
    and then each child branch's tokens, in field order, so same-item offsets
    never depend on how many items a record holds. Storage stays field-major;
    spans sum every parcel's instance span.
    """
    if order is not None:
        parcels = sorted(parcels, key=lambda parcel: order[parcel.origin])
    first = parcels[0]
    device = first.payload.device
    shape = tuple(first.payload.shape[:-2])
    items, widths = extents(parcels)
    stride = torch.stack(widths).sum(dim=0)
    start = stride.cumsum(dim=-1) - stride
    lead = torch.zeros_like(stride)
    spans = torch.zeros(shape, dtype=torch.int64, device=device)
    positions, slots = [], []
    for parcel, width in zip(parcels, widths, strict=True):
        count = parcel.payload.shape[-2]
        repeats = count // items if items else 0
        local = (
            torch.arange(count, device=device).expand(*shape, count) if parcel.positions is None else parcel.positions
        )
        heads = width.cumsum(dim=-1) - width
        positions.append((start + lead - heads).repeat_interleave(repeats, dim=-1) + local)
        slots.append(torch.ones_like(parcel.present) if parcel.slots is None else parcel.slots)
        spans = spans + (count if parcel.spans is None else parcel.spans)
        lead = lead + width
    return torch.cat(positions, dim=-1), spans, torch.cat(slots, dim=-1)


arrangements: Mapping[type, Callable[[list[Parcel]], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]] = (
    MappingProxyType({Coordinates: sequence, Steps: rows})
)


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
        allowed: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        causal: bool = False,
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
            allowed=allowed,
            mask=mask,
            causal=causal,
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

        attention = branch.attention
        self.encoder = mixers[type(branch.mixer)](
            branch.mixer, branch=branch, schema=schema, dropout=dropout, rule=rules[branch.direction]
        )

        widths = [
            int(child.active) if isinstance(child, Leaf) else schema.branch_outputs[child.address]
            for child in branch.fields
        ]
        child_width = row_width(branch.grain, widths)
        self.input_width = None if branch.capacity is None or child_width is None else branch.capacity * child_width
        # Skipped targets occupy slots but never carry a present token, so the
        # count lanes normalize by embedded leaves only; adding a hidden target
        # then leaves every existing prediction in place.
        visible = [
            int(child.embedded) if isinstance(child, Leaf) else schema.branch_outputs[child.address]
            for child in branch.fields
        ]
        visible_width = row_width(branch.grain, visible)
        self.visible_width = (
            None if branch.capacity is None or visible_width is None else branch.capacity * visible_width
        )
        self.parent_capacity = getattr(branch.parent, "capacity", 1)
        self.child_order = {child.address: index for index, child in enumerate(branch.fields)}
        # A child branch reads its parent's own fields before encoding: one
        # projection of their mean reaches every token and every reduction
        # query, so items know the record they belong to. It starts at a
        # standard scale: from zero, the stack takes twice as long to learn
        # to read the parent at all.
        parent_fields = getattr(branch.parent, "fields", None) if address != Address() else None
        context_fields = [child for child in (parent_fields or ()) if isinstance(child, Leaf) and child.embedded]
        self.context_origins = frozenset(child.address for child in context_fields)
        self.conditioning: torch.nn.Linear | None = None
        if context_fields:
            self.conditioning = torch.nn.Linear(schema.d_model, schema.d_model, bias=False)
            torch.nn.init.xavier_uniform_(self.conditioning.weight)

        self.coordinate_origins = frozenset(
            child.address for child in branch.fields if isinstance(child, Leaf) and child.active
        )
        # A reduced child branch contributes its summary tokens to each item's
        # coordinate, so leaf-times-child interactions form before reduction.
        self.coordinate_branches: dict[Address, int] = {
            child.address: cast(int, schema.branch_outputs[child.address])
            for child in branch.fields
            if not isinstance(child, Leaf)
            and cast("Branch", child).reduction is not None
            and schema.branch_outputs[child.address]
        }
        coordinate_tokens = len(self.coordinate_origins) + sum(self.coordinate_branches.values())
        self.coordinate_encoder: RotaryTransformerEncoderLayer | None = None
        if attention is not None and coordinate_tokens > 1:
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

        self.time_index = branch.time_index
        self.timed = branch.time_index is not None
        self.contrastive = branch.contrastive
        # Crops cut whole rows, so a contrastive branch needs the row of every token.
        self.stepped = stepped[branch.direction] or self.timed or self.contrastive is not None
        self.grain = grains[type(branch.grain)](branch.grain, branch=branch, schema=schema)
        self.arrangement = arrangements[type(branch.grain)]

        self.reduction = branch.reduction
        # An unbounded visible width leaves the lanes to the capacity observed while training.
        capacity = None if self.visible_width is None else max(1, self.visible_width)
        self.pool: Reducer | None = (
            None
            if branch.reduction is None
            else reducers[type(branch.reduction)](
                branch.reduction, branch=branch, schema=schema, capacity=capacity, dropout=dropout
            )
        )
        self.n_outputs: int | None = None if self.pool is None else self.pool.n_context
        # Built last so a plain branch keeps its parameter order and state dict.
        self.elapsed: Elapsed | None = Elapsed(d_model=schema.d_model) if self.timed else None

    def admit(self, parcels: list[Parcel]) -> list[Parcel]:
        """Check that every parcel comes from a child and order them by field order.

        Args:
            parcels: Child parcels addressed to this branch.

        Returns:
            The parcels sorted by the declaration order of their origins.

        Raises:
            ValueError: If no parcel is given or one comes from a non-child address.
        """
        if not parcels:
            raise ValueError(f"branch encoder '{self.origin}' requires at least one child parcel")

        unknown = [parcel.origin for parcel in parcels if parcel.origin not in self.child_order]
        if unknown:
            origins = ", ".join(repr(str(origin)) for origin in unknown)
            raise ValueError(f"branch encoder '{self.origin}' received parcel(s) from non-child address(es): {origins}")
        return sorted(parcels, key=lambda parcel: self.child_order[parcel.origin])

    def inform(
        self, parcels: list[Parcel], *, context: list[Parcel] | None
    ) -> tuple[list[Parcel], torch.Tensor | None]:
        """Add the projected parent record to every token of every parcel.

        Args:
            parcels: Ordered child parcels.
            context: Parcels of the parent fields, or ``None`` for no conditioning.

        Returns:
            The conditioned parcels and the ``(rows, d_model)`` projection, or the
            unchanged parcels and ``None`` when nothing conditions this branch.
        """
        conditioned: torch.Tensor | None = None
        if self.conditioning is not None and context:
            # The parent's record reaches every token before its fields are mixed,
            # so leaf-times-parent interactions form where leaf-times-leaf ones do.
            conditioned = self.condition(context, math.prod(parcels[0].payload.shape[:-2]))
            parcels = [
                Parcel(
                    payload=(
                        parcel.payload + conditioned.reshape(*parcel.payload.shape[:-2], 1, conditioned.shape[-1])
                    ).masked_fill(~parcel.present.unsqueeze(-1), 0.0),
                    present=parcel.present,
                    origin=parcel.origin,
                    destination=parcel.destination,
                    positions=parcel.positions,
                    spans=parcel.spans,
                    slots=parcel.slots,
                    widths=parcel.widths,
                    batch_size=parcel.payload.shape[0],
                )
                for parcel in parcels
            ]
        return parcels, conditioned

    def mix(
        self,
        selected: torch.Tensor,
        *,
        present: torch.Tensor,
        positions: torch.Tensor | None,
        steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the sequence layers over the rows that hold at least one token.

        Args:
            selected: Tokens of shape ``(rows, L, d_model)`` with padding zeroed.
            present: Boolean token presence of shape ``(rows, L)``.
            positions: Structural positions of shape ``(rows, L)``, or ``None`` when not positioned.
            steps: Row index of each token, shape ``(rows, L)``, or ``None`` when the direction ignores steps.

        Returns:
            Mixed tokens of shape ``(rows, L, d_model)``.
        """
        from relflow.architecture.compiler import canonical

        prepared = (
            type(self.encoder) is TransformerMixer
            and bool(self.encoder)
            and not customized([self.encoder], additional=(TransformerMixer,))
        )
        full = prepared and bool(present.all())
        # Plain branches keep the exact compute call forms they had before steps existed.
        compute = self.compute if steps is None else partial(self.compute, steps=steps)
        if prepared and not full:
            values, packing = pack(
                selected, present, positions, steps=steps, rule=cast(TransformerMixer, self.encoder).rule
            )
            values = compute(values, present, packing=packing)
            selected = unpack(
                values,
                packing,
                selected,
                present,
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
            if positions is not None:
                selected = compute(selected, None if unmasked else present, positions=positions)
            else:
                selected = compute(selected, None if unmasked else present)
        return selected

    def contrast(
        self,
        tokens: torch.Tensor,
        *,
        present: torch.Tensor,
        positions: torch.Tensor | None,
        steps: torch.Tensor,
        views: torch.Tensor | None,
        extent: int,
        hidden: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Encode two crops of every record with the branch mixer and contrast them.

        The crops share the parameters and the direction rule of the full record,
        which alone continues to the parent and the decoders.

        Args:
            tokens: Tokens of shape ``(rows, L, d_model)`` with padding zeroed.
            present: Boolean token presence of shape ``(rows, L)``.
            positions: Positions of shape ``(rows, L)``, or ``None`` when not positioned.
            steps: Row index of each token, shape ``(rows, L)``.
            views: First row of each crop, shape ``(rows, 2)``, or ``None`` outside training.
            extent: Rows of the branch axis.
            hidden: Source rows each crop hides, shape ``(rows, 2, extent)``, or ``None`` to hide nothing.

        Returns:
            Per-record shares of the loss, shape ``(rows,)``, or ``None`` when there are no views.
        """
        if self.contrastive is None or views is None:
            return None
        cut = crops(
            tokens,
            present=present,
            positions=positions,
            steps=steps,
            starts=views,
            extent=extent,
            width=self.contrastive.views.width,
            hidden=hidden,
        )
        # Dense attention over a row without a live token returns NaN, so only live crops are mixed.
        indices = cut.present.any(dim=1).nonzero(as_tuple=False).reshape(-1)
        mixed = tokens.new_zeros(cut.tokens.shape)
        if indices.numel():
            live = cut.present.index_select(0, indices)
            chosen = self.encoder(
                cut.tokens.index_select(0, indices).masked_fill(~live.unsqueeze(-1), 0.0),
                present=live,
                positions=None if cut.positions is None else cut.positions.index_select(0, indices),
                steps=cut.steps.index_select(0, indices),
                packing=None,
            )
            mixed = mixed.index_copy(0, indices, chosen.masked_fill(~live.unsqueeze(-1), 0.0).to(dtype=mixed.dtype))
        elif torch.is_grad_enabled():
            mixed = mixed + anchor(self)
        first, second, aligned = pairs(mixed, views=cut)
        return hierarchical(first, second, present=aligned)

    def credit(
        self, shares: torch.Tensor | None, *, indices: torch.Tensor, rows: int, like: torch.Tensor, views: bool
    ) -> torch.Tensor | None:
        """Spread the shares of the live rows over every row of the batch.

        Args:
            shares: Shares of the rows at ``indices``, or ``None`` when no row was live.
            indices: Rows that held a live token.
            rows: Number of rows in the batch.
            like: Tensor whose dtype and device the result takes.
            views: Whether this batch trains the contrastive objective.

        Returns:
            Shares of shape ``(rows,)``, or ``None`` outside contrastive training. A batch
            without live rows returns zeros that keep every parameter in the graph, so every
            distributed rank backpropagates through the same parameters.
        """
        if not views:
            return None
        spread = like.new_zeros(rows)
        if shares is not None:
            return spread.index_copy(0, indices, shares.to(dtype=spread.dtype))
        if torch.is_grad_enabled():
            return spread + anchor(self)
        return spread

    def reduce(
        self,
        selected: torch.Tensor,
        *,
        present: torch.Tensor,
        evidence: torch.Tensor,
        context: torch.Tensor | None,
        positions: torch.Tensor | None,
        width: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reduce mixed tokens to this branch summary, or keep the sequence without a reduction.

        Args:
            selected: Mixed tokens of shape ``(rows, L, d_model)``.
            present: Boolean token presence of shape ``(rows, L)``.
            evidence: Tokens before mixing, with padding zeroed, used for presence mass.
            context: Parent projection of shape ``(rows, d_model)``, or ``None``.
            positions: Structural positions of shape ``(rows, L)``, or ``None``.
            width: Number of output tokens per row.

        Returns:
            The output tokens and their boolean presence.
        """
        if self.pool is None:
            return selected, present
        output = self.pool(
            selected,
            present=present,
            context=None if context is None else context.unsqueeze(1).expand(-1, width, -1),
            evidence=evidence,
            memory_positions=positions,
        )
        return output, torch.ones((selected.shape[0], width), dtype=torch.bool, device=present.device)

    def forward(
        self,
        parcels: list[Parcel],
        *,
        geometry: TensorDict | None = None,
        context: list[Parcel] | None = None,
        sequenced: bool = False,
    ) -> Parcel:
        """Encode one branch; ``sequenced`` also routes its encoded sequence to the decoders beneath it."""
        parcels = self.admit(parcels)
        parcels, conditioned = self.inform(parcels, context=context)
        parcels = self.grain(self.contextualize(parcels))
        positions, spans, slots = sequence(parcels)
        steps = stepwise(parcels) if self.stepped else None
        positioned = self.timed or any(parcel.positions is not None for parcel in parcels)
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
        if self.pool is not None:
            self.pool.observe(present)
        flat_steps = None if steps is None else steps.reshape(rows, L)
        flat_positions = positions.reshape(rows, L)
        if flat_steps is not None:
            flat_positions = timeline(structural=flat_positions, steps=flat_steps, times=self.clock(geometry))
        flat_gaps = self.lapse(geometry, steps=flat_steps)
        starts = self.starts(geometry, rows=rows)
        drawn = self.drawn(geometry, rows=rows)
        shares: torch.Tensor | None = None
        active = present.any(dim=1)
        indices = active.nonzero(as_tuple=False).reshape(-1)

        output_width = L if self.n_outputs is None else self.n_outputs
        count = indices.numel()
        complete = count > 0 and count == encoded.shape[0]
        shape = (encoded.shape[0], output_width, C)
        selected = encoded
        selected_output = encoded
        selected_output_present = present
        if count:
            selected = encoded if complete else encoded.index_select(0, indices)
            selected_present = present.index_select(0, indices)
            selected_positions = flat_positions if complete else flat_positions.index_select(0, indices)
            selected_steps = (
                None if flat_steps is None else flat_steps if complete else flat_steps.index_select(0, indices)
            )
            selected = selected.masked_fill(~selected_present.unsqueeze(-1), 0.0)
            if self.elapsed is not None and flat_gaps is not None:
                selected_gaps = flat_gaps if complete else flat_gaps.index_select(0, indices)
                selected = (selected + self.elapsed(selected_gaps)).masked_fill(~selected_present.unsqueeze(-1), 0.0)
            evidence = selected
            shares = self.contrast(
                selected,
                present=selected_present,
                positions=selected_positions if positioned else None,
                steps=cast(torch.Tensor, selected_steps),
                views=None if starts is None else starts if complete else starts.index_select(0, indices),
                hidden=None if drawn is None else drawn if complete else drawn.index_select(0, indices),
                extent=0
                if starts is None
                else cast("Geometry", cast(TensorDict, geometry)[self.origin]).slots.shape[-1],
            )
            selected_context = (
                None if conditioned is None else conditioned if complete else conditioned.index_select(0, indices)
            )

            selected = self.mix(
                selected,
                present=selected_present,
                positions=selected_positions if positioned else None,
                steps=selected_steps,
            )
            selected_output, selected_output_present = self.reduce(
                selected,
                present=selected_present,
                evidence=evidence,
                context=selected_context,
                positions=selected_positions if positioned else None,
                width=output_width,
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
        credited = self.credit(shares, indices=indices, rows=rows, like=encoded, views=starts is not None)
        objective = None if credited is None else credited.reshape(N, *dims)
        if self.n_outputs is None:
            return self.route(
                reduced,
                reduced_present,
                positions,
                spans,
                slots,
                capacity=self.input_width,
                geometry=geometry,
                objective=objective,
            )

        summary = self.route(
            reduced,
            reduced_present,
            torch.arange(output_width, device=encoded.device).expand(N, *dims, output_width),
            torch.full((N, *dims), output_width, device=encoded.device, dtype=torch.int64),
            torch.ones((N, *dims, output_width), device=encoded.device, dtype=torch.bool),
            capacity=self.n_outputs,
            geometry=geometry,
            objective=objective,
        )
        if not sequenced:
            return summary
        # The sequence stays at this branch: it is addressed to itself, never to the parent.
        # It is materialized only for branches with a decoder beneath them, since the full
        # width of a nested branch costs far more than its summary.
        if count and count == encoded.shape[0]:
            stacked = selected
        else:
            stacked = encoded.new_zeros(encoded.shape)
            if count:
                stacked = stacked.index_copy(0, indices, selected)
        summary.sequence = self.route(
            stacked.reshape(N, *dims, L, C),
            present.reshape(N, *dims, L),
            positions,
            spans,
            slots,
            capacity=self.input_width,
            geometry=geometry,
            destination=self.origin,
        )
        return summary

    def arrange(self, parcels: list[Parcel]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Number the tokens of this branch the way its grain sequences them.

        Args:
            parcels: The child parcels addressed to this branch.

        Returns:
            Positions, spans and slots of the sequence tokens, one per row under ``Steps``.
        """
        ordered = sorted(parcels, key=lambda parcel: self.child_order[parcel.origin])
        return self.arrangement(ordered)

    def lapse(self, geometry: TensorDict | None, *, steps: torch.Tensor | None) -> torch.Tensor | None:
        """Give every token the time gap that precedes its row, or ``None`` without ``time_index``.

        Args:
            geometry: Batch geometry carrying the branch times.
            steps: Row index of every token, shape ``(rows, L)``.

        Returns:
            Float gaps of shape ``(rows, L)`` in ``time_step`` units, or ``None``.
        """
        if self.elapsed is None or steps is None:
            return None
        times = cast(torch.Tensor, self.clock(geometry))
        slots = cast("Geometry", cast(TensorDict, geometry)[self.origin]).slots
        return gaps(times, slots=slots).reshape(steps.shape[0], -1).gather(1, steps)

    def clock(self, geometry: TensorDict | None) -> torch.Tensor | None:
        """Return the time of every row slot of this branch, or ``None`` without ``time_index``.

        Raises:
            ValueError: If the branch sets ``time_index`` but the batch geometry carries no times.
        """
        if not self.timed:
            return None
        entry = None if geometry is None else geometry.get(self.origin, None)
        times = getattr(entry, "times", None)
        if times is None:
            message = (
                f"branch '{self.origin}' sets time_index={self.time_index!r} but the batch geometry has no times; "
                "encode batches with model.encode or an rf data module so geometry carries them"
            )
            raise ValueError(message)
        return times

    def route(
        self,
        payload: torch.Tensor,
        present: torch.Tensor,
        positions: torch.Tensor,
        spans: torch.Tensor,
        slots: torch.Tensor,
        *,
        capacity: int | None,
        geometry: TensorDict | None,
        destination: Address | None = None,
        objective: torch.Tensor | None = None,
    ) -> Parcel:
        """Flatten one output per instance onto the parent's item axis, numbering positions item by item."""
        N, *dims, width, C = payload.shape
        widths: torch.Tensor | None = None
        if dims:
            parent = cast("Geometry", geometry[self.destination]) if geometry is not None else None
            if parent is not None:
                # A null parent item keeps its span so sibling positions stay put, but owns no tokens.
                retained = torch.arange(dims[-1], device=payload.device) < parent.lengths.unsqueeze(-1)
                slots = slots.masked_fill(~parent.slots.unsqueeze(-1), False)
                spans = spans.masked_fill(~retained, 0)
            # Each parent item owns this instance's span; positions run across items in order.
            widths = spans
            offsets = spans.cumsum(dim=-1) - spans
            positions = (positions + offsets.unsqueeze(-1)).reshape(N, *dims[:-1], dims[-1] * width)
            slots = slots.reshape(N, *dims[:-1], dims[-1] * width)
            spans = (
                torch.full((N, *dims[:-1]), self.parent_capacity * capacity, device=payload.device, dtype=torch.int64)
                if self.parent_capacity is not None and capacity is not None
                else spans.sum(dim=-1)
            )
            payload = payload.reshape(N, *dims[:-1], dims[-1] * width, C)
            present = present.reshape(N, *dims[:-1], dims[-1] * width)
        elif width == 1:
            payload, present, positions, slots = payload[:, 0], present[:, 0], positions[:, 0], slots[:, 0]

        return Parcel(
            payload=payload,
            present=present,
            origin=self.origin,
            destination=self.destination if destination is None else destination,
            positions=positions,
            spans=spans,
            slots=slots,
            widths=widths,
            objective=objective,
            batch_size=N,
        )

    def starts(self, geometry: TensorDict | None, *, rows: int) -> torch.Tensor | None:
        """Return the crop starts of this branch, shape ``(rows, 2)``, or ``None`` outside contrastive training."""
        entry = None if geometry is None else geometry.get(self.origin, None)
        views = getattr(entry, "views", None)
        return None if self.contrastive is None or views is None else views.reshape(rows, 2)

    def drawn(self, geometry: TensorDict | None, *, rows: int) -> torch.Tensor | None:
        """Return the rows each crop hides, shape ``(rows, 2, extent)``, or ``None`` when none are drawn."""
        entry = None if geometry is None else geometry.get(self.origin, None)
        hidden = getattr(entry, "hidden", None)
        return None if self.contrastive is None or hidden is None else hidden.reshape(rows, 2, -1)

    def condition(self, context: list[Parcel], rows: int) -> torch.Tensor:
        """Project the mean of the parent's present leaf embeddings at each instance of this branch."""
        assert self.conditioning is not None
        selected = [parcel for parcel in context if parcel.origin in self.context_origins]
        if selected:
            payload = torch.stack([parcel.payload for parcel in selected], dim=-2)
            present = torch.stack([parcel.present for parcel in selected], dim=-1)
            total = payload.masked_fill(~present.unsqueeze(-1), 0.0).sum(dim=-2)
            mean = (total / present.sum(dim=-1, keepdim=True).clamp_min(1)).reshape(-1, payload.shape[-1])
            if mean.shape[0] != rows:
                raise ValueError(
                    f"branch '{self.origin}' expected its parent's fields over {rows} instances, got {mean.shape[0]}"
                )
        else:
            mean = self.conditioning.weight.new_zeros((rows, self.conditioning.in_features))
        return self.conditioning(mean)

    def compute(
        self,
        inputs: torch.Tensor,
        present: torch.Tensor | None,
        *,
        packing: Packed | None = None,
        positions: torch.Tensor | None = None,
        steps: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode prepared sequences; absent presence means every coordinate is live.

        Layers receive the branch's structural positions whenever it has them.
        """
        return self.encoder(inputs, present=present, positions=positions, steps=steps, packing=packing)

    def contextualize(self, parcels: list[Parcel]) -> list[Parcel]:
        """Mix each item's fields: its leaves and the summaries of its reduced child branches."""

        if self.coordinate_encoder is None:
            return parcels

        # Tokens keep schema field order, so an appended field never moves an existing slot.
        selected = [
            index
            for index, parcel in enumerate(parcels)
            if parcel.origin in self.coordinate_origins or parcel.origin in self.coordinate_branches
        ]
        outputs = [self.coordinate_branches.get(parcels[index].origin, 1) for index in selected]
        if sum(outputs) < 2:
            return parcels

        leaves = [index for index in selected if parcels[index].origin in self.coordinate_origins]
        shapes = {tuple(parcels[index].payload.shape) for index in leaves}
        if len(shapes) > 1:
            origins = ", ".join(str(parcels[index].origin) for index in leaves)
            raise ValueError(f"coordinate fields in branch '{self.origin}' must share one shape: {origins}")
        if leaves:
            shape = tuple(parcels[leaves[0]].payload.shape)
        else:
            parcel = parcels[selected[0]]
            items = parcel.payload.shape[-2] // outputs[0]
            shape = (*parcel.payload.shape[:-2], items, parcel.payload.shape[-1])
        payloads, presences = [], []
        for index, width in zip(selected, outputs, strict=True):
            parcel = parcels[index]
            expected = (*shape[:-2], shape[-2] * width, shape[-1])
            if tuple(parcel.payload.shape) != expected:
                raise ValueError(
                    f"branch '{self.origin}' expected {expected} from '{parcel.origin}' for {width} token(s) per item, "
                    f"got {tuple(parcel.payload.shape)}"
                )
            payloads.append(parcel.payload.reshape(*shape[:-1], width, shape[-1]))
            presences.append(parcel.present.reshape(*shape[:-1], width))
        payload = torch.cat(payloads, dim=-2)
        present = torch.cat(presences, dim=-1)
        field_count = payload.shape[-2]
        channel_count = payload.shape[-1]
        rows = math.prod(payload.shape[:-2])
        inputs = payload.reshape(rows, field_count, channel_count)
        presence = present.reshape(rows, field_count)
        prepared = bool(field_count) and not customized((self.coordinate_encoder,))
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
        cursor = 0
        for index, width in zip(selected, outputs, strict=True):
            parcel = parcels[index]
            contextualized[index] = Parcel(
                payload=mixed[..., cursor : cursor + width, :].reshape(parcel.payload.shape),
                present=parcel.present,
                origin=parcel.origin,
                destination=parcel.destination,
                positions=parcel.positions,
                spans=parcel.spans,
                slots=parcel.slots,
                widths=parcel.widths,
                batch_size=parcel.payload.shape[0],
            )
            cursor += width
        return contextualized

    if TYPE_CHECKING:
        __call__ = forward
