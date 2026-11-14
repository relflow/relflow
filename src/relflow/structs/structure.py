"""Structured schema nodes that group tensorfield requests."""

from __future__ import annotations

import builtins
import datetime
from collections.abc import Mapping
from types import MappingProxyType
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import pydantic

from relflow.structs import declaration
from relflow.structs.enums import AttentionInput, AttentionMode, Direction, Overflow, OverflowInput
from relflow.structs.grain import Coordinates, GrainConfig, Steps
from relflow.structs.mixer import Conv, MixerConfig, Transformer
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.tree import Leaf, Mask, MaskInput, Node, Rate
from relflow.tensorfields import extensions as _extensions  # noqa: F401
from relflow.tensorfields.base import TENSORFIELDS

__all__ = ["Branch", "Mask", "RequestTypes", "OMITTED"]

RequestTypes: TypeAlias = Leaf

OMITTED = object()

# Options left at these values are omitted from dumps, so a plain branch keeps
# the schema, YAML and checkpoint text it had before the options existed.
IMPLICIT: Mapping[str, object] = MappingProxyType(
    {
        "grain": Coordinates(),
        "mixer": Transformer(),
        "direction": Direction.bidirectional,
        "time_index": None,
        "time_step": None,
    }
)


def bounds(branch: Branch) -> None:
    """Keep a collection cap and its overflow policy as one configuration."""

    if branch.length is not None and (
        not isinstance(branch.length, int) or isinstance(branch.length, bool) or branch.length < 1
    ):
        raise ValueError(
            f"branch '{branch.address or branch.name}' length must be a positive integer or None, got {branch.length!r}"
        )
    if branch.length is None:
        if branch.overflow is not None:
            location = f" '{branch.address or branch.name}'" if branch.address or branch.name else ""
            raise ValueError(
                f"unbounded branch{location} cannot configure overflow={branch.overflow!r}; "
                "set a finite length or remove the overflow policy"
            )
    elif branch.overflow is None:
        branch.overflow = Overflow.head


class BranchMeta(declaration.Declarations, type(Node)):
    """Collect branch children before Pydantic processes configuration fields."""


class Branch(Node, metaclass=BranchMeta):
    """Repeated nested object group in a `relflow` schema.

    Parent keywords supply child names, such as ``amount=rf.Number``. Use a
    ``fields`` mapping for names that collide with configuration options.
    ``length=None`` retains every collection slot and resolves its width per
    batch. A positive integer caps the collection and enables ``overflow`` to
    choose which excess slots to retain. ``overflow="random"`` draws a full,
    contiguous window during training and keeps the tail during validation,
    testing, and prediction. Crops replay from the data seed, epoch, batch
    ordinal, branch address, and parent position within the batch; changing
    batching or row order can change them. ``reduction`` controls the context
    passed to the parent branch.

    During inference, unbounded collections in untrusted input are a
    denial-of-service (DoS/DDoS) attack vector: requests can exhaust memory or
    compute. Use finite branch lengths and enforce request-size limits before
    parsing public requests. Batch size alone does not bound nested collections.

    ``attention=None`` disables coordinate and sequence attention independently
    of the reduction. ``ffnx`` multiplies this node's feed-forward hidden width
    and defaults to 8 independently of its parent. ``grain`` chooses what one
    sequence token stands for, ``mixer`` chooses the sequence mixer (``None``
    disables sequence mixing while keeping coordinate attention), and
    ``direction`` sets which steps a token may see. ``time_index`` names a
    child leaf whose values give each row a position in time, relative to the
    first row of its record and non-decreasing; ``time_step`` is the unit of
    those positions. They set attention positions and a learned gap embedding;
    they do not resample rows.
    ``mask`` accepts the same policies as a leaf and applies to its descendants.
    Subclasses may declare children in their class bodies; constructing those
    subclasses configures the enclosing branch without replacing children.
    A bare branch subclass uses its default options, just like a bare leaf class.
    """

    model_config = pydantic.ConfigDict(extra="forbid", validate_default=True)

    type: Annotated[Literal["branch"], pydantic.Field(default="branch")] = "branch"
    query: str | None = None
    attention: AttentionMode | None = AttentionMode.mha
    length: Annotated[pydantic.StrictInt | None, pydantic.Field(gt=0)] = None
    overflow: Overflow | None = None
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    reduction: ReductionConfig | None = pydantic.Field(default_factory=Attention)
    grain: GrainConfig = pydantic.Field(default_factory=Coordinates)
    mixer: MixerConfig | None = pydantic.Field(default_factory=Transformer)
    direction: Direction = Direction.bidirectional
    time_index: str | None = None
    time_step: float | datetime.timedelta | None = None
    # The before-validator normalizes this public Boolean default to a tuple.
    mask: tuple[Mask, ...] = cast(tuple[Mask, ...], pydantic.Field(default=False))
    fields: list[Branch | pydantic.SerializeAsAny[pydantic.InstanceOf[Leaf]]] = pydantic.Field(default_factory=list)

    def __init__(
        self,
        *,
        fields: Mapping[str, Branch | Leaf | builtins.type[Leaf] | builtins.type[Branch]] | None = None,
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        length: int | None = None,
        overflow: OverflowInput | None = None,
        attention: AttentionInput = cast(AttentionInput, OMITTED),
        n_layers: int = cast(int, OMITTED),
        n_heads: int = cast(int, OMITTED),
        ffnx: int = cast(int, OMITTED),
        reduction: ReductionConfig | None = cast(ReductionConfig | None, OMITTED),
        grain: GrainConfig = cast(GrainConfig, OMITTED),
        mixer: MixerConfig | None = cast(MixerConfig | None, OMITTED),
        direction: Direction = cast(Direction, OMITTED),
        time_index: str | None = cast(str | None, OMITTED),
        time_step: float | datetime.timedelta | None = cast(float | datetime.timedelta | None, OMITTED),
        dropout: Rate | None = cast(Rate | None, OMITTED),
        mask: MaskInput = False,
        type: Literal["branch"] = "branch",
        **children: Branch | Leaf | builtins.type[Leaf] | builtins.type[Branch],
    ) -> None:
        from relflow.structs.experiment import bind_fields

        fields = declaration.fields(builtins.type(self), fields, children)
        data: dict[str, Any] = dict(
            fields=bind_fields(fields, children),
            query=query,
            description=description,
            embed=embed,
            length=length,
            overflow=overflow,
            attention=attention,
            n_layers=n_layers,
            n_heads=n_heads,
            ffnx=ffnx,
            reduction=reduction,
            grain=grain,
            mixer=mixer,
            direction=direction,
            time_index=time_index,
            time_step=time_step,
            dropout=dropout,
            mask=mask,
            type=type,
        )
        # Pydantic owns ordinary defaults; omitted options remain available for
        # a model preset to fill without replacing explicit defaults or None.
        super().__init__(**{name: value for name, value in data.items() if value is not OMITTED})

    @classmethod
    def chronological(
        cls,
        *,
        length: int | None = None,
        overflow: OverflowInput | None = cast(OverflowInput | None, OMITTED),
        grain: GrainConfig = cast(GrainConfig, OMITTED),
        mixer: MixerConfig | None = cast(MixerConfig | None, OMITTED),
        direction: Direction = cast(Direction, OMITTED),
        **options: Any,
    ) -> Self:
        """Build a branch of ordered rows with time series defaults.

        The result is a plain ``Branch``: one token per row (``grain=Steps()``),
        a dilated convolution mixer (``mixer=Conv()``) and causal steps
        (``direction=Direction.causal``). A bounded branch keeps the most
        recent rows (``overflow="tail"``). Every explicit argument replaces its
        default. Rows are positioned by their count unless ``time_index`` is
        set, in which case rows also receive a learned time-gap embedding.

        The causal default suits forecasting and per-row targets. Classification
        and other whole-series targets need
        ``direction=rf.Direction.bidirectional``, because a causal branch lets
        no row see the rows after it.

        Args:
            length: Row cap, or ``None`` to keep every row.
            overflow: Which rows to keep beyond ``length``; ``"tail"`` when omitted and bounded.
            grain: What one sequence token stands for.
            mixer: The sequence mixer.
            direction: Which steps a token may see.
            **options: Any other ``Branch`` option, child field or ``fields`` mapping.

        Returns:
            A plain ``Branch``.

        Raises:
            ValueError: If the resulting options contradict each other, such as
                ``mixer=Conv()`` without ``grain=Steps()`` or a ``time_index``
                that names no child. The message names the branch and the remedy.
        """
        defaults: dict[str, Any] = {"grain": Steps(), "mixer": Conv(), "direction": Direction.causal}
        if length is not None:
            defaults["overflow"] = Overflow.tail
        given: dict[str, Any] = {
            name: value
            for name, value in (("overflow", overflow), ("grain", grain), ("mixer", mixer), ("direction", direction))
            if value is not OMITTED
        }
        merged: dict[str, Any] = {**defaults, **given, **options}
        return cls(length=length, **merged)

    @pydantic.model_serializer(mode="wrap")
    def serialize_options(self, handler: pydantic.SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Dump the branch without options that still hold their implicit value."""
        payload = handler(self)
        implicit = {name for name, value in IMPLICIT.items() if getattr(self, name) == value}
        return {key: value for key, value in payload.items() if key not in implicit}

    @pydantic.field_validator("mask", mode="before")
    @classmethod
    def check_mask(cls, value: Any) -> tuple[Mask, ...]:
        return Mask.normalize(value)

    @pydantic.field_validator("fields", mode="before")
    @classmethod
    def check_fields(cls, value: Any) -> Any:
        """Resolve serialized leaf requests against the live extension registry."""

        if not isinstance(value, (list, tuple)):
            return value

        fields: list[Any] = []
        for field in value:
            if type(field) is Leaf:
                from relflow.structs.experiment import bind_tree_field

                fields.append(bind_tree_field(cast(str, field.name), field))
                continue
            if not isinstance(field, Mapping):
                fields.append(field)
                continue

            payload = dict(field)
            field_type = payload.get("type")
            if field_type == "branch":
                fields.append(Branch.model_validate(payload))
                continue
            if not isinstance(field_type, str) or field_type not in TENSORFIELDS:
                raise ValueError(f"unknown tensor field type: {field_type}")

            fields.append(TENSORFIELDS[field_type].Request.model_validate(payload))

        return fields

    @pydantic.model_validator(mode="after")
    def check_overflow(self) -> Self:
        bounds(self)
        return self

    @pydantic.model_validator(mode="after")
    def check_query(self) -> Self:
        if self.query is not None:
            from relflow.data.query import compile

            compile(self.query)
        return self

    @pydantic.model_validator(mode="after")
    def check_time_index(self) -> Self:
        if self.time_index is None:
            return self
        child = next((field for field in self.fields if field.name == self.time_index), None)
        where = f"branch '{self.address or self.name}'"
        if child is None:
            raise ValueError(
                f"{where} sets time_index={self.time_index!r}, but has no child named {self.time_index!r}; "
                "name an existing child leaf or add one"
            )
        if isinstance(child, Branch):
            raise ValueError(
                f"{where} sets time_index={self.time_index!r}, which must name a leaf, not a child branch; "
                "name a leaf child holding the time of each row"
            )
        return self

    @pydantic.model_validator(mode="after")
    def check_time_step(self) -> Self:
        if self.time_step is None:
            return self
        where = f"branch '{self.address or self.name}'"
        if self.time_index is None:
            raise ValueError(
                f"{where} sets time_step={self.time_step!r}, which requires time_index; name the time child"
            )
        step = self.time_step
        seconds = step.total_seconds() if isinstance(step, datetime.timedelta) else step
        if seconds <= 0:
            raise ValueError(f"{where} time_step must be positive, got {self.time_step!r}")
        return self

    @pydantic.model_validator(mode="after")
    def check_steps(self) -> Self:
        if not isinstance(self.grain, Steps):
            return self
        unreduced = next((f for f in self.fields if isinstance(f, Branch) and f.reduction is None), None)
        if unreduced is None:
            return self
        raise ValueError(
            f"branch '{self.address or self.name}' sets grain=Steps(), so rows pool to one token, but child branch "
            f"'{unreduced.name}' needs a reduction; set reduction=rf.Attention() or rf.Mean() on it"
        )

    @pydantic.model_validator(mode="after")
    def check_conv(self) -> Self:
        if not isinstance(self.mixer, Conv) or isinstance(self.grain, Steps):
            return self
        raise ValueError(
            f"branch '{self.address or self.name}' sets mixer=Conv(), which convolves over rows and "
            "requires grain=Steps(); set grain=rf.Steps() or mixer=rf.Transformer()"
        )

    def model_post_init(self, __context: object) -> None:
        for field in self.fields:
            field.parent = self

    @pydantic.model_validator(mode="after")
    def check_unique_child_names(self) -> Self:
        seen: set[str | None] = set()
        for field in self.fields:
            if field.name is None:
                raise ValueError("branch children require names; supply child keywords or a fields mapping")
            if field.name in seen:
                raise ValueError(f"duplicate field name: {field.name}")
            seen.add(field.name)

        return self

    def post_bind_validate(self) -> None:
        bounds(self)
        if len(self.mask) == 0:
            return None

        active_leaves = [
            descendant
            for descendant in getattr(self, "descendants", ())
            if isinstance(descendant, Leaf) and getattr(descendant, "active", True)
        ]
        if not active_leaves:
            raise ValueError(f"branch '{self.address}' has a mask but no active descendant leaves")

        return None
