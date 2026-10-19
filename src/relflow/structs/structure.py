"""Structured schema nodes that group tensorfield requests."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import pydantic

from relflow.structs.enums import AttentionInput, AttentionMode, Overflow, OverflowInput
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.tree import Leaf, Mask, MaskInput, Node, Rate
from relflow.tensorfields import extensions as _extensions  # noqa: F401
from relflow.tensorfields.base import TENSORFIELDS

RequestTypes: TypeAlias = Leaf

OMITTED = object()


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


class Branch(Node):
    """Repeated nested object group in a `relflow` schema.

    Parent keywords supply child names, such as ``amount=rf.Number``. Use a
    ``fields`` mapping for names that collide with configuration options.
    ``length=None`` retains every collection slot and resolves its width per
    batch. A positive integer caps the collection and enables ``overflow`` to
    choose which excess slots to retain. ``reduction`` controls the context
    passed to the parent branch.

    During inference, unbounded collections in untrusted input are a
    denial-of-service (DoS/DDoS) attack vector: requests can exhaust memory or
    compute. Use finite branch lengths and enforce request-size limits before
    parsing public requests. Batch size alone does not bound nested collections.

    ``attention=None`` disables coordinate and sequence attention independently
    of the reduction. ``ffnx`` multiplies this node's feed-forward hidden width
    and defaults to 8 independently of its parent.
    ``mask`` accepts the same policies as a leaf and applies to its descendants.
    """

    model_config = pydantic.ConfigDict(extra="forbid", validate_default=True)

    type: Annotated[Literal["branch"], pydantic.Field(default="branch")] = "branch"
    query: str | None = None
    attention: AttentionMode | None = AttentionMode.mha
    length: Annotated[pydantic.StrictInt | None, pydantic.Field(gt=0)] = None
    overflow: Overflow | None = None
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    reduction: ReductionConfig | None = pydantic.Field(default_factory=Attention)
    # The before-validator normalizes this public Boolean default to a tuple.
    mask: tuple[Mask, ...] = cast(tuple[Mask, ...], pydantic.Field(default=False))
    fields: list[Self | pydantic.SerializeAsAny[pydantic.InstanceOf[Leaf]]] = pydantic.Field(default_factory=list)

    def __init__(
        self,
        *,
        fields: Mapping[str, Branch | Leaf | builtins.type[Leaf]] | None = None,
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
        dropout: Rate | None = cast(Rate | None, OMITTED),
        mask: MaskInput = False,
        type: Literal["branch"] = "branch",
        **children: Branch | Leaf | builtins.type[Leaf],
    ) -> None:
        from relflow.structs.experiment import bind_fields

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
            dropout=dropout,
            mask=mask,
            type=type,
        )
        # Pydantic owns ordinary defaults; omitted options remain available for
        # a model preset to fill without replacing explicit defaults or None.
        super().__init__(**{name: value for name, value in data.items() if value is not OMITTED})

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
                fields.append(cls.model_validate(payload))
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
