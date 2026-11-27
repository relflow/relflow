"""Structured schema nodes that group tensorfield requests."""

from __future__ import annotations

import builtins
from collections.abc import Mapping
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import pydantic

from relflow.structs import declaration
from relflow.structs.enums import AttentionInput, AttentionMode, Overflow, OverflowInput
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.tasks import Names, TaskInput, selection
from relflow.structs.tree import Field, HideInput, Leaf, Node, Rate, selections
from relflow.structs.twigs import TWIGS, Twig
from relflow.tensorfields import extensions as _extensions  # noqa: F401
from relflow.tensorfields.base import TENSORFIELDS

__all__ = ["Branch", "RequestTypes", "OMITTED"]

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


class BranchMeta(declaration.Declarations, type(Node)):
    """Collect branch children before Pydantic processes configuration fields."""


class Branch(Field, metaclass=BranchMeta):
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
    passed to the parent branch. An ``Index`` child positions the records in
    this branch's attention by their index values instead of their order.

    During inference, unbounded collections in untrusted input are a
    denial-of-service (DoS/DDoS) attack vector: requests can exhaust memory or
    compute. Use finite branch lengths and enforce request-size limits before
    parsing public requests. Batch size alone does not bound nested collections.

    ``attention=None`` disables coordinate and sequence attention independently
    of the reduction. ``cross=True`` mixes this branch's fields with a rank-eight
    multiplicative cross. ``ffnx`` multiplies this node's feed-forward hidden width
    and defaults to 8 independently of its parent.
    ``hide`` hides selected rows across descendants, except direct children named by ``ignore``.
    ``ablate`` samples an entire child collection per parent during training.
    ``predict=True`` makes every active descendant leaf a supervised target.
    ``task=rf.reconstruct(...)`` opts into beta self-supervised training on selected descendant values.
    Subclasses may declare children in their class bodies; constructing those
    subclasses configures the enclosing branch without replacing children.
    A bare branch subclass uses its default options, just like a bare leaf class.
    """

    model_config = pydantic.ConfigDict(extra="forbid", validate_default=True)

    type: Annotated[Literal["branch"], pydantic.Field(default="branch")] = "branch"
    query: str | None = None
    attention: AttentionMode | None = AttentionMode.mha
    cross: pydantic.StrictBool = False
    length: Annotated[pydantic.StrictInt | None, pydantic.Field(gt=0)] = None
    overflow: Overflow | None = None
    n_layers: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    reduction: ReductionConfig | None = pydantic.Field(default_factory=Attention)
    ignore: Names = ()
    fields: list[
        Branch | pydantic.SerializeAsAny[pydantic.InstanceOf[Leaf]] | pydantic.SerializeAsAny[pydantic.InstanceOf[Twig]]
    ] = pydantic.Field(default_factory=list)

    def __init__(
        self,
        *,
        fields: Mapping[str, Branch | Leaf | Twig | builtins.type[Leaf] | builtins.type[Branch] | builtins.type[Twig]]
        | None = None,
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        length: int | None = None,
        overflow: OverflowInput | None = None,
        attention: AttentionInput = cast(AttentionInput, OMITTED),
        cross: bool = cast(bool, OMITTED),
        n_layers: int = cast(int, OMITTED),
        n_heads: int = cast(int, OMITTED),
        ffnx: int = cast(int, OMITTED),
        reduction: ReductionConfig | None = cast(ReductionConfig | None, OMITTED),
        dropout: Rate | None = cast(Rate | None, OMITTED),
        hide: HideInput = (),
        ignore: str | tuple[str, ...] = (),
        ablate: float = 0.0,
        predict: bool = False,
        task: TaskInput = None,
        type: Literal["branch"] = "branch",
        **children: Branch | Leaf | Twig | builtins.type[Leaf] | builtins.type[Branch] | builtins.type[Twig],
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
            cross=cross,
            n_layers=n_layers,
            n_heads=n_heads,
            ffnx=ffnx,
            reduction=reduction,
            dropout=dropout,
            hide=hide,
            ignore=ignore,
            ablate=ablate,
            predict=predict,
            task=task,
            type=type,
        )
        # Pydantic owns ordinary defaults; omitted options remain available for
        # a model preset to fill without replacing explicit defaults or None.
        super().__init__(**{name: value for name, value in data.items() if value is not OMITTED})

    @pydantic.field_validator("fields", mode="before")
    @classmethod
    def check_fields(cls, value: Any, info: pydantic.ValidationInfo) -> Any:
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
                fields.append(Branch.model_validate(payload, context=info.context))
                continue
            if isinstance(field_type, str) and field_type in TWIGS:
                fields.append(TWIGS[field_type].model_validate(payload))
                continue
            if not isinstance(field_type, str) or field_type not in TENSORFIELDS:
                raise ValueError(f"unknown tensor field type: {field_type}")

            fields.append(TENSORFIELDS[field_type].Request.model_validate(payload, context=info.context))

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

        self.validate_exemptions()
        return self

    def validate_exemptions(self) -> None:
        names = {field.name for field in self.fields}
        exemptions = set(self.ignore)
        for task in self.declared_tasks:
            if (rule := selection(task)) is not None:
                exemptions.update(rule.ignore)
        missing = sorted(exemptions - names)
        if missing:
            raise ValueError(
                f"branch '{self.name or '/'}' ignore names unknown direct children {missing}; choose from {sorted(str(name) for name in names)}"
            )

    def post_bind_validate(self) -> None:
        bounds(self)
        self.validate_exemptions()
        if self.ablate and self.address == "/":
            raise ValueError("ablate applies to a leaf or child branch; the model root is always present")
        if self.predict and self.address == "/":
            raise ValueError("predict=True requires a leaf or child branch; select the targets below the model root")
        if not selections(self):
            return None

        active_leaves = [
            descendant
            for descendant in getattr(self, "descendants", ())
            if isinstance(descendant, Leaf) and getattr(descendant, "active", True)
        ]
        if not active_leaves:
            raise ValueError(f"branch '{self.address}' requires active descendant leaves for hide, predict, or task")

        return None
