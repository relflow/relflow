"""Structured schema nodes that group tensorfield requests."""

from __future__ import annotations

import builtins
from collections.abc import Generator, Mapping
from typing import Annotated, Any, Literal, Self, TypeAlias, cast

import pydantic
from rich.console import Console, ConsoleOptions
from rich.text import Text

from relflow.structs.enums import AttentionInput, AttentionMode, Overflow, OverflowInput
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.tree import Leaf, Mask, MaskInput, Node, Rate
from relflow.tensorfields import extensions as _extensions  # noqa: F401
from relflow.tensorfields.base import TENSORFIELDS

RequestTypes: TypeAlias = Leaf


class Branch(Node):
    """Repeated nested object group in a `relflow` schema.

    Parent keywords supply child names, such as ``amount=rf.Number``. Use a
    ``fields`` mapping for names that collide with configuration options. ``length``
    limits the repeated collection; ``overflow`` chooses which excess rows to
    retain. ``reduction`` controls the context passed to the parent branch.
    ``mask`` accepts the same policies as a leaf and applies to its descendants.
    """

    model_config = pydantic.ConfigDict(extra="forbid", validate_default=True)

    type: Annotated[Literal["branch"], pydantic.Field(default="branch")] = "branch"
    query: str | None = None
    attention: AttentionMode = AttentionMode.mha
    length: Annotated[int, pydantic.Field(gt=0, default=1)] = 1
    overflow: Overflow = Overflow.head
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
        length: int = 1,
        overflow: OverflowInput = Overflow.head,
        attention: AttentionInput = AttentionMode.mha,
        n_layers: int = 1,
        n_heads: int = 4,
        reduction: ReductionConfig | None = Attention(),
        dropout: Rate | None = None,
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
            reduction=reduction,
            dropout=dropout,
            mask=mask,
            type=type,
        )
        super().__init__(**data)

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

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Generator[Text, None, None]:
        is_root = getattr(getattr(self, "parent", None), "type", None) == "schema"
        display_type = "root" if is_root else self.type
        attributes = ("query", "attention", "n_layers", "n_heads", "reduction", "dropout")
        if not is_root:
            attributes = ("length", "overflow", *attributes)

        heading = Text()
        if self.name is not None:
            heading.append(self.name, style=self.RICH_NAME_STYLE)
            heading.append(" ")
        heading.append(f"[{display_type}]", style=self.RICH_TYPE_STYLE)
        if self.embed:
            heading.append(" ")
            heading.append("embed", style="bold #065f46")
        for name in attributes:
            value = getattr(self, name, None)
            if value is None and name != "reduction":
                continue
            if name == "reduction":
                if value is None:
                    value = "none"
                elif isinstance(value, Attention):
                    settings = value.model_dump(exclude={"type"}, exclude_none=True)
                    if settings.get("position") is True:
                        settings.pop("position")
                    parameters = ", ".join(f"{key}={setting}" for key, setting in settings.items())
                    value = f"{value.type}({parameters})"
                else:
                    value = value.type
            elif isinstance(value, float) and value.is_integer():
                value = int(value)
            elif hasattr(value, "value"):
                value = value.value
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        if self.mask:
            heading.append(" ")
            heading.append("mask=", style="dim")
            heading.append(str(len(self.mask)), style="cyan")

        yield heading

        for index, child in enumerate(self.fields):
            connector = "`-- " if index == len(self.fields) - 1 else "|-- "
            continuation = "    " if index == len(self.fields) - 1 else "|   "
            lines = list(child.__rich_console__(console, options))
            if not lines:
                continue
            first = Text()
            first.append(connector, style=self.RICH_TREE_STYLE)
            if isinstance(lines[0], Text):
                first.append_text(lines[0])
            else:
                first.append(str(lines[0]))
            yield first
            for line in lines[1:]:
                nested = Text()
                nested.append(continuation, style=self.RICH_TREE_STYLE)
                if isinstance(line, Text):
                    nested.append_text(line)
                else:
                    nested.append(str(line))
                yield nested
