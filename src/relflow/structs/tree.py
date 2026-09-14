"""Schema tree node primitives used by models and tensorfields."""

import functools
import io
from abc import ABC
from collections.abc import Generator
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, TypeAlias, TypedDict, cast

import pydantic
from anytree import NodeMixin
from pydantic_core import CoreSchema
from rich.console import Console, ConsoleOptions
from rich.text import Text

from relflow.structs.enums import Overflow

if TYPE_CHECKING:
    from relflow.structs.structure import Branch

Rate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]


class Mask(pydantic.BaseModel):
    """Select coordinates to hide, drop, or reconstruct.

    ``query`` reads an explicit Boolean selection from the input. ``rate``
    samples coordinates during training. ``skip=True`` prevents embedding;
    ``reconstruct=True`` makes selected coordinates supervised targets.
    ``dropout`` defaults to the opposite of ``reconstruct``. These last two
    effects cannot both be enabled.
    """

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")

    query: str | None = None
    rate: Annotated[float, pydantic.Field(ge=0.0, le=1.0)] | None = None
    skip: bool = False
    dropout: bool | None = None
    reconstruct: bool = False

    @pydantic.field_validator("rate", mode="before")
    @classmethod
    def check_rate(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise TypeError("Mask.rate must be a number, not a boolean")
        return value

    @pydantic.model_validator(mode="after")
    def check_policy(self) -> Self:
        dropout = not self.reconstruct if self.dropout is None else self.dropout
        if dropout and self.reconstruct:
            raise ValueError("Mask.dropout=True cannot be combined with reconstruct=True")

        if self.query is not None:
            from relflow.data.query import compile

            compile(self.query)

        object.__setattr__(self, "dropout", dropout)
        return self

    @classmethod
    def normalize(cls, value: Any) -> tuple["Mask", ...]:
        """Normalize one public node value to immutable canonical policies."""
        if value is False:
            return ()
        if value is True:
            return (cls(skip=True, dropout=False, reconstruct=True),)
        if value is None:
            raise TypeError("mask cannot be None; use False or an empty list or tuple")
        if isinstance(value, cls):
            policies = (value,)
        elif isinstance(value, float):
            policies = (cls(rate=value),)
        elif isinstance(value, (list, tuple)):
            if not all(isinstance(policy, cls) for policy in value):
                raise TypeError("mask list and tuple entries must be Mask objects")
            policies = tuple(value)
        else:
            raise TypeError("mask must be False, True, a float, a Mask, or a list or tuple of Mask objects")

        normalized: list[Mask] = []
        seen: set[Mask] = set()
        for policy in policies:
            if policy.rate == 0.0:
                continue
            if policy.rate == 1.0:
                policy = policy.model_copy(update={"rate": None})
            if policy in seen:
                continue
            seen.add(policy)
            normalized.append(policy)
        return tuple(normalized)


MaskInput: TypeAlias = Mask | float | bool | list[Mask] | tuple[Mask, ...]


class FieldOptions(TypedDict, total=False):
    """Shared constructor inputs; requests normalize ``mask`` to a tuple.

    ``mask=True`` declares a supervised target that is never embedded. A float
    adds training-only random masking. Use explicit ``Mask`` objects to select
    coordinates or configure reconstruction. ``query`` is always explicit;
    names alone never infer a structural query.
    """

    description: str | None
    embed: bool
    n_heads: int
    dropout: float | None
    active: bool
    query: str | None
    nullable: bool
    pooling: Literal["query", "mean"]
    decoder_position: bool | None
    weight: float
    mask: MaskInput
    n_linear: int


class Renderable(ABC):
    """Base class for objects rendered consistently through Rich."""

    RICH_NAME_STYLE: ClassVar[str] = "bold white on #1f2937"
    RICH_TYPE_STYLE: ClassVar[str] = "bold yellow on #3f3f46"
    RICH_TREE_STYLE: ClassVar[str] = "bold dim"

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Generator[Text, None, None]:
        yield Text(repr(self))

    def __rich_repr__(self) -> Generator[str, None, None]:
        yield str(self)

    def __str__(self) -> str:
        console = Console(file=io.StringIO(), record=True, width=120, force_jupyter=False)
        console.print(self)
        return console.export_text(clear=False).rstrip("\n")

    def _repr_mimebundle_(self, include: list[str] | None = None, exclude: list[str] | None = None) -> dict[str, str]:
        return {
            "text/plain": str(self),
            "text/html": self._repr_html_(),
        }

    def _repr_html_(self) -> str:
        console = Console(file=io.StringIO(), record=True, width=120, force_jupyter=False)
        console.print(self)
        return console.export_html(
            inline_styles=True,
            clear=False,
            code_format=(
                '<pre style="font-family: Menlo, Consolas, monospace; '
                "white-space: pre-wrap; margin: 0; padding: 0; border: 0; "
                'background: transparent;"><code>{code}</code></pre>'
            ),
        )

    def _mime_(self) -> tuple[str, str]:
        return "text/html", self._repr_html_()


class Selection(list, Renderable):
    """List-like selection result with readable Rich and pprint output."""

    # Rich supports opting out of its optional representation protocol.
    __rich_repr__: ClassVar[None] = None  # pyrefly: ignore [bad-override]

    def __repr__(self) -> str:
        return str(self)

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Generator[Text, None, None]:
        if not self:
            yield Text("[]")
            return

        yield Text("[", style="dim")
        for index, node in enumerate(self):
            lines = list(node.__rich_console__(console, options))
            for line_index, line in enumerate(lines):
                text = Text("  ")
                if isinstance(line, Text):
                    text.append_text(line)
                else:
                    text.append(str(line))
                if index < len(self) - 1 and line_index == len(lines) - 1:
                    text.append(",")
                yield text
        yield Text("]", style="dim")


class Address(str):
    """Slash-delimited stable path to a schema node."""

    def __new__(cls, *parts: str) -> "Address":
        if len(parts) == 0:
            value = ""
        elif len(parts) == 1:
            value = parts[0]
        else:
            value = "/".join(parts)

        if not isinstance(value, str):
            raise TypeError("Address parts must be strings")

        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


class Node(NodeMixin, Renderable, pydantic.BaseModel):
    """Base schema tree node shared by branches and tensorfield requests."""

    model_config = pydantic.ConfigDict(extra="forbid")

    if TYPE_CHECKING:

        @property
        def name(self) -> str | None:
            """The name assigned when a parent binds this node."""
            ...

        @name.setter
        def name(self, value: str | None) -> None: ...

        @property
        def type(self) -> str:
            """The concrete node's fixed schema discriminator."""
            ...
    else:
        name: str | None = None
        type: str
    description: str | None = None
    embed: bool = False
    n_heads: Annotated[int, pydantic.Field(gt=0, default=4)] = 4
    dropout: Rate | None = None

    @classmethod
    def __get_pydantic_core_schema__(cls, source: Any, handler: pydantic.GetCoreSchemaHandler) -> CoreSchema:
        """Restore serialized nodes through validators without replaying public constructors."""
        schema = handler(source)
        model = schema
        while model["type"] in ("function-before", "function-after", "function-wrap"):
            model = model["schema"]
        if model["type"] == "model":
            model["custom_init"] = False
        return schema

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Keep parent-assigned names out of Pydantic's generated constructor signature."""
        super().__pydantic_init_subclass__(**kwargs)
        cls.__signature__ = cls.__signature__.replace(
            parameters=[parameter for parameter in cls.__signature__.parameters.values() if parameter.name != "name"]
        )

    @functools.cached_property
    def address(self) -> Address:
        return Address(*(cast(str, node.name) for node in self.path[1:]))

    @functools.cached_property
    def heritage(self) -> list[Address]:
        return [node.address for node in self.path[1:]]

    @pydantic.model_validator(mode="after")
    def check_node_name(self) -> Self:
        if self.name is None:
            return self

        if not isinstance(self.name, str) or not self.name:
            raise ValueError("name must be a non-empty string")

        if not all(c.isalnum() or c in "_-" for c in self.name):
            raise ValueError("name may contain only letters, digits, '_' or '-'")

        return self

    @pydantic.model_validator(mode="after")
    def check_n_heads_is_even(self) -> Self:
        if not isinstance(self.n_heads, int):
            raise ValueError("n_heads must be an integer")

        if self.n_heads % 2 != 0:
            raise ValueError("n_heads must be even")

        return self

    @pydantic.field_validator("description", mode="before")
    @classmethod
    def check_description(cls, value: str | None) -> str | None:
        if value is None:
            return None

        if not isinstance(value, str):
            raise ValueError("description must be a string when provided")

        normalized = value.strip()
        return normalized or None

    def post_bind_validate(self) -> None:
        return None


class Leaf(Node):
    """Base tensorfield request node.

    Concrete tensorfield constructors such as `Number` and `Category` inherit
    from this class through their registered request models.
    Options must be declared fields; use ``description`` for notes.
    """

    model_config = pydantic.ConfigDict(validate_default=True)

    active: bool = True
    embed: bool = False
    if TYPE_CHECKING:

        @property
        def type(self) -> str: ...
    else:
        type: str
    query: str | None = None
    nullable: bool = True
    pooling: Literal["query", "mean"] = "query"
    decoder_position: pydantic.StrictBool | None = None
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)] = 1.0
    # The before-validator normalizes this public Boolean default to a tuple.
    mask: tuple[Mask, ...] = cast(tuple[Mask, ...], pydantic.Field(default=False))
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1

    def __init__(self, **data: Any) -> None:
        if "name" in data:
            raise TypeError("tensorfield names come from the parent; use field_name=Tensorfield(...)")
        super().__init__(**data)

    @pydantic.field_validator("mask", mode="before")
    @classmethod
    def check_mask(cls, value: Any) -> tuple[Mask, ...]:
        return Mask.normalize(value)

    @pydantic.field_validator("type")
    @classmethod
    def check_type(cls, value: str) -> str:
        from relflow.tensorfields import extensions as _extensions  # noqa: F401
        from relflow.tensorfields.base import TENSORFIELDS

        if value not in TENSORFIELDS:
            raise ValueError(f"unknown tensor field type: {value}")

        return value

    @pydantic.model_validator(mode="after")
    def check_query(self) -> Self:
        if self.query is None:
            return self

        from relflow.data.query import compile

        compile(self.query)
        return self

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Generator[Text, None, None]:
        flags = ["active" if self.active else "inactive"]
        if self.embed:
            flags.append("embed")
        if any(policy.reconstruct for owner in self.path for policy in getattr(owner, "mask", ())):
            flags.append("reconstruct")

        heading = Text()
        if self.name is not None:
            heading.append(self.name, style=self.RICH_NAME_STYLE)
            heading.append(" ")
        heading.append(f"[{self.type}]", style=self.RICH_TYPE_STYLE)
        for flag in flags:
            heading.append(" ")
            heading.append(
                flag,
                style={
                    "active": "bold #64748b",
                    "inactive": "bold #7f1d1d",
                    "embed": "bold #065f46",
                    "reconstruct": "bold #713f12",
                }.get(flag, "bold"),
            )
        if self.query is not None:
            heading.append(" ")
            heading.append("query=", style="dim")
            heading.append(self.query, style="cyan")
        yield heading

        common_names = (
            "pooling",
            "decoder_position",
            "weight",
            "n_heads",
            "n_linear",
            "dropout",
        )
        common = Text()
        first = True
        for name in common_names:
            value = getattr(self, name, None)
            if value is None:
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif hasattr(value, "value"):
                value = value.value
            if not first:
                common.append(" ")
            common.append(f"{name}=", style="dim")
            common.append(str(value), style="cyan")
            first = False
        if common.plain:
            line = Text(" ")
            line.append_text(common)
            yield line

        excluded = {"name", "type", "description", "active", "embed", "query", "nullable", "mask", *common_names}
        specific = Text()
        first = True
        for name, field in type(self).model_fields.items():
            if name in excluded:
                continue
            value = getattr(self, name, None)
            if value is None:
                continue
            if isinstance(value, float) and value.is_integer():
                value = int(value)
            elif hasattr(value, "value"):
                value = value.value
            if not first:
                specific.append(" ")
            label = field.serialization_alias or field.alias or name
            specific.append(f"{label}=", style="dim")
            specific.append(str(value), style="cyan")
            first = False
        if specific.plain:
            line = Text(" ")
            line.append_text(specific)
            yield line

    @functools.cached_property
    def shape(self) -> tuple[int, ...]:
        out: list[int] = []

        for node in self.path:
            if node.type == "branch":
                out.append(cast("Branch", node).length)

        return tuple(out)

    @functools.cached_property
    def overflows(self) -> tuple[Overflow, ...]:
        out: list[Overflow] = []

        for node in self.path:
            if node.type == "branch":
                out.append(cast("Branch", node).overflow)

        return tuple(out)
