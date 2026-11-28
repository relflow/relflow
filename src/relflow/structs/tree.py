"""Schema tree node primitives used by models and tensorfields."""

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, TypeAlias, TypedDict, cast

import pydantic
from anytree import NodeMixin
from pydantic_core import CoreSchema

from relflow.structs.enums import Overflow
from relflow.structs.tasks import SUPERVISED, Probability, Selection, Task, TaskInput, selection

if TYPE_CHECKING:
    from relflow.structs.structure import Branch

Rate: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, lt=1.0)]


HideInput: TypeAlias = float | str | tuple[float | str, ...]


class FieldOptions(TypedDict, total=False):
    """Shared leaf options for input visibility, supervised prediction, and explicit tasks."""

    description: str | None
    embed: bool
    n_heads: int
    ffnx: int
    dropout: float | None
    active: bool
    query: str | tuple[str, ...] | None
    nullable: bool
    pooling: Literal["query", "mean"]
    decoder_position: bool | None
    weight: float
    hide: HideInput
    predict: bool
    ablate: float
    task: TaskInput
    n_linear: int


class Address(str):
    """Absolute schema path from child names or a serialized path.

    ``Address("items", "sku")`` is ``/items/sku``; ``Address()`` is the model
    root ``/``. An explicit empty string denotes an unbound or metadata node.
    """

    def __new__(cls, *parts: str) -> "Address":
        if not all(isinstance(part, str) for part in parts):
            raise TypeError("Address parts must be strings")
        if len(parts) == 0:
            value = "/"
        elif len(parts) == 1:
            value = parts[0]
        else:
            value = parts[0].rstrip("/") + "/" + "/".join(parts[1:])

        if value and not value.startswith("/"):
            value = "/" + value

        return str.__new__(cls, value)

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type: Any, handler: Any):
        from pydantic_core import core_schema

        return core_schema.no_info_after_validator_function(cls, core_schema.str_schema())


class Node(NodeMixin, pydantic.BaseModel):
    """Base schema tree node shared by branches, tensorfield requests, and twigs."""

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
    ffnx: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] = 8
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
        if self.parent is None or self.root.type != "schema":
            return Address("")
        return Address(*(cast(str, node.name) for node in self.path[2:]))

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


def check_policy_options(values: Mapping[str, Any], *, branch: bool) -> None:
    """Validate SDK argument forms before Pydantic restores persisted mappings and arrays."""
    if isinstance(values.get("hide"), list):
        raise TypeError("hide expects a float, a Boolean query string, or a tuple of those selectors")
    if branch and isinstance(values.get("ignore"), list):
        raise TypeError("ignore expects a direct child name or a tuple of names")
    task = values.get("task")
    if task is not None and not isinstance(task, Task):
        if not isinstance(task, tuple) or not all(isinstance(item, Task) for item in task):
            raise TypeError("task expects a task such as rf.reconstruct(...), a tuple of tasks, or None")


class Field(Node):
    """A schema node with input visibility and learning task declarations."""

    hide: HideInput = ()
    predict: pydantic.StrictBool = False
    ablate: Probability = 0.0
    task: TaskInput = None

    def __init__(self, **data: Any) -> None:
        check_policy_options(data, branch=not isinstance(self, Leaf))
        super().__init__(**data)

    @pydantic.field_validator("hide", mode="before")
    @classmethod
    def check_hide(cls, value: Any) -> HideInput:
        from relflow.data.query import compile

        sequence = isinstance(value, (tuple, list))
        selectors = value if sequence else (value,)
        for selector in selectors:
            if isinstance(selector, bool):
                raise TypeError(
                    "hide expects a probability or Boolean query path. Use predict=True on a supervised leaf or child branch, or omit hide to disable input masking."
                )
            if isinstance(selector, str):
                compile(selector)
            elif not isinstance(selector, float):
                raise TypeError(f"hide expects a float probability or a Boolean query path, got {selector!r}")
            elif not 0.0 <= selector < 1.0:
                raise ValueError(f"hide probability must be in [0, 1), got {selector!r}")
        return tuple(selectors) if sequence else value

    @property
    def declared_tasks(self) -> tuple[Task, ...]:
        explicit = () if self.task is None else self.task if isinstance(self.task, tuple) else (self.task,)
        return (SUPERVISED, *explicit) if self.predict else explicit


def selections(node: Field) -> tuple[Selection, ...]:
    """Lower a node's declarations to the field actions needed by the data pipeline."""
    from relflow.structs.structure import Branch

    selectors = node.hide if isinstance(node.hide, tuple) else (node.hide,)
    ignore = node.ignore if isinstance(node, Branch) else ()
    rules = [
        Selection("hide", query=value, ignore=ignore)
        if isinstance(value, str)
        else Selection("hide", rate=value, ignore=ignore)
        for value in dict.fromkeys(selectors)
        if value != 0.0
    ]
    rules.extend(rule for task in node.declared_tasks if (rule := selection(task)) is not None)
    return tuple(rules)


class Leaf(Field):
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
    query: tuple[str, ...] | None = None
    nullable: bool = True
    pooling: Literal["query", "mean"] = "query"
    decoder_position: pydantic.StrictBool | None = None
    weight: Annotated[float, pydantic.Field(gt=0.0, default=1.0)] = 1.0
    n_linear: Annotated[int, pydantic.Field(gt=0, default=1)] = 1

    def __init__(self, **data: Any) -> None:
        if "name" in data:
            raise TypeError("tensorfield names come from the parent; use field_name=Tensorfield(...)")
        super().__init__(**data)

    @pydantic.model_validator(mode="after")
    def check_task(self) -> Self:
        for task in self.declared_tasks:
            task.validate_placement(site="leaf", address=self.address or None)
        return self

    @pydantic.field_validator("query", mode="before")
    @classmethod
    def check_queries(cls, value: Any) -> Any:
        return (value,) if isinstance(value, str) else value

    @property
    def embedded(self) -> bool:
        """Whether any source row can contribute this leaf to encoder context."""
        if not self.active:
            return False
        child_name = None
        for node in reversed(self.path):
            if isinstance(node, Field):
                for task in node.declared_tasks:
                    rule = selection(task)
                    if (
                        rule
                        and rule.skips_input
                        and not rule.sampled
                        and rule.query is None
                        and child_name not in rule.ignore
                    ):
                        return False
                child_name = node.name
        return True

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

        from relflow.data.query import scope

        for expression in self.query:
            scope(expression)
        return self

    def queries(self) -> tuple[str, ...]:
        """Return explicit paths, or the same-named path after the leaf is bound."""
        from relflow.data.query import member

        return (member(cast(str, self.name)),) if self.query is None else self.query

    def post_bind_validate(self) -> None:
        """Check scoped queries and source counts against the live registry."""
        from relflow.data.query import origin
        from relflow.tensorfields.base import TENSORFIELDS

        count = len(self.queries())
        expected = TENSORFIELDS[self.type].queries
        if isinstance(expected, int):
            allowed = count == expected
        elif isinstance(expected, tuple):
            allowed = count in expected
        else:
            start = 0 if expected.start is None else expected.start
            step = 1 if expected.step is None else expected.step
            allowed = (
                count >= start and (expected.stop is None or count < expected.stop) and (count - start) % step == 0
            )
        if not allowed:
            raise ValueError(
                f"field '{self.address}' with extension '{self.type}' expects queries {expected!r}, got {count}"
            )
        for expression in self.queries():
            origin(expression, address=str(self.address))

    @functools.cached_property
    def shape(self) -> tuple[int | None, ...]:
        out: list[int | None] = []

        for node in self.path:
            if node.type == "branch":
                out.append(cast("Branch", node).length)

        return tuple(out)

    @functools.cached_property
    def overflows(self) -> tuple[Overflow | None, ...]:
        out: list[Overflow | None] = []

        for node in self.path:
            if node.type == "branch":
                out.append(cast("Branch", node).overflow)

        return tuple(out)
