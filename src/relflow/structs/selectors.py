"""Composable predicates for selecting schema nodes."""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from typing import Any, TypeAlias

import pydantic

from relflow.structs.structure import Branch
from relflow.structs.tree import Leaf, Node

SchemaField: TypeAlias = Branch | Leaf


class NodePredicate(pydantic.BaseModel):
    """Composable predicate used to select schema nodes."""

    model_config = pydantic.ConfigDict(frozen=True, arbitrary_types_allowed=True, extra="forbid")

    func: Callable[[Node], bool]

    @classmethod
    def from_selector(cls, value: "NodeSelector") -> "NodePredicate":
        if isinstance(value, cls):
            return value

        if isinstance(value, NodeAttribute):
            return cls(func=lambda node: has_model_attribute(node, value.name) and value.get(node) is True)

        if not callable(value):
            raise TypeError("node predicates must be where(...) expressions or callables")

        return cls(func=value)

    def __call__(self, node: Node) -> bool:
        return self.func(node)

    def __and__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> "NodePredicate":
        predicate = NodePredicate.from_selector(other)
        return NodePredicate(func=lambda node: self(node) and predicate(node))

    def __or__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> "NodePredicate":
        predicate = NodePredicate.from_selector(other)
        return NodePredicate(func=lambda node: self(node) or predicate(node))

    def __invert__(self) -> "NodePredicate":
        return NodePredicate(func=lambda node: not self(node))


def predicate(func: Callable[[Node], bool]) -> NodePredicate:
    """Wrap a callable for composition with schema predicates."""
    return NodePredicate(func=func)


_QUERYABLE_BUILTINS = frozenset(
    {
        "address",
        "parent",
        "children",
        "ancestors",
        "descendants",
        "reconstruct",
    }
)


class NodeAttribute(pydantic.BaseModel):
    """Queryable schema node attribute returned by `where(...)`."""

    model_config = pydantic.ConfigDict(frozen=True)

    name: str = pydantic.Field(
        description=(
            "Queryable node attribute. Built-ins include name, type, address, parent, "
            "children, ancestors, descendants, and reconstruct. Declared tensorfield "
            "options, including description and extension-specific fields, are also queryable."
        )
    )

    @classmethod
    def named(cls, name: str) -> "NodeAttribute":
        return cls(name=name)

    def get(self, node: Node, default: Any = None) -> Any:
        if self.name == "address":
            return str(node.address)
        if self.name == "parent":
            parent = getattr(node, "parent", None)
            return None if parent is None or not getattr(parent, "address", None) else str(parent.address)
        if self.name == "children":
            return tuple(str(child.address) for child in getattr(node, "children", ()))
        if self.name == "ancestors":
            return tuple(str(parent.address) for parent in getattr(node, "ancestors", ()) if parent.address)
        if self.name == "descendants":
            return tuple(str(child.address) for child in getattr(node, "descendants", ()))
        if self.name == "reconstruct":
            return (
                isinstance(node, Leaf)
                and node.active
                and any(policy.reconstruct for owner in node.path for policy in getattr(owner, "mask", ()))
            )

        return getattr(node, self.name, default)

    def exists(self) -> NodePredicate:
        return NodePredicate(func=lambda node: has_model_attribute(node, self.name))

    def __and__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> NodePredicate:
        return NodePredicate.from_selector(self) & other

    def __or__(self, other: "NodePredicate | NodeAttribute | Callable[[Node], bool]") -> NodePredicate:
        return NodePredicate.from_selector(self) | other

    def __invert__(self) -> NodePredicate:
        return NodePredicate(func=lambda node: not bool(self.get(node, False)))

    def __bool__(self) -> bool:
        raise TypeError("Use ~where(...) for negated predicates; Python 'not where(...)' cannot build a predicate")

    def is_in(self, values: Iterable[Any]) -> NodePredicate:
        candidates = tuple(values)
        return NodePredicate(func=lambda node: self.get(node) in candidates)

    def matches(self, pattern: str | re.Pattern[str]) -> NodePredicate:
        regex = re.compile(pattern) if isinstance(pattern, str) else pattern
        return NodePredicate(func=lambda node: regex.search(str(self.get(node, ""))) is not None)

    def contains(self, value: Any) -> NodePredicate:
        return NodePredicate(func=lambda node: value in (self.get(node) or ()))

    def is_null(self) -> NodePredicate:
        return NodePredicate(func=lambda node: self.get(node) is None)

    def is_not_null(self) -> NodePredicate:
        return NodePredicate(func=lambda node: self.get(node) is not None)

    def __eq__(self, other: Any) -> NodePredicate:  # type: ignore[override]
        return NodePredicate(func=lambda node: self.get(node) == other)

    def __ne__(self, other: Any) -> NodePredicate:  # type: ignore[override]
        return NodePredicate(func=lambda node: self.get(node) != other)


def where(name: str) -> NodeAttribute:
    """Start a schema predicate against a node attribute."""
    return NodeAttribute.named(name)


NodeSelector: TypeAlias = NodePredicate | NodeAttribute | Callable[[Node], bool]


def has_model_attribute(node: Node, name: str) -> bool:
    if name in _QUERYABLE_BUILTINS:
        return True

    fields = getattr(type(node), "model_fields", {})
    return name in fields or hasattr(type(node), name)
