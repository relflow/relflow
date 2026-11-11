"""Collect class-body children before framework configuration is interpreted."""

from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

from relflow.structs.tree import Leaf, Node

if TYPE_CHECKING:
    from relflow.structs.experiment import TreeFieldInput

__all__ = ["Declarations", "fields"]


class Declarations(type):
    """Consume child declarations so they cannot shadow enclosing-node APIs."""

    def __new__(mcls, name: str, bases: tuple[type, ...], namespace: dict[str, Any], **kwargs: Any) -> type:
        local: dict[str, Any] = {}
        for key, value in tuple(namespace.items()):
            if isinstance(value, type) and issubclass(value, Node):
                if not issubclass(value, Leaf):
                    raise TypeError(f"{name}.{key} must be an instantiated branch; use {value.__name__}(...)")
            elif not isinstance(value, Node):
                continue
            if key.startswith("__"):
                raise TypeError(f"{name}.{key} cannot declare a child using a Python protocol name")
            local[key] = namespace.pop(key)

        inherited: dict[str, Any] = {}
        for base in bases:
            for key, value in getattr(base, "__relflow_fields__", {}).items():
                if key in inherited and inherited[key] is not value and key not in local:
                    raise TypeError(f"{name}.{key} has conflicting inherited children; redeclare the child explicitly")
                if key in namespace:
                    raise TypeError(f"{name}.{key} must replace an inherited child with a tensorfield or branch")
                inherited.setdefault(key, value)

        annotations = dict(namespace.get("__annotations__", {}))
        for key in local:
            annotations.pop(key, None)
        namespace["__annotations__"] = annotations
        namespace["__relflow_fields__"] = MappingProxyType(inherited | local)
        return super().__new__(mcls, name, bases, namespace, **kwargs)


def fields(
    cls: type,
    definitions: Mapping[str, "TreeFieldInput"] | None,
    children: Mapping[str, "TreeFieldInput"],
) -> Mapping[str, "TreeFieldInput"] | None:
    """Keep constructors of declarative classes exclusive to node configuration."""
    declared = getattr(cls, "__relflow_fields__", {})
    if not declared:
        return definitions
    if definitions is not None or children:
        supplied = sorted(set(children) | ({"fields"} if definitions is not None else set()))
        raise TypeError(
            f"{cls.__name__} declares its children in the class body; constructor arguments configure the node. "
            f"Remove child definitions {supplied!r} and use a subclass to change children"
        )
    return declared
