"""Serializable schema, node predicates, and mutation helpers."""

from __future__ import annotations

import functools
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias, cast

import pydantic
from anytree import LevelOrderGroupIter, PreOrderIter
from rich.console import Console, ConsoleOptions
from rich.text import Text

from relflow.structs.enums import AttentionInput, AttentionMode, Component, Overflow, Strata
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.selectors import (
    NodeAttribute,
    NodePredicate,
    NodeSelector,
    SchemaField,
    SelectionCacheEntry,
    SelectionKey,
    has_model_attribute,
    predicate,
    where,
)
from relflow.structs.structure import Branch, Mask, RequestTypes
from relflow.structs.tree import Address, Leaf, MaskInput, Node, Rate, Selection
from relflow.tensorfields.base import TENSORFIELDS

__all__ = [
    "Schema",
    "NodeAttribute",
    "NodePredicate",
    "NodeSelector",
    "SchemaField",
    "SelectionCacheEntry",
    "SelectionKey",
    "TreeFieldInput",
    "predicate",
    "where",
]

TreeFieldInput: TypeAlias = SchemaField | type[Leaf]

_MISSING = object()


def bind_tree_field(name: str, value: TreeFieldInput) -> SchemaField:
    """Bind a fresh subtree without changing the reusable definition or its parent links."""
    if isinstance(value, type) and issubclass(value, Leaf):
        value = value()
    if not isinstance(value, (Branch, Leaf)):
        raise TypeError(f"tree field {name!r} must be a Branch, Leaf, or Leaf class; got {type(value).__name__}")

    excluded = {"name", "mask"}
    if isinstance(value, Branch):
        excluded.add("fields")
    payload = value.model_dump(mode="python", round_trip=True, exclude=excluded)
    payload.update(name=name, mask=value.mask)
    if isinstance(value, Branch):
        payload["fields"] = [bind_tree_field(cast(str, child.name), child) for child in value.fields]
    constructor = TENSORFIELDS[value.type].Request if type(value) is Leaf else type(value)
    return constructor.model_validate(payload)


def bind_fields(
    fields: Mapping[str, TreeFieldInput] | None,
    children: Mapping[str, TreeFieldInput],
) -> list[SchemaField]:
    """Merge parent-owned names and reject duplicates before binding any children."""
    if fields is None:
        fields = {}
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping from child names to definitions")
    duplicates = fields.keys() & children.keys()
    if duplicates:
        raise ValueError(f"duplicate field name(s): {sorted(duplicates)}")
    return [bind_tree_field(name, value) for name, value in (*fields.items(), *children.items())]


class Schema(Node):
    """Serializable schema and training metadata used to build a `Model`."""

    model_config = pydantic.ConfigDict(extra="forbid")

    # The metadata root has fixed values excluded from serialized field data.
    name: Literal["schema"] = pydantic.Field(default="schema", exclude=True)  # pyrefly: ignore [bad-override-mutable-attribute]
    type: Literal["schema"] = pydantic.Field(default="schema", exclude=True)
    description: Literal[None] = pydantic.Field(default=None, exclude=True)  # pyrefly: ignore [bad-override-mutable-attribute]
    d_model: Annotated[int, pydantic.Field(gt=0, default=128)]
    fields: Branch

    embed: ClassVar[None] = None  # pyrefly: ignore [bad-override]
    dropout: ClassVar[None] = None  # pyrefly: ignore [bad-override]

    _selection_cache: dict[SelectionKey, SelectionCacheEntry] = pydantic.PrivateAttr(default_factory=dict)

    @pydantic.model_validator(mode="before")
    @classmethod
    def check_masks(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        values = dict(data)

        def restore_policy(value: Mapping[str, Any]) -> Mask:
            return Mask.model_validate(dict(value))

        def restore(node: Any) -> Any:
            if not isinstance(node, Mapping):
                return node
            payload = dict(node)
            if "mask" in payload:
                value = payload["mask"]
                if isinstance(value, Mapping):
                    payload["mask"] = restore_policy(value)
                elif isinstance(value, (list, tuple)):
                    payload["mask"] = tuple(
                        restore_policy(policy) if isinstance(policy, Mapping) else policy for policy in value
                    )
            if isinstance(payload.get("fields"), (list, tuple)):
                payload["fields"] = [restore(field) for field in payload["fields"]]
            return payload

        if "fields" in values:
            values["fields"] = restore(values["fields"])
        return values

    @classmethod
    def from_tree(
        cls,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        fields: Mapping[str, TreeFieldInput] | None = None,
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionInput = AttentionMode.mha,
        reduction: ReductionConfig | None = Attention(),
        dropout: Rate | None = None,
        mask: MaskInput = False,
        **field_kwargs: TreeFieldInput,
    ) -> Self:
        """Build schema from tree fields."""
        root_fields = bind_fields(fields, field_kwargs)
        if not root_fields:
            raise ValueError("from_tree requires at least one field")
        branch = Branch.model_validate(
            dict(
                query=query,
                description=description,
                embed=embed,
                attention=attention,
                n_layers=n_layers,
                n_heads=n_heads,
                reduction=reduction,
                length=1,
                overflow=Overflow.error,
                dropout=dropout,
                mask=mask,
                fields=root_fields,
            )
        )
        return cls(d_model=d_model, fields=branch)

    def model_post_init(self, __context):
        def materialize(branch: Branch) -> Branch:
            fields: list[Branch | RequestTypes] = []
            for field in list(branch.fields):
                field.parent = None

                if isinstance(field, Branch):
                    fields.append(materialize(field))
                elif type(field) is Leaf:
                    fields.append(bind_tree_field(cast(str, field.name), field))
                else:
                    fields.append(field)

            branch.fields = fields
            for field in branch.fields:
                field.parent = branch

            return branch

        self.fields = materialize(self.fields)
        self.fields.length = 1
        self.fields.overflow = Overflow.error
        self.fields.parent = self
        self.clear_tree_caches()
        self.post_bind_validate()

    @pydantic.field_serializer("fields", mode="wrap")
    def serialize_root(self, root: Branch, handler: pydantic.SerializerFunctionWrapHandler) -> dict[str, Any]:
        """Only child nodes have parent-assigned names in a persisted schema."""
        payload = handler(root)
        payload.pop("name", None)
        return payload

    @property
    def reconstruct(self) -> list[Address]:
        role = NodePredicate(
            func=lambda node: (
                isinstance(node, Leaf)
                and node.active
                and any(policy.reconstruct for owner in node.path for policy in getattr(owner, "mask", ()))
            ),
            key=("role", "reconstruct"),
        )
        return [Address(str(node.address)) for node in self.select(role)]

    @property
    def objectives(self) -> list[Address]:
        return self.reconstruct

    @property
    def decodes(self) -> list[Address]:
        role = NodePredicate(
            func=lambda node: (
                isinstance(node, Leaf)
                and node.active
                and any(
                    policy.reconstruct and policy.rate is None
                    for owner in node.path
                    for policy in getattr(owner, "mask", ())
                )
            ),
            key=("role", "decodes"),
        )
        return [Address(str(node.address)) for node in self.select(role)]

    @property
    def embed(self) -> list[Address]:  # noqa: F811
        role = NodePredicate(
            func=lambda node: getattr(node, "embed", False) is True and (not isinstance(node, Leaf) or node.active),
            key=("role", "embed"),
        )
        return [Address(str(node.address)) for node in self.select(role)]

    @functools.cached_property
    def branches(self) -> dict[Address, Branch]:
        return {node.address: node for node in self.descendants if isinstance(node, Branch)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if isinstance(node, Leaf)}

    @functools.cached_property
    def active_requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.requests.values() if node.active}

    @functools.cached_property
    def shapes(self) -> dict[Address, tuple[int, ...]]:
        return {request.address: request.shape for request in self.requests.values()}

    @functools.cached_property
    def branch_outputs(self) -> dict[Address, int]:
        """Static routed token width produced by every structurally active branch."""

        outputs: dict[Address, int] = {}
        for depth in reversed(self.depthwise):
            for address in depth:
                branch = self.branches[address]
                child_width = 0
                for child in branch.fields:
                    if isinstance(child, Leaf):
                        child_width += int(child.active)
                    elif child.address in outputs:
                        child_width += outputs[child.address]

                input_width = branch.length * child_width
                if input_width == 0:
                    outputs[address] = 0
                elif branch.reduction is None:
                    outputs[address] = input_width
                elif isinstance(branch.reduction, Attention):
                    outputs[address] = branch.reduction.n_outputs
                else:
                    outputs[address] = 1

        return outputs

    def overflows(self, address: Address) -> tuple[Overflow, ...]:
        return (Overflow.error, *self.requests[Address(str(address))].overflows)

    def masks_for(self, address: Address) -> tuple[tuple[Address, Mask], ...]:
        request = self.requests[Address(str(address))]
        return tuple(
            (node.address, policy) for node in request.path if isinstance(node, (Branch, Leaf)) for policy in node.mask
        )

    def forward_for(self, strata: Strata | str) -> list[Address]:
        selected = self.decodes if Strata.normalize(strata) is Strata.predict else self.objectives
        addresses = {*selected, *self.embed}
        return [node.address for node in PreOrderIter(self.fields) if node.address in addresses]

    @functools.cached_property
    def depthwise(self) -> list[list[Address]]:
        out: list[list[Address]] = []
        for depth in LevelOrderGroupIter(self.fields):
            branches = [node.address for node in depth if isinstance(node, Branch)]
            if branches:
                out.append(branches)

        return out

    def clear_tree_caches(self) -> None:
        for name in ("branches", "requests", "active_requests", "shapes", "branch_outputs", "depthwise"):
            self.__dict__.pop(name, None)

        for node in PreOrderIter(self.fields):
            for name in ("address", "heritage", "shape", "overflows"):
                node.__dict__.pop(name, None)

    def clear_selection_cache(self) -> None:
        self._selection_cache.clear()

    def __getstate__(self) -> dict[str, Any]:
        """Serialize canonical schema state without derived predicate closures."""

        state = super().__getstate__()
        private = state.get("__pydantic_private__")
        if private is not None:
            state["__pydantic_private__"] = {**private, "_selection_cache": {}}
        return state

    def refresh_selection_cache(self) -> None:
        self._selection_cache = {
            key: entry.model_copy(
                update={
                    "nodes": tuple(
                        node
                        for node in PreOrderIter(self.fields)
                        if (entry.include_root or node is not self.fields)
                        if entry.predicate(node)
                    )
                }
            )
            for key, entry in self._selection_cache.items()
        }

    def post_bind_validate(self) -> None:
        if self.fields.name is not None:
            raise ValueError(
                f"model root must be anonymous, got name {self.fields.name!r}; "
                "rebuild the schema without a root name (named-root checkpoints are not supported)"
            )
        for branch in self.branches.values():
            branch.post_bind_validate()
            if branch.embed and self.branch_outputs[branch.address] == 0:
                raise ValueError(
                    f"branch '{branch.address}' has embed=True but no active descendant output; "
                    "activate at least one descendant leaf or disable branch embedding"
                )
            if isinstance(branch.reduction, Attention):
                n_heads = branch.reduction.n_heads or branch.n_heads
                if self.d_model % n_heads != 0 or self.d_model // n_heads < 2:
                    raise ValueError(
                        f"branch '{branch.address}' Attention reduction requires n_heads to divide "
                        "d_model with at least two dimensions per head"
                    )

        for request in self.requests.values():
            request.post_bind_validate()

        self.validate_capabilities()

    def validate_capabilities(self) -> None:
        """Validate effective leaf roles against registered extension components."""

        from relflow.tensorfields.base import TENSORFIELDS

        for address, request in self.active_requests.items():
            owners = [
                owner.address
                for owner in request.path
                if any(policy.reconstruct for policy in getattr(owner, "mask", ()))
            ]
            required: list[Component] = []
            if owners:
                required.extend((Component.Decoder, Component.loss))
            elif request.embed:
                required.append(Component.Decoder)

            extension = TENSORFIELDS[request.type]
            missing = [component.value for component in required if component not in extension.components]
            if not missing:
                continue

            names = ", ".join(missing)
            if owners:
                sources = ", ".join(repr(str(owner)) for owner in owners)
                raise ValueError(
                    f"reconstruction mask at {sources} reaches leaf '{address}', but extension "
                    f"'{extension.name}' is missing required component(s): {names}; register both Decoder and loss"
                )
            raise ValueError(
                f"embedded leaf '{address}' uses extension '{extension.name}', which is missing required "
                f"component: {names}; register a Decoder or set embed=False"
            )

    def select(
        self,
        *predicates: NodeSelector,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        if predicates:
            normalized = tuple(NodePredicate.from_selector(item) for item in predicates)
            combined = NodePredicate(
                func=lambda node: all(item(node) for item in normalized),
                key=("and", tuple(item.key for item in normalized)),
                cacheable=all(item.cacheable for item in normalized),
            )
        else:
            combined = NodePredicate(func=lambda node: True, key=("all",))

        key = ("select", include_root, combined.key)

        if use_cache and combined.cacheable and key in self._selection_cache:
            return Selection(self._selection_cache[key].nodes)

        nodes = tuple(
            node for node in PreOrderIter(self.fields) if (include_root or node is not self.fields) if combined(node)
        )

        if use_cache and combined.cacheable:
            self._selection_cache[key] = SelectionCacheEntry(
                key=key,
                predicate=combined,
                include_root=include_root,
                nodes=nodes,
            )

        return Selection(nodes)

    def update(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: Any,
    ) -> None:
        """Mutate matching schema nodes."""
        if not values:
            raise ValueError("update requires at least one field value")

        nodes = self.select(*predicates, include_root=include_root, use_cache=use_cache)
        if not strict and nodes:
            missing = [name for name in values if not any(has_model_attribute(node, name) for node in nodes)]
            if missing:
                raise AttributeError(
                    f"selected schema nodes have no attribute(s): {missing}; "
                    "declare options explicitly and use description for notes"
                )
        snapshots: list[tuple[Node, str, Any, bool]] = []
        try:
            for node in nodes:
                missing = [name for name in values if not has_model_attribute(node, name)]
                if missing and strict:
                    label = str(node.address) or node.name
                    raise AttributeError(f"{label} has no attribute(s): {missing}")

                applicable_values = {name: value for name, value in values.items() if has_model_attribute(node, name)}

                if validate and applicable_values:
                    payload = node.model_dump(mode="python", round_trip=True, exclude={"mask", *applicable_values})
                    payload["mask"] = cast(SchemaField, node).mask
                    payload.update(applicable_values)
                    validated = type(node).model_validate(payload)
                    applicable_values = {name: getattr(validated, name) for name in applicable_values}

                for name, value in applicable_values.items():
                    snapshots.append(
                        (
                            node,
                            name,
                            getattr(node, name, _MISSING),
                            name in getattr(node, "model_fields_set", set()),
                        )
                    )
                    setattr(node, name, value)
                    if name in getattr(type(node), "model_fields", {}):
                        node.model_fields_set.add(name)

            self.clear_tree_caches()
            self.post_bind_validate()
        except Exception:
            for node, name, original, was_set in reversed(snapshots):
                if original is _MISSING:
                    if hasattr(node, name):
                        delattr(node, name)
                else:
                    setattr(node, name, original)
                if name in getattr(type(node), "model_fields", {}):
                    if was_set:
                        node.model_fields_set.add(name)
                    else:
                        node.model_fields_set.discard(name)
            self.clear_tree_caches()
            self.refresh_selection_cache()
            raise

        self.refresh_selection_cache()

    def extend(
        self,
        *predicates: NodeSelector,
        fields: Mapping[str, TreeFieldInput] | None = None,
        include_root: bool = True,
        use_cache: bool = True,
        **children: TreeFieldInput,
    ) -> None:
        """Append parent-named fields under the single branch selected by predicates."""
        new_fields = bind_fields(fields, children)
        if not new_fields:
            raise ValueError("extend requires at least one schema field")
        candidates = [
            node
            for node in self.select(*predicates, include_root=include_root, use_cache=use_cache)
            if isinstance(node, Branch)
        ]
        if len(candidates) != 1:
            raise ValueError(f"extend requires exactly one matching branch node, found {len(candidates)}")
        parent = candidates[0]
        existing_names = {field.name for field in parent.fields}
        duplicate_names = sorted({cast(str, field.name) for field in new_fields if field.name in existing_names})
        if duplicate_names:
            raise ValueError(f"duplicate field name(s): {duplicate_names}")

        original_fields = list(parent.fields)
        try:
            parent.fields.extend(new_fields)
            for field in new_fields:
                field.parent = parent

            self.clear_tree_caches()
            self.post_bind_validate()
        except Exception:
            parent.fields = original_fields
            for field in new_fields:
                field.parent = None
            self.clear_tree_caches()
            self.post_bind_validate()
            self.refresh_selection_cache()
            raise

        self.refresh_selection_cache()

    def delete(
        self,
        *predicates: NodeSelector,
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        """Permanently remove selected schema nodes from the tree."""
        if not predicates:
            raise ValueError("delete requires at least one predicate")

        selected = self.select(*predicates, include_root=include_root, use_cache=use_cache)
        if not selected:
            raise ValueError("delete matched no nodes")
        if self.fields in selected:
            raise ValueError("delete cannot remove the root branch")

        selected_ids = {id(node) for node in selected}
        roots = [
            node
            for node in selected
            if not any(
                id(ancestor) in selected_ids for ancestor in getattr(node, "ancestors", ()) if ancestor is not self
            )
        ]
        removed_by_id = {id(node): node for node in roots}
        for node in roots:
            removed_by_id.update({id(descendant): descendant for descendant in getattr(node, "descendants", ())})
        removed_addresses = {node.address for node in removed_by_id.values()}

        remaining_request_addresses = {address for address in self.requests if address not in removed_addresses}
        if not remaining_request_addresses:
            raise ValueError("delete would remove every request")

        remaining_branch_addresses = {address for address in self.branches if address not in removed_addresses}
        for address in remaining_branch_addresses:
            if not any(
                descendant.address in remaining_request_addresses for descendant in self.branches[address].descendants
            ):
                raise ValueError(f"delete would leave branch '{address}' without request descendants")

        for node in roots:
            parent = node.parent
            if not isinstance(parent, Branch):
                raise ValueError(f"delete cannot remove '{node.address}' because it has no branch parent")
            parent.fields = [field for field in parent.fields if field is not node]
            node.parent = None

        self.clear_tree_caches()
        self.post_bind_validate()
        self.refresh_selection_cache()

    @contextmanager
    def override(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: Any,
    ) -> Generator[None, None, None]:
        nodes = self.select(*predicates, include_root=include_root, use_cache=use_cache)
        snapshot = [
            (
                node,
                name,
                getattr(node, name, _MISSING),
                name in getattr(node, "model_fields_set", set()),
            )
            for node in nodes
            for name in values
            if has_model_attribute(node, name)
        ]

        self.update(
            *predicates,
            strict=strict,
            include_root=include_root,
            validate=validate,
            use_cache=use_cache,
            **values,
        )

        try:
            yield
        finally:
            for node, name, original, was_set in snapshot:
                if original is _MISSING:
                    if getattr(node, name, _MISSING) is _MISSING:
                        continue
                    delattr(node, name)
                else:
                    setattr(node, name, original)
                    if name in getattr(type(node), "model_fields", {}):
                        if was_set:
                            node.model_fields_set.add(name)
                        else:
                            node.model_fields_set.discard(name)

            self.clear_tree_caches()
            self.post_bind_validate()
            self.refresh_selection_cache()

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Generator[Text, None, None]:
        heading = Text()
        heading.append(self.name, style=self.RICH_NAME_STYLE)
        heading.append(" ")
        heading.append(f"[{self.type}]", style=self.RICH_TYPE_STYLE)
        for name, value in (
            ("d_model", self.d_model),
            ("branches", len(self.branches)),
            ("fields", len(self.active_requests)),
            ("reconstruct", len(self.reconstruct)),
            ("embeds", len(self.embed)),
        ):
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        yield heading

        lines = list(self.fields.__rich_console__(console, options))
        if not lines:
            return
        first = Text()
        first.append("`-- ", style=self.RICH_TREE_STYLE)
        if isinstance(lines[0], Text):
            first.append_text(lines[0])
        else:
            first.append(str(lines[0]))
        yield first
        for line in lines[1:]:
            nested = Text()
            nested.append("    ", style=self.RICH_TREE_STYLE)
            if isinstance(line, Text):
                nested.append_text(line)
            else:
                nested.append(str(line))
            yield nested
