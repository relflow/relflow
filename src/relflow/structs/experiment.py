"""Serializable schema, node predicates, and mutation helpers."""

from __future__ import annotations

import functools
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Annotated, Any, ClassVar, Literal, Self, TypeAlias, TypeVar, cast

import pydantic
from anytree import LevelOrderGroupIter, PreOrderIter

from relflow.structs.declaration import check_root_policies
from relflow.structs.enums import AttentionInput, AttentionMode, Component, Overflow, Strata
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.selectors import (
    NodeAttribute,
    NodePredicate,
    NodeSelector,
    SchemaField,
    has_model_attribute,
    predicate,
    where,
)
from relflow.structs.structure import Branch, RequestTypes
from relflow.structs.tasks import SUPERVISED, BoundTask, Reconstruction, Selection, TaskInput, selection, validate
from relflow.structs.tree import Address, HideInput, Leaf, Node, Rate, check_policy_options
from relflow.structs.twigs import Twig
from relflow.tensorfields.base import TENSORFIELDS

__all__ = [
    "Schema",
    "NodeAttribute",
    "NodePredicate",
    "NodeSelector",
    "SchemaField",
    "TreeFieldInput",
    "predicate",
    "where",
]

TreeFieldInput: TypeAlias = SchemaField | type[Leaf] | type[Branch] | type[Twig]
Defaults: TypeAlias = Callable[[TreeFieldInput], Mapping[str, Any]]

_MISSING = object()


@dataclass(frozen=True)
class Roles:
    """Fixed output roles derived once for the current schema tree."""

    objectives: tuple[Address, ...]
    decodes: tuple[Address, ...]
    embed: tuple[Address, ...]
    reconstruct: tuple[Address, ...]
    fixed_targets: frozenset[Address]

    @functools.cached_property
    def reconstruction_targets(self) -> frozenset[Address]:
        """Membership for reconstruction predicates composed with other selectors."""
        return frozenset(self.reconstruct)


def bind_tree_field(name: str, value: TreeFieldInput, *, defaults: Defaults | None = None) -> SchemaField:
    """Bind a fresh subtree without changing the reusable definition or its parent links."""
    if isinstance(value, type) and issubclass(value, Branch):
        value = value()
    if isinstance(value, type) and issubclass(value, (Leaf, Twig)):
        payload = {} if defaults is None else dict(defaults(value))
        bound = value.model_validate({**payload, "name": name})
        if defaults is not None:
            object.__setattr__(bound, "__pydantic_fields_set__", set(type(bound).model_fields))
        return bound
    if not isinstance(value, (Branch, Leaf, Twig)):
        raise TypeError(
            f"tree field {name!r} must be a Branch, Leaf, or Twig instance or class; got {type(value).__name__}"
        )

    excluded = {"name"}
    if isinstance(value, Branch):
        excluded.add("fields")
    payload = value.model_dump(mode="python", round_trip=True, exclude=excluded)
    payload["name"] = name
    if defaults is not None:
        payload.update(
            {option: default for option, default in defaults(value).items() if option not in value.model_fields_set}
        )
    if isinstance(value, Branch):
        payload["fields"] = [bind_tree_field(cast(str, child.name), child, defaults=defaults) for child in value.fields]
    if isinstance(value, Branch) and getattr(type(value), "__relflow_fields__", {}):
        constructor = Branch
    else:
        constructor = TENSORFIELDS[value.type].Request if type(value) is Leaf else type(value)
    bound = constructor.model_validate(payload)
    # Cloning retains materialized defaults (including extension factories),
    # while preserving which options the reusable definition supplied.
    if defaults is None:
        object.__setattr__(bound, "__pydantic_fields_set__", value.model_fields_set | {"name"})
    return bound


def bind_fields(
    fields: Mapping[str, TreeFieldInput] | None,
    children: Mapping[str, TreeFieldInput],
    *,
    defaults: Defaults | None = None,
) -> list[SchemaField]:
    """Merge parent-owned names and reject duplicates before binding any children."""
    if fields is None:
        fields = {}
    if not isinstance(fields, Mapping):
        raise TypeError("fields must be a mapping from child names to definitions")
    duplicates = fields.keys() & children.keys()
    if duplicates:
        raise ValueError(f"duplicate field name(s): {sorted(duplicates)}")
    return [bind_tree_field(name, value, defaults=defaults) for name, value in (*fields.items(), *children.items())]


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
    ffnx: ClassVar[None] = None  # pyrefly: ignore [bad-override]

    @pydantic.model_validator(mode="before")
    @classmethod
    def check_root_bounds(cls, data: Any) -> Any:
        if not isinstance(data, Mapping) or not isinstance(data.get("fields"), Mapping):
            return data
        root = dict(data["fields"])
        if root.get("length") is None:
            root["length"] = 1
        if root.get("overflow") is None:
            root["overflow"] = Overflow.error
        return {**data, "fields": root}

    @classmethod
    def from_tree(
        cls,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        ffnx: int = 8,
        fields: Mapping[str, TreeFieldInput] | None = None,
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionInput = AttentionMode.mha,
        cross: bool = False,
        reduction: ReductionConfig | None = Attention(),
        dropout: Rate | None = None,
        hide: HideInput = (),
        ignore: str | tuple[str, ...] = (),
        task: TaskInput = None,
        **field_kwargs: TreeFieldInput,
    ) -> Self:
        """Build schema from tree fields."""
        return build(
            cls,
            d_model=d_model,
            fields=fields,
            children=field_kwargs,
            options=dict(
                query=query,
                description=description,
                embed=embed,
                attention=attention,
                cross=cross,
                n_layers=n_layers,
                n_heads=n_heads,
                ffnx=ffnx,
                reduction=reduction,
                dropout=dropout,
                hide=hide,
                ignore=ignore,
                task=task,
            ),
        )

    def model_post_init(self, __context):
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

    @functools.cached_property
    def tasks(self) -> tuple[BoundTask, ...]:
        """Resolve declaration sites and active target leaves before building any model components."""
        supervised_owners: list[Address] = []
        supervised_targets: dict[Address, None] = {}
        tasks: list[BoundTask] = []
        for node in PreOrderIter(self.fields):
            # Twigs declare no tasks: they shape their branch and are never answers.
            if isinstance(node, Twig):
                continue
            for task in node.declared_tasks:
                site = "root" if node is self.fields else "leaf" if isinstance(node, Leaf) else "branch"
                task.validate_placement(site=site, address=node.address)
                rule = selection(task)
                if rule is None:
                    continue
                if isinstance(node, Leaf):
                    targets = (node.address,) if node.active else ()
                else:
                    targets = tuple(
                        leaf.address
                        for child in node.fields
                        if child.name not in rule.ignore
                        for leaf in PreOrderIter(child)
                        if isinstance(leaf, Leaf) and leaf.active
                    )
                if not targets:
                    continue
                if task is SUPERVISED:
                    supervised_owners.append(node.address)
                    supervised_targets.update(dict.fromkeys(targets))
                else:
                    tasks.append(BoundTask(task, (node.address,), targets))
        if supervised_owners:
            tasks.insert(0, BoundTask(SUPERVISED, tuple(supervised_owners), tuple(supervised_targets)))
        result = tuple(tasks)
        validate(result)
        return result

    @functools.cached_property
    def field_pools(self) -> dict[tuple[Address, Selection], tuple[Address, ...]]:
        """Eligible leaves in schema order for each declaration that samples whole fields."""
        return {
            (owner, rule): binding.targets
            for binding in self.tasks
            if (rule := selection(binding.task)) is not None and rule.fields is not None
            for owner in binding.owners
        }

    @functools.cached_property
    def roles(self) -> Roles:
        objectives = {address for binding in self.tasks for address in binding.targets}
        reconstruction = {
            address for binding in self.tasks if isinstance(binding.task, Reconstruction) for address in binding.targets
        }
        fixed_targets = frozenset(
            address
            for binding in self.tasks
            if (rule := selection(binding.task)) is not None and rule.fields is None
            for address in binding.targets
        )
        decodes = {
            address
            for binding in self.tasks
            if (rule := selection(binding.task)) is not None and not rule.sampled
            for address in binding.targets
        }
        embeds = tuple(
            node.address
            for node in PreOrderIter(self.fields)
            if node.embed and (not isinstance(node, Leaf) or node.active)
        )
        return Roles(
            tuple(a for a in self.requests if a in objectives),
            tuple(a for a in self.requests if a in decodes),
            embeds,
            tuple(a for a in self.requests if a in reconstruction),
            fixed_targets,
        )

    @property
    def reconstruct(self) -> list[Address]:
        return list(self.roles.reconstruct)

    @property
    def objectives(self) -> list[Address]:
        return list(self.roles.objectives)

    @property
    def decodes(self) -> list[Address]:
        return list(self.roles.decodes)

    @property
    def embed(self) -> list[Address]:  # noqa: F811
        return list(self.roles.embed)

    @functools.cached_property
    def branches(self) -> dict[Address, Branch]:
        return {node.address: node for node in self.descendants if isinstance(node, Branch)}

    @functools.cached_property
    def requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.descendants if isinstance(node, Leaf)}

    @functools.cached_property
    def twigs(self) -> dict[Address, Twig]:
        return {node.address: node for node in self.descendants if isinstance(node, Twig)}

    @functools.cached_property
    def active_requests(self) -> dict[Address, RequestTypes]:
        return {node.address: node for node in self.requests.values() if node.active}

    @functools.cached_property
    def shapes(self) -> dict[Address, tuple[int | None, ...]]:
        return {request.address: request.shape for request in self.requests.values()}

    @functools.cached_property
    def branch_outputs(self) -> dict[Address, int | None]:
        """Symbolic routed widths; ``None`` depends on the observed collections."""

        outputs: dict[Address, int | None] = {}
        for depth in reversed(self.depthwise):
            for address in depth:
                branch = self.branches[address]
                widths: list[int | None] = []
                for child in branch.fields:
                    if isinstance(child, Leaf):
                        widths.append(int(child.active))
                    elif child.address in outputs:
                        widths.append(outputs[child.address])

                if not any(width != 0 for width in widths):
                    outputs[address] = 0
                elif branch.reduction is None:
                    outputs[address] = (
                        None
                        if branch.length is None or any(width is None for width in widths)
                        else branch.length * sum(cast(int, width) for width in widths)
                    )
                elif isinstance(branch.reduction, Attention):
                    outputs[address] = branch.reduction.n_outputs
                else:
                    outputs[address] = 1

        return outputs

    @functools.cached_property
    def branch_widths(self) -> dict[Address, int | None]:
        """Symbolic encoded sequence widths per branch instance; ``None`` depends on the observed collections."""

        outputs = self.branch_outputs
        widths: dict[Address, int | None] = {}
        for address, branch in self.branches.items():
            parts = [
                int(child.active) if isinstance(child, Leaf) else outputs[child.address]
                for child in branch.fields
                if not isinstance(child, Twig)
            ]
            widths[address] = (
                None
                if branch.length is None or any(part is None for part in parts)
                else branch.length * sum(cast(int, part) for part in parts)
            )
        return widths

    def overflows(self, address: Address) -> tuple[Overflow | None, ...]:
        return (Overflow.error, *self.requests[Address(str(address))].overflows)

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
        for name in (
            "tasks",
            "field_pools",
            "roles",
            "branches",
            "requests",
            "twigs",
            "active_requests",
            "shapes",
            "branch_outputs",
            "branch_widths",
            "depthwise",
        ):
            self.__dict__.pop(name, None)

        for node in PreOrderIter(self.fields):
            for name in ("address", "heritage", "shape", "overflows"):
                node.__dict__.pop(name, None)

    def post_bind_validate(self) -> None:
        if self.fields.length != 1 or self.fields.overflow != Overflow.error:
            raise ValueError("model root must remain a singleton with length=1 and overflow='error'")
        if self.fields.name is not None:
            raise ValueError(
                f"model root must be anonymous, got name {self.fields.name!r}; "
                "rebuild the schema without a root name (named-root checkpoints are not supported)"
            )
        for branch in self.branches.values():
            branch.post_bind_validate()
            if branch.attention is not None and (
                self.d_model % branch.n_heads != 0 or self.d_model // branch.n_heads < 2
            ):
                raise ValueError(
                    f"branch '{branch.address}' encoder: d_model must be divisible by nhead "
                    f"with at least two dimensions per head; got d_model={self.d_model}, "
                    f"n_heads={branch.n_heads}; override d_model or this branch's n_heads"
                )
            if branch.embed and self.branch_outputs[branch.address] == 0:
                raise ValueError(
                    f"branch '{branch.address}' has embed=True but no active descendant output; "
                    "activate at least one descendant leaf or disable branch embedding"
                )
            if isinstance(branch.reduction, Attention):
                n_heads = branch.reduction.n_heads or branch.n_heads
                minimum = 2 if branch.reduction.position else 1
                if self.d_model % n_heads != 0 or self.d_model // n_heads < minimum:
                    raise ValueError(
                        f"branch '{branch.address}' Attention reduction requires n_heads to divide "
                        f"d_model with at least {minimum} dimensions per head; got d_model={self.d_model}, "
                        f"n_heads={n_heads}, position={branch.reduction.position}; "
                        "override d_model or the reduction's n_heads"
                    )

        for twig in self.twigs.values():
            twig.post_bind_validate()

        decoded = {*self.objectives, *self.embed}
        for request in self.requests.values():
            request.post_bind_validate()
            if request.address not in decoded or request.pooling != "query":
                continue
            positional = (
                any(length is None or length > 1 for length in request.shape)
                if request.decoder_position is None
                else request.decoder_position
            )
            minimum = 2 if positional else 1
            if self.d_model % request.n_heads != 0 or self.d_model // request.n_heads < minimum:
                raise ValueError(
                    f"leaf '{request.address}' query decoder requires n_heads to divide d_model "
                    f"with at least {minimum} dimensions per head; got d_model={self.d_model}, "
                    f"n_heads={request.n_heads}, decoder_position={request.decoder_position}; "
                    "override d_model or this leaf's n_heads"
                )

        self.validate_capabilities()

    def validate_capabilities(self) -> None:
        """Validate effective leaf roles against registered extension components."""

        from relflow.tensorfields.base import TENSORFIELDS

        objectives = set(self.roles.objectives)
        for address, request in self.active_requests.items():
            required: list[Component] = []
            if address in objectives:
                required.extend((Component.Decoder, Component.loss))
            elif request.embed:
                required.append(Component.Decoder)

            extension = TENSORFIELDS[request.type]
            missing = [component.value for component in required if component not in extension.components]
            if not missing:
                continue

            names = ", ".join(missing)
            if address in objectives:
                bindings = [binding for binding in self.tasks if address in binding.targets]
                sources = ", ".join(
                    f"{binding.task.kind} at '{owner}'"
                    for binding in bindings
                    for owner in binding.owners
                    if owner == "/" or owner == address or str(address).startswith(str(owner) + "/")
                )
                raise ValueError(
                    f"task {sources} reaches leaf '{address}', but extension "
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
    ) -> list[Node]:
        if len(predicates) == 1:
            attribute = predicates[0]
            if type(attribute) is NodeAttribute and attribute.name == "reconstruct":
                requests = self.requests
                return [requests[address] for address in self.roles.reconstruct]
        normalized = tuple(NodePredicate.from_selector(item) for item in predicates)
        return [
            node
            for node in PreOrderIter(self.fields)
            if (include_root or node is not self.fields)
            if all(predicate(node) for predicate in normalized)
        ]

    def update(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> None:
        """Mutate matching schema nodes."""
        if not values:
            raise ValueError("update requires at least one field value")
        nodes = self.select(*predicates, include_root=include_root)
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
                    check_policy_options(applicable_values, branch=isinstance(node, Branch))
                    excluded = set(applicable_values)
                    if isinstance(node, Branch):
                        excluded.add("fields")
                    payload = node.model_dump(mode="python", round_trip=True, exclude=excluded)
                    if isinstance(node, Branch):
                        payload["fields"] = [bind_tree_field(cast(str, child.name), child) for child in node.fields]
                    payload.update(applicable_values)
                    validated = type(node).model_validate(payload)
                    applicable_values = {name: getattr(validated, name) for name in applicable_values}
                    if isinstance(node, Branch) and "length" in applicable_values:
                        applicable_values["overflow"] = cast(Branch, validated).overflow

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
            raise

    def extend(
        self,
        *predicates: NodeSelector,
        fields: Mapping[str, TreeFieldInput] | None = None,
        include_root: bool = True,
        **children: TreeFieldInput,
    ) -> None:
        """Append parent-named fields under the single branch selected by predicates."""
        self.graft(*predicates, fields=bind_fields(fields, children), include_root=include_root)

    def graft(
        self,
        *predicates: NodeSelector,
        fields: list[SchemaField],
        include_root: bool = True,
    ) -> None:
        """Attach independent bound fields and restore the tree if validation fails."""
        if not fields:
            raise ValueError("extend requires at least one schema field")
        candidates = [node for node in self.select(*predicates, include_root=include_root) if isinstance(node, Branch)]
        if len(candidates) != 1:
            raise ValueError(f"extend requires exactly one matching branch node, found {len(candidates)}")
        parent = candidates[0]
        existing_names = {field.name for field in parent.fields}
        duplicate_names = sorted({cast(str, field.name) for field in fields if field.name in existing_names})
        if duplicate_names:
            raise ValueError(f"duplicate field name(s): {duplicate_names}")

        original_fields = list(parent.fields)
        try:
            parent.fields.extend(fields)
            for field in fields:
                field.parent = parent

            self.clear_tree_caches()
            self.post_bind_validate()
        except Exception:
            parent.fields = original_fields
            for field in fields:
                field.parent = None
            self.clear_tree_caches()
            self.post_bind_validate()
            raise

    def delete(
        self,
        *predicates: NodeSelector,
        include_root: bool = False,
    ) -> None:
        """Permanently remove selected schema nodes from the tree."""
        if not predicates:
            raise ValueError("delete requires at least one predicate")

        selected = self.select(*predicates, include_root=include_root)
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
            if isinstance(parent, Branch) and (
                node.name in parent.ignore
                or any(
                    (rule := selection(task)) is not None and node.name in rule.ignore for task in parent.declared_tasks
                )
            ):
                raise ValueError(
                    f"delete would remove ignored child '{node.address}'; update its parent's ignore names first"
                )

        parents: dict[int, tuple[Branch, list[SchemaField]]] = {}
        for node in roots:
            parent = node.parent
            if not isinstance(parent, Branch):
                raise ValueError(f"delete cannot remove '{node.address}' because it has no branch parent")
            parents[id(parent)] = (parent, parent.fields)

        try:
            for parent, children in parents.values():
                parent.fields = [child for child in children if id(child) not in selected_ids]
                parent.children = tuple(parent.fields)
            self.clear_tree_caches()
            self.post_bind_validate()
        except Exception:
            for parent, children in parents.values():
                parent.fields = children
                parent.children = tuple(children)
            self.clear_tree_caches()
            raise

    @contextmanager
    def override(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> Generator[None, None, None]:
        nodes = self.select(*predicates, include_root=include_root)
        # ``update`` re-derives a Branch's overflow whenever its length changes,
        # so restoring the length alone can leave an invalid pair behind.
        snapshot = [
            (
                node,
                name,
                getattr(node, name, _MISSING),
                name in getattr(node, "model_fields_set", set()),
            )
            for node in nodes
            for name in ([*values, "overflow"] if isinstance(node, Branch) and "length" in values else values)
            if has_model_attribute(node, name)
        ]

        self.update(
            *predicates,
            strict=strict,
            include_root=include_root,
            validate=validate,
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


S = TypeVar("S", bound=Schema)


def build(
    schema: type[S],
    /,
    *,
    d_model: int,
    fields: Mapping[str, TreeFieldInput] | None,
    children: Mapping[str, TreeFieldInput],
    options: Mapping[str, Any],
    defaults: Defaults | None = None,
) -> S:
    """Bind named inputs once and validate their anonymous singleton root."""
    check_root_policies(children)
    check_policy_options(options, branch=True)
    bound = bind_fields(fields, children, defaults=defaults)
    if not bound:
        raise ValueError("from_tree requires at least one field")
    root = Branch.model_validate({**options, "length": 1, "overflow": Overflow.error, "fields": bound})
    return schema(d_model=d_model, fields=root)
