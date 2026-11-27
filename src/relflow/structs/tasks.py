"""Learning task declarations, their ownership of answers and their compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import singledispatch
from typing import TYPE_CHECKING, Annotated, Any, ClassVar, Literal, Self, TypeAlias, TypeVar

import pydantic

if TYPE_CHECKING:
    from relflow.structs.tree import Address


def check_probability(value: Any) -> float:
    if not isinstance(value, float):
        raise TypeError(f"expected a float probability, got {value!r}")
    return value


Probability: TypeAlias = Annotated[float, pydantic.Field(ge=0.0, le=1.0), pydantic.BeforeValidator(check_probability)]


def check_ignore(value: Any) -> tuple[str, ...]:
    names = (value,) if isinstance(value, str) else value
    if not isinstance(names, (tuple, list)) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError(f"ignore requires a direct child name or tuple of names, got {value!r}")
    if len(set(names)) != len(names):
        raise ValueError(f"ignore requires distinct direct child names, got {value!r}")
    return tuple(names)


Names: TypeAlias = Annotated[tuple[str, ...], pydantic.BeforeValidator(check_ignore)]


@dataclass(frozen=True, slots=True)
class Selection:
    """One action on selected field positions in the Arrow data pipeline."""

    action: Literal["hide", "predict", "reconstruct"]
    query: str | None = None
    rate: float | None = None
    span: int | None = None
    ignore: tuple[str, ...] = ()
    ablate: bool = False
    fields: int | None = None

    @property
    def sampled(self) -> bool:
        return self.rate is not None or self.fields is not None

    @property
    def skips_input(self) -> bool:
        return self.action == "predict" or self.ablate


TaskType = TypeVar("TaskType", bound="Task")
DECLARATIONS: dict[str, type[Task]] = {}


class Task(pydantic.BaseModel):
    """A typed learning objective. Both tasks must accept a proposed combination."""

    model_config = pydantic.ConfigDict(frozen=True, extra="forbid")
    kind: str
    # Tasks sharing an objective can use the same pristine answer and per-leaf loss.
    shared_objective: ClassVar[str | None] = None

    @staticmethod
    def register(declaration: type[TaskType]) -> type[TaskType]:
        """Register a distinct serialized kind so the task's contract survives schema copies."""
        kind = declaration.model_fields["kind"].default
        if not isinstance(kind, str) or not kind or kind == "supervised":
            raise ValueError("an explicit task needs a distinct string kind; supervised is reserved for predict=True")
        if kind in DECLARATIONS and DECLARATIONS[kind] is not declaration:
            raise ValueError(f"task kind {kind!r} is already registered")
        DECLARATIONS[kind] = declaration
        return declaration

    def conflict(self, other: Task, *, shared: frozenset[str]) -> str | None:
        """Return a reason to reject the pair. By default each task owns its answers exclusively."""
        if self.shared_objective is not None and self.shared_objective == other.shared_objective:
            return None
        return "both tasks claim the same answers" if shared else None

    def validate_placement(self, *, site: Literal["root", "branch", "leaf"], address: str | None) -> None:
        """Check the node kind and, once bound, its absolute schema address.

        A reusable leaf is checked with ``address=None`` before it has a parent.
        Building or editing the tree checks each declaration at its full address.
        """


@singledispatch
def selection(task: Task) -> Selection | None:
    """Compile an implemented task for the field pipeline independently of its compatibility contract."""
    raise ValueError(f"task {task.kind!r} has no execution implementation for field objectives")


class Supervised(Task):
    """One implicit task for all predict=True declarations in a model."""

    kind: Literal["supervised"] = "supervised"


@selection.register
def supervise(task: Supervised) -> Selection:
    return Selection("predict")


SUPERVISED = Supervised()


@Task.register
class Reconstruction(Task):
    """Beta self-supervised task that learns to recover selected input values."""

    kind: Literal["reconstruct"] = "reconstruct"
    shared_objective: ClassVar[str | None] = "reconstruct"
    rate: Probability | None = None
    query: str | None = None
    span: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] | None = None
    ignore: Names = ()
    ablate: pydantic.StrictBool = False
    fields: Annotated[pydantic.StrictInt, pydantic.Field(gt=0)] | None = None

    @pydantic.model_validator(mode="after")
    def check_selection(self) -> Self:
        if self.span is not None and self.rate is None:
            raise ValueError("reconstruct(span=...) requires a rate for sampling row spans")
        if self.query is not None:
            from relflow.data.query import compile

            compile(self.query)
        return self

    def validate_placement(self, *, site: Literal["root", "branch", "leaf"], address: str | None) -> None:
        if site == "leaf":
            leaf = f"leaf '{address}'" if address is not None else "a leaf"
            if self.fields is not None or self.ignore:
                option = "fields" if self.fields is not None else "ignore"
                raise ValueError(
                    f"reconstruct({option}=...) requires a branch or model with child fields. Received {leaf}."
                )
            if self.span is not None:
                raise ValueError(f"reconstruct(span=...) requires a child branch with repeated rows. Received {leaf}.")
        if site == "root" and self.span is not None:
            raise ValueError("reconstruct(span=...) selects repeated child branch rows; the model root is a singleton")


@selection.register
def reconstruct_fields(task: Reconstruction) -> Selection | None:
    if task.rate == 0.0:
        return None
    return Selection(
        "reconstruct",
        task.query,
        None if task.rate == 1.0 else task.rate,
        task.span,
        task.ignore,
        ablate=task.ablate,
        fields=task.fields,
    )


def reconstruct(
    *,
    rate: float | None = None,
    query: str | None = None,
    span: int | None = None,
    ignore: str | tuple[str, ...] = (),
    ablate: bool = False,
    fields: int | None = None,
) -> Reconstruction:
    """Create an opt-in self-supervised reconstruction task (beta).

    Select observations by Boolean query, sampling probability, or a run of child branch rows.
    Omit selectors to reconstruct every observation. A branch can exempt direct children by name.
    With ``ablate=True``, selected coordinates leave the encoder and keep their original positions as answers.
    On a model or branch, ``fields=N`` samples up to N distinct active descendant leaves per batch.
    The selected leaves share the owner's row selection. Other leaves remain available as context.
    """
    if isinstance(ignore, list):
        raise TypeError("reconstruct(ignore=...) expects a direct child name or a tuple of names")
    return Reconstruction(
        rate=rate,
        query=query,
        span=span,
        ignore=(ignore,) if isinstance(ignore, str) else ignore,
        ablate=ablate,
        fields=fields,
    )


def restore_task(value: Any) -> Task:
    """Restore the concrete declaration from its persisted kind at the schema boundary."""
    if isinstance(value, Task) and DECLARATIONS.get(value.kind) is type(value):
        return value
    kind = value.get("kind") if isinstance(value, Mapping) else value
    if isinstance(kind, str) and kind in DECLARATIONS:
        return DECLARATIONS[kind].model_validate(value)
    raise ValueError(
        f"unknown explicit task kind {kind!r}. Choose from {sorted(DECLARATIONS)} or declare predict=True for supervision"
    )


TaskValue: TypeAlias = Annotated[pydantic.SerializeAsAny[Task], pydantic.BeforeValidator(restore_task)]
TaskInput: TypeAlias = TaskValue | tuple[TaskValue, ...] | None


@dataclass(frozen=True, slots=True)
class BoundTask:
    """A task with the schema addresses that declare it and the active leaves it teaches."""

    task: Task
    owners: tuple[Address, ...]
    targets: tuple[Address, ...]


def validate(tasks: tuple[BoundTask, ...]) -> None:
    """Skip mutually compatible standard contracts. A custom contract can veto any peer."""
    if len(tasks) < 2:
        return
    groups: dict[str | None, list[tuple[BoundTask, frozenset[str]]]] = {}
    for right in tasks:
        group = right.task.shared_objective if type(right.task).conflict is Task.conflict else None
        targets: frozenset[str] = frozenset(right.targets)
        for other_group, peers in groups.items():
            if group is not None and group == other_group:
                continue
            for left, left_targets in peers:
                shared = left_targets & targets
                reason = left.task.conflict(right.task, shared=shared) or right.task.conflict(left.task, shared=shared)
                if reason:
                    overlaps = f" at {', '.join(sorted(shared))}" if shared else ""
                    raise ValueError(
                        f"task conflict between {left.task.kind} on {', '.join(left.owners)} and "
                        f"{right.task.kind} on {', '.join(right.owners)}: {reason}{overlaps}"
                    )
        groups.setdefault(group, []).append((right, targets))
