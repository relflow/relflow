"""Structural schema nodes that shape their branch without becoming tokens."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar, Literal, Self, cast

import pydantic

from relflow.structs.tree import Address, Node

if TYPE_CHECKING:
    from relflow.structs.structure import Branch

__all__ = ["TWIGS", "Index", "Twig", "indexed"]

TWIGS: dict[str, type[Twig]] = {}


class Twig(Node):
    """Base schema node that shapes its parent branch without becoming a token.

    A twig reads one value per record of its branch, as a leaf does, but it
    is never embedded, masked, decoded, or written, and it owns no
    architecture options. Its parent supplies its name. ``query`` selects
    another member of the same record; the twig otherwise reads its
    same-named member. Concrete twigs register their ``type`` discriminator
    when their class is defined.
    """

    model_config = pydantic.ConfigDict(validate_default=True)

    # A twig contributes no token, so the shared node architecture options do not apply.
    embed: ClassVar[None] = None  # pyrefly: ignore [bad-override]
    n_heads: ClassVar[None] = None  # pyrefly: ignore [bad-override]
    ffnx: ClassVar[None] = None  # pyrefly: ignore [bad-override]
    dropout: ClassVar[None] = None  # pyrefly: ignore [bad-override]
    query: str | None = None

    def __init__(self, **data: Any) -> None:
        if "name" in data:
            raise TypeError("twig names come from the parent; use field_name=Twig(...)")
        super().__init__(**data)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        kind = cls.model_fields["type"].default
        if isinstance(kind, str):
            TWIGS[kind] = cls

    @pydantic.model_validator(mode="after")
    def check_n_heads_is_even(self) -> Self:
        """Replace the inherited head check; a twig owns no attention heads."""
        return self

    @pydantic.model_validator(mode="after")
    def check_query(self) -> Self:
        if self.query is not None:
            from relflow.data.query import scope

            scope(self.query)
        return self

    def queries(self) -> tuple[str, ...]:
        """Return the explicit path, or the same-named member after the twig is bound."""
        from relflow.data.query import member

        return (member(cast(str, self.name)),) if self.query is None else (self.query,)

    def post_bind_validate(self) -> None:
        """Keep the source within the twig's own branch records."""
        from relflow.data.query import origin

        (expression,) = self.queries()
        depth, _ = origin(expression, address=str(self.address))
        if depth:
            raise ValueError(
                f"{type(self).__name__} '{self.address}' query {expression!r} reads an ancestor branch; "
                "a twig reads one value from each record of its own branch"
            )


class Index(Twig):
    """Position the records of a branch at their index values in rotary attention.

    An index reads one integer, date, or timestamp per record of its branch.
    That branch's encoder then rotates every token of a record by the record's
    index instead of its ordinal, so attention sees index differences, such
    as the time between two events, rather than how many records lie between
    them. Integer indexes count their own units. Dates and timestamps count
    ``granularity`` units, which default to days; integer indexes reject
    ``granularity``. Positions start at the smallest index of each collection,
    so records need not be sorted and only differences reach attention.

    A branch holds at most one Index, and the model root, which holds one
    record per observation, holds none. Every retained record needs a
    non-null index. The index is never embedded or predicted; model the same
    member with another field, such as ``rf.Period(query=...)``, when it is
    also a feature. Sequences routed to the parent branch and to decoders
    keep their structural positions.
    """

    type: Literal["index"] = "index"
    granularity: Literal["seconds", "minutes", "hours", "days", "weeks"] | None = None

    def post_bind_validate(self) -> None:
        super().post_bind_validate()
        from relflow.structs.reduction import Attention

        branch = cast("Branch", self.parent)
        if branch.address == Address():
            raise ValueError(
                f"Index '{self.address}' orders the records of a repeated Branch, but the model root holds "
                "one record per observation; move the Index into the child Branch whose records it orders"
            )
        others = [cast(str, child.name) for child in branch.fields if isinstance(child, Index)]
        if len(others) > 1:
            raise ValueError(f"branch '{branch.address}' declares Index twigs {others}; a branch takes at most one")
        positional = isinstance(branch.reduction, Attention) and branch.reduction.position
        if branch.attention is None and not positional:
            raise ValueError(
                f"Index '{self.address}' has no rotary attention to position: branch '{branch.address}' sets "
                "attention=None and its reduction has no position; enable attention or remove the Index"
            )


def indexed(branch: Branch) -> Index | None:
    """Return the Index that positions a branch's records, if it declares one."""
    return next((child for child in branch.fields if isinstance(child, Index)), None)
