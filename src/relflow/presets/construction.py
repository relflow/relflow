"""Generate typed model classmethods from immutable preset configurations."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, cast

from relflow.presets.base import Preset, resolve
from relflow.structs.enums import AttentionInput
from relflow.structs.experiment import TreeFieldInput
from relflow.structs.reduction import ReductionConfig
from relflow.structs.structure import OMITTED
from relflow.structs.tree import MaskInput, Rate

if TYPE_CHECKING:
    from relflow.architecture.root import Model, OptimizerConfig, SchedulerConfig

__all__ = ["factory"]


def factory(profile: Preset):
    """Capture a preset in a classmethod with an inferred signature and subclass return type."""
    if not isinstance(profile, Preset):
        raise TypeError(f"factory requires a Preset, got {type(profile).__name__}")
    # Revalidate copies before capturing the immutable construction policy.
    profile = Preset.model_validate(profile.model_dump(mode="python", round_trip=True))

    def construct[M: Model](
        cls: type[M],
        /,
        *,
        d_model: int = cast(int, OMITTED),
        n_layers: int = cast(int, OMITTED),
        n_heads: int = cast(int, OMITTED),
        batch_size: int = 1,
        fields: Mapping[str, TreeFieldInput] | None = None,
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionInput = cast(AttentionInput, OMITTED),
        reduction: ReductionConfig | None = cast(ReductionConfig | None, OMITTED),
        dropout: Rate | None = cast(Rate | None, OMITTED),
        mask: MaskInput = False,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        **field_kwargs: TreeFieldInput,
    ) -> M:
        if "schema" in field_kwargs:
            raise TypeError("Model preset factories cannot accept schema; use Model(schema=...) for a resolved schema")
        schema = resolve(
            profile,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            fields=fields,
            query=query,
            description=description,
            embed=embed,
            attention=attention,
            reduction=reduction,
            dropout=dropout,
            mask=mask,
            **field_kwargs,
        )
        model = cls(schema=schema, batch_size=batch_size, optimizer=optimizer, scheduler=scheduler)
        model.preset = profile
        return model

    name = profile.name
    construct.__name__ = name
    construct.__qualname__ = f"Model.{name}"
    construct.__doc__ = f"Build the {name} preset; explicit options override defaults at their own node."
    return classmethod(construct)
