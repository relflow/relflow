"""Sequence mixers of a branch, picked once from a table by mixer setting."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Protocol, cast, override

import torch

from relflow.architecture.attention import merge
from relflow.architecture.convolution import ConvMixer
from relflow.architecture.direction import Rule, bidirectional, triangular
from relflow.architecture.packed import Packed
from relflow.structs.grain import Steps
from relflow.structs.mixer import Conv, Transformer

if TYPE_CHECKING:
    from relflow.architecture.encoder import RotaryTransformerEncoderLayer
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


class Mixer(Protocol):
    """Mix the tokens of every branch sequence.

    Transformer stacks are the one mixer the compiler and packing know by type;
    every other mixer only has to be called.
    """

    def __call__(
        self,
        tokens: torch.Tensor,
        *,
        present: torch.Tensor | None,
        positions: torch.Tensor | None,
        steps: torch.Tensor | None,
        packing: Packed | None,
    ) -> torch.Tensor:
        """Return mixed tokens shaped like ``tokens``, ``(rows, L, d_model)``."""
        ...


class MixerBuilder[S](Protocol):
    """Build the mixer of one branch from its mixer setting."""

    def __call__(self, mixer: S, *, branch: Branch, schema: Schema, dropout: float, rule: Rule) -> Mixer: ...


class TransformerMixer(torch.nn.ModuleList):
    """Apply rotary transformer layers in order.

    Subclassing ``ModuleList`` keeps the saved state keys ``encoder.{i}.*``,
    ``len`` for the compiler and iteration for packing.

    Attributes:
        rule: Direction rule fixed when the branch is built.
        triangular: Whether the rule on one token per row is a plain causal mask, which SDPA builds itself.
    """

    def __init__(
        self, layers: Iterable[torch.nn.Module] = (), *, rule: Rule = bidirectional, triangular: bool = False
    ) -> None:
        super().__init__(layers)
        self.rule = rule
        self.triangular = triangular

    @override
    def forward(
        self,
        tokens: torch.Tensor,
        *,
        present: torch.Tensor | None,
        positions: torch.Tensor | None,
        steps: torch.Tensor | None,
        packing: Packed | None,
    ) -> torch.Tensor:
        """Encode prepared sequences; absent presence means every token is live.

        Layers receive structural positions whenever the branch has them.

        Args:
            tokens: Tokens of shape ``(rows, L, d_model)``, or packed live tokens.
            present: Boolean presence of shape ``(rows, L)``, or ``None`` for all live.
            positions: Structural positions, or ``None`` when the branch has none.
            steps: Row index of each token, read by the direction rule; ``None`` for bidirectional.
            packing: Layout of packed live tokens, or ``None`` for dense input.

        Returns:
            Mixed tokens shaped like ``tokens``.
        """
        extra: dict[str, Any] = {}
        # Padding-free rows need no explicit mask; the check is one host sync per forward.
        if packing is None and self.triangular and present is not None and bool(present.all()):
            present, extra = None, {"causal": True}
        elif packing is None:
            allowed = self.rule(steps=steps)
            # Built once here: every layer would otherwise rebuild the same (rows, L, L) mask.
            extra = {} if allowed is None else {"mask": merge(allowed, present=cast(torch.Tensor, present))}
        for layer in cast("Iterable[RotaryTransformerEncoderLayer]", self):
            # Layers see the keywords only when they apply, so replaced layers keep their arity.
            apply = partial(layer, **extra) if extra else layer
            if packing is not None:
                tokens = layer(tokens, present=present, packing=packing)
            elif positions is not None:
                tokens = apply(tokens, present=present, positions=positions)
            else:
                tokens = apply(tokens, present=present)
        return tokens


def transformer(mixer: Transformer, *, branch: Branch, schema: Schema, dropout: float, rule: Rule) -> TransformerMixer:
    """Build the standard transformer stack; ``attention=None`` leaves it empty.

    Args:
        mixer: The transformer setting.
        branch: Branch that owns the sequence.
        schema: Schema that owns the branch.
        dropout: Dropout inherited from the branch.
        rule: Direction rule of the branch.

    Returns:
        A stack of ``branch.n_layers`` layers, or an empty stack without attention.
    """
    from relflow.architecture.encoder import RotaryTransformerEncoderLayer

    attention = branch.attention
    if attention is None:
        return TransformerMixer(rule=rule)
    layers = [
        RotaryTransformerEncoderLayer(
            d_model=schema.d_model,
            nhead=branch.n_heads,
            n_kv_heads=attention.kv_heads(branch.n_heads),
            dropout=dropout,
            ffnx=branch.ffnx,
        )
        for _ in range(branch.n_layers)
    ]
    return TransformerMixer(
        layers, rule=rule, triangular=triangular[branch.direction] and isinstance(branch.grain, Steps)
    )


def idle(mixer: None, *, branch: Branch, schema: Schema, dropout: float, rule: Rule) -> TransformerMixer:
    """Build no sequence mixing while coordinate attention still runs.

    Args:
        mixer: Always ``None``.
        branch: Branch that owns the sequence.
        schema: Schema that owns the branch.
        dropout: Dropout inherited from the branch.
        rule: Direction rule of the branch.

    Returns:
        An empty stack.
    """
    return TransformerMixer(rule=rule)


def conv(mixer: Conv, *, branch: Branch, schema: Schema, dropout: float, rule: Rule) -> ConvMixer:
    """Build the dilated convolution mixer of a branch that holds one token per row.

    Args:
        mixer: The convolution setting.
        branch: Branch that owns the sequence; its direction sets the padding.
        schema: Schema that owns the branch.
        dropout: Unused; the convolution has no dropout.
        rule: Unused; the convolution pads by direction instead of masking.

    Returns:
        A convolution mixer of ``mixer.n_blocks`` blocks.
    """
    return ConvMixer(width=schema.d_model, n_blocks=mixer.n_blocks, direction=branch.direction)


mixers: Mapping[type, MixerBuilder[Any]] = MappingProxyType({Transformer: transformer, Conv: conv, type(None): idle})

__all__ = ["Mixer", "MixerBuilder", "TransformerMixer", "conv", "idle", "mixers", "transformer"]
