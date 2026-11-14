"""Sequence mixers of a branch, picked once from a table by mixer setting."""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping
from functools import partial
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, Protocol, cast, override

import torch

from relflow.architecture.direction import Rule, bidirectional
from relflow.architecture.packed import Packed
from relflow.structs.mixer import Conv, Transformer

if TYPE_CHECKING:
    from relflow.architecture.encoder import RotaryTransformerEncoderLayer
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


class Mixer(Protocol):
    """Mix the tokens of every branch sequence.

    A mixer is a stack of layers: the compiler reads its length and packing
    iterates over its layers.
    """

    def __len__(self) -> int:
        """Number of layers in the stack."""
        ...

    def __iter__(self) -> Iterator[torch.nn.Module]:
        """Iterate over the layers in order."""
        ...

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
    """

    def __init__(self, layers: Iterable[torch.nn.Module] = (), *, rule: Rule = bidirectional) -> None:
        super().__init__(layers)
        self.rule = rule

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
        allowed = None if packing is not None else self.rule(steps=steps)
        for layer in cast("Iterable[RotaryTransformerEncoderLayer]", self):
            # Layers see the keyword only when it applies, so replaced layers keep their arity.
            apply = layer if allowed is None else partial(layer, allowed=allowed)
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
    return TransformerMixer(layers, rule=rule)


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


def pending(mixer: Conv, *, branch: Branch, schema: Schema, dropout: float, rule: Rule) -> NoReturn:
    """Reject a mixer this relflow version does not build.

    Args:
        mixer: The unsupported setting.
        branch: Branch that owns the sequence.
        schema: Schema that owns the branch.
        dropout: Dropout inherited from the branch.
        rule: Direction rule of the branch.

    Raises:
        ValueError: Always, naming the branch and the supported mixers.
    """
    message = (
        f"branch '{branch.address}' sets mixer={mixer!r}, which this relflow version does not build yet; "
        "use rf.Transformer() or mixer=None"
    )
    raise ValueError(message)


mixers: Mapping[type, MixerBuilder[Any]] = MappingProxyType({Transformer: transformer, Conv: pending, type(None): idle})

__all__ = ["Mixer", "MixerBuilder", "TransformerMixer", "idle", "mixers", "pending", "transformer"]
