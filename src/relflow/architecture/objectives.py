"""Weighted training objectives keyed by the address that owns them."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import torch

from relflow.data.datasets.base import EncodedInput
from relflow.structs.enums import Strata
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["Evaluate", "Objective", "Select", "objectives", "reconstruction"]

type Select = Callable[[EncodedInput, Strata], torch.Tensor]
type Evaluate = Callable[..., torch.Tensor]


@dataclass(frozen=True, kw_only=True)
class Objective:
    """One weighted scalar loss scored at an address.

    Attributes:
        address: Node address that owns the objective and the logged metric.
        weight: Factor applied to the unweighted loss.
        select: Returns a 0-d boolean tensor, true when the batch holds anything this objective scores.
        evaluate: Returns the unweighted scalar loss, called with the keywords `module`, `prediction`,
            `inputs`, and `strata`.
    """

    address: Address
    weight: float
    select: Select
    evaluate: Evaluate


def targets(inputs: EncodedInput, strata: Strata, *, address: Address) -> torch.Tensor:
    """Report whether any observation at a leaf is a reconstruction target."""
    return cast(TensorFieldBase, inputs[address]).trainable.any()


def score(
    *, module: "Model", prediction: Prediction, inputs: EncodedInput, strata: Strata, address: Address
) -> torch.Tensor:
    """Evaluate the loss of the extension that owns a leaf."""
    return TENSORFIELDS[module.schema.requests[address].type].loss(
        module=module, prediction=prediction, batch=cast(TensorFieldBase, inputs[address]), strata=strata
    )


def reconstruction(module: "Model", *, address: Address) -> Objective:
    """Build the objective of a reconstructed leaf.

    Args:
        module: Model that owns the schema.
        address: Leaf address.

    Returns:
        The weighted leaf objective.
    """
    return Objective(
        address=address,
        weight=module.schema.requests[address].weight,
        select=partial(targets, address=address),
        evaluate=partial(score, address=address),
    )


def objectives(module: "Model") -> tuple[Objective, ...]:
    """Build every objective of a model in schema order.

    Args:
        module: Model that owns the schema.

    Returns:
        Weighted objectives in pre-order.
    """
    return tuple(reconstruction(module, address=address) for address in module.schema.objectives)
