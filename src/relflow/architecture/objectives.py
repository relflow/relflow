"""Weighted training objectives keyed by the address that owns them."""

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, cast

import torch
from anytree import PreOrderIter
from tensordict import TensorDict

from relflow.data.arrow import Geometry
from relflow.data.datasets.base import EncodedInput
from relflow.structs.contrastive import Contrastive
from relflow.structs.enums import Metric, Strata, TensorKey
from relflow.structs.packages import Prediction
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["Evaluate", "Objective", "Select", "contrast", "objectives", "reconstruction"]

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


def holds(inputs: EncodedInput, strata: Strata, *, address: Address) -> torch.Tensor:
    """Report whether a branch has records and trains its crops in this batch."""
    geometry = cast(Geometry, cast(TensorDict, inputs[Address()])[address])
    return geometry.slots.any() & (geometry.views is not None)


def divergence(
    *, module: "Model", prediction: Prediction, inputs: EncodedInput, strata: Strata, address: Address
) -> torch.Tensor:
    """Sum the per-record shares of a branch objective and log the loss at the branch."""
    shares = cast(torch.Tensor, prediction.payload[TensorKey.objective])
    return cast(torch.Tensor, module.track((address, strata, Metric.loss), value=shares.sum()))


def contrast(module: "Model", *, address: Address) -> Objective:
    """Build the contrastive objective of a branch.

    Args:
        module: Model that owns the schema.
        address: Branch address.

    Returns:
        The weighted branch objective.
    """
    settings = cast(Contrastive, module.schema.branches[address].contrastive)
    return Objective(
        address=address,
        weight=settings.weight,
        select=partial(holds, address=address),
        evaluate=partial(divergence, address=address),
    )


def objectives(module: "Model") -> tuple[Objective, ...]:
    """Build every objective of a model in schema order.

    Args:
        module: Model that owns the schema.

    Returns:
        Weighted leaf and branch objectives in pre-order of the schema.
    """
    leaves = (reconstruction(module, address=address) for address in module.schema.objectives)
    branches = (
        contrast(module, address=address)
        for address, branch in module.schema.branches.items()
        if branch.contrastive is not None
    )
    order = {Address(str(node.address)): index for index, node in enumerate(PreOrderIter(module.schema.fields))}
    return tuple(sorted((*leaves, *branches), key=lambda entry: order[entry.address]))
