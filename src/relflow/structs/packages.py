import torch
from tensordict import TensorClass, TensorDict

from relflow.structs.tree import Address


class Parcel(TensorClass):
    """One tensor routed between tree nodes with its position geometry.

    ``positions`` number tokens within each instance of the destination branch
    and ``spans`` count that instance's positions. ``slots`` mark tokens that
    stand for real records. ``widths`` count the positions each destination
    item owns in this parcel; leaves omit it and own one position per item.
    """

    payload: torch.Tensor
    present: torch.Tensor
    origin: Address
    destination: Address | None
    positions: torch.Tensor | None = None
    spans: torch.Tensor | None = None
    slots: torch.Tensor | None = None
    widths: torch.Tensor | None = None


# @jaxtyped(typechecker=beartype)
class Prediction(TensorClass):
    address: Address
    payload: TensorDict
    geometry: TensorDict | None = None
    positions: torch.Tensor | None = None
    spans: torch.Tensor | None = None
    slots: torch.Tensor | None = None
