import torch
from tensordict import TensorClass, TensorDict

from relflow.structs.tree import Address


class Parcel(TensorClass):
    payload: torch.Tensor
    present: torch.Tensor
    origin: Address
    destination: Address | None
    positions: torch.Tensor | None = None
    spans: torch.Tensor | None = None
    slots: torch.Tensor | None = None


# @jaxtyped(typechecker=beartype)
class Prediction(TensorClass):
    address: Address
    payload: TensorDict
    geometry: TensorDict | None = None
    positions: torch.Tensor | None = None
    spans: torch.Tensor | None = None
    slots: torch.Tensor | None = None
