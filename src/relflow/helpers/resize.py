"""Stage tensor-row growth while preserving gradients and optimizer history."""

from collections.abc import Callable, Sequence
from math import sqrt
from typing import Any

import torch
from torch import Tensor, nn
from torch.optim import Optimizer

__all__ = ["Resize"]


def growth(size: int, previous: int) -> None:
    if isinstance(size, bool) or not isinstance(size, int) or size < previous:
        raise ValueError(f"row size must be an integer >= {previous}, got {size!r}; shrinking is unsupported")


def mapping(old: Tensor, values: Tensor, indices: Tensor | Sequence[int] | None, address: str) -> Tensor:
    """Validate a row replacement and its injective old-to-new row mapping."""
    if old.layout != torch.strided or values.layout != torch.strided or old.ndim == 0 or values.ndim == 0:
        raise ValueError(f"{address} requires dense tensors with a row axis, got {old.layout} and {values.layout}")
    if values.shape[1:] != old.shape[1:] or values.shape[0] < old.shape[0]:
        raise ValueError(f"{address} must grow only its row axis from {tuple(old.shape)}, got {tuple(values.shape)}")
    if values.dtype != old.dtype or values.device != old.device:
        raise ValueError(
            f"{address} must retain dtype {old.dtype} and device {old.device}, got {values.dtype} on {values.device}"
        )
    if old.device.type == "meta":
        raise ValueError(f"{address} requires materialized rows; move the tensor off the meta device before resizing")

    if indices is None:
        return torch.arange(old.shape[0], device=old.device)
    destinations = torch.as_tensor(indices, device=old.device)
    if destinations.ndim != 1 or destinations.numel() != old.shape[0]:
        raise ValueError(
            f"{address} requires one destination per old row ({old.shape[0]}), got {tuple(destinations.shape)}"
        )
    if destinations.numel() and destinations.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{address} row destinations must be integers, got {destinations.dtype}")
    destinations = destinations.to(dtype=torch.int64).clone()
    if destinations.numel() and (destinations.min() < 0 or destinations.max() >= values.shape[0]):
        raise ValueError(f"{address} row destinations must be in [0, {values.shape[0]}), got {destinations.tolist()}")
    if destinations.unique().numel() != destinations.numel():
        raise ValueError(f"{address} row destinations must be unique, got {destinations.tolist()}")
    return destinations


def rows(values: Tensor, size: int, indices: Tensor) -> Tensor:
    """Remap dense or sparse COO history, leaving newly introduced rows zero."""
    shape = (size, *values.shape[1:])
    indices = indices.to(device=values.device)
    if values.layout == torch.strided:
        return values.new_zeros(shape).index_copy_(0, indices, values)
    if values.layout == torch.sparse_coo and values.sparse_dim() > 0:
        coordinates = values._indices().clone()
        coordinates[0] = indices[coordinates[0]]
        return torch.sparse_coo_tensor(
            coordinates, values._values().clone(), shape, device=values.device, check_invariants=True
        )
    raise ValueError(f"row history requires dense or sparse COO tensors, got {values.layout}")


class Resize:
    """Accumulate replacements, then validate and commit them between training steps.

    ``parameters`` exposes staged ``(old, new)`` pairs and remains available after
    commit. Parameter values are copied when staged; buffer values are prepared
    by their owner. Gradients and optimizer state are read at commit so completed
    backward calls or caller-owned gradient synchronization can still accumulate,
    replace, or clear gradients in the meantime. Do not run an optimizer step
    between staging and commit, or retain a forward graph across commit.

    Adam/AdamW moments (including AMSGrad), SGD momentum, and scalar steps are
    supported. Unknown state entries fail before any owner or optimizer changes.
    Optimizer groups and scheduler objects stay in place. A successful plan is
    single-use; repeating ``commit`` is harmless.
    """

    def __init__(self, optimizers: Sequence[Optimizer] = ()) -> None:
        self.optimizers = tuple(dict.fromkeys(optimizers))
        self.parameters: list[tuple[nn.Parameter, nn.Parameter]] = []
        self._changes: list[tuple[object, str, object, object]] = []
        self._indices: dict[nn.Parameter, Tensor] = {}
        self._committed = False
        self._publications: list[Callable[[], object]] = []

    def publish(self, action: Callable[[], object]) -> None:
        """Publish extension metadata only after tensor and optimizer validation."""
        if self._committed:
            raise RuntimeError("cannot publish metadata after commit; create a new Resize plan")
        self._publications.append(action)

    def stage(self, owner: object, name: str, value: object) -> None:
        """Each existing owner attribute may be replaced once in a plan."""
        address = f"{type(owner).__name__}.{name}"
        if self._committed:
            raise RuntimeError(f"{address} cannot be staged after commit; create a new Resize plan")
        if any(target is owner and field == name for target, field, _, _ in self._changes):
            raise ValueError(f"{address} already has a staged replacement")
        self._changes.append((owner, name, getattr(owner, name), value))

    def parameter(
        self,
        owner: nn.Module,
        name: str,
        values: Tensor,
        indices: Tensor | Sequence[int] | None = None,
    ) -> nn.Parameter:
        """Stage initialized rows, copying old rows to their mapped destinations."""
        old = owner._parameters.get(name)
        address = f"{type(owner).__name__}.{name}"
        if not isinstance(old, nn.Parameter):
            raise ValueError(f"{address} must be a registered parameter, got {type(old).__name__}")
        if old in self._indices:
            raise ValueError(f"{address} parameter already has a staged replacement")
        destinations = mapping(old, values, indices, address)
        with torch.no_grad():
            data = values.detach().clone().index_copy_(0, destinations, old)
        new = nn.Parameter(data, requires_grad=old.requires_grad)
        self.stage(owner, name, new)
        self.parameters.append((old, new))
        self._indices[old] = destinations
        return new

    def buffer(self, owner: nn.Module, name: str, values: Tensor) -> None:
        """Publish caller-prepared values at commit, retaining buffer persistence.

        The owner defines initialization and row mapping for its buffer; unlike
        ``parameter``, this operation does not copy or rearrange old values.
        """
        old = owner._buffers.get(name)
        address = f"{type(owner).__name__}.{name}"
        if not isinstance(old, Tensor):
            raise ValueError(f"{address} must be a registered tensor buffer, got {type(old).__name__}")
        if not isinstance(values, Tensor):
            raise ValueError(f"{address} replacement must be a tensor, got {type(values).__name__}")
        self.stage(owner, name, values)

    def attribute(self, owner: object, name: str, value: object) -> None:
        """Stage existing metadata alongside the tensor replacements it describes."""
        if isinstance(owner, nn.Module) and (name in owner._parameters or name in owner._buffers):
            raise ValueError(f"{type(owner).__name__}.{name} is a registered tensor; use parameter() or buffer()")
        self.stage(owner, name, value)

    def embedding(self, module: nn.Embedding, size: int, *, tail: bool = False) -> None:
        """Grow embedding rows, optionally moving the old last row to the new last."""
        previous = module.num_embeddings
        growth(size, previous)
        if size == previous:
            return
        indices = torch.arange(previous, device=module.weight.device)
        if tail and previous:
            indices[-1] = size - 1
        values = module.weight.new_empty((size, module.embedding_dim))
        nn.init.normal_(values)
        self.parameter(module, "weight", values, indices)
        self.attribute(module, "num_embeddings", size)
        if tail and module.padding_idx == previous - 1:
            self.attribute(module, "padding_idx", size - 1)

    def linear(self, module: nn.Linear, size: int) -> None:
        """Grow output rows with PyTorch's ordinary fan-in initialization."""
        growth(size, module.out_features)
        if size == module.out_features:
            return
        values = module.weight.new_empty((size, module.in_features))
        nn.init.kaiming_uniform_(values, a=sqrt(5))
        self.parameter(module, "weight", values)
        if module.bias is not None:
            bias = module.bias.new_empty(size)
            bound = 1 / sqrt(module.in_features) if module.in_features else 0
            nn.init.uniform_(bias, -bound, bound)
            self.parameter(module, "bias", bias)
        self.attribute(module, "out_features", size)

    @torch.no_grad()
    def commit(self) -> None:
        """Prepare all gradient/state migrations before applying any replacement."""
        if self._committed:
            return
        for owner, name, old, _ in self._changes:
            if getattr(owner, name) is not old:
                raise ValueError(f"{type(owner).__name__}.{name} changed after staging; create a new Resize plan")

        gradients: list[tuple[nn.Parameter, Tensor | None]] = []
        states: list[tuple[Optimizer, nn.Parameter, nn.Parameter, dict[str, Any]]] = []
        moments = {"exp_avg", "exp_avg_sq", "max_exp_avg_sq", "momentum_buffer"}
        for old, new in self.parameters:
            indices = self._indices[old]
            gradients.append((new, None if old.grad is None else rows(old.grad, new.shape[0], indices)))
            for optimizer in self.optimizers:
                if old not in optimizer.state:
                    continue
                migrated: dict[str, Any] = {}
                for name, value in optimizer.state[old].items():
                    address = f"{type(optimizer).__name__} state {name!r} for parameter {tuple(old.shape)}"
                    if name == "step":
                        if isinstance(value, (int, float)) or (
                            isinstance(value, Tensor) and value.layout == torch.strided and value.numel() == 1
                        ):
                            migrated[name] = value
                            continue
                        raise ValueError(f"{address} must be scalar, got {value!r}")
                    if name not in moments:
                        raise ValueError(
                            f"{address} is unsupported; expected Adam/AdamW moments, SGD momentum, or step"
                        )
                    if not isinstance(value, Tensor) or value.shape != old.shape:
                        actual = tuple(value.shape) if isinstance(value, Tensor) else type(value).__name__
                        raise ValueError(f"{address} must have shape {tuple(old.shape)}, got {actual}")
                    migrated[name] = rows(value, new.shape[0], indices)
                states.append((optimizer, old, new, migrated))

        replacements = dict(self.parameters)
        for new, gradient in gradients:
            new.grad = gradient
        for owner, name, _, value in self._changes:
            setattr(owner, name, value)
        for optimizer in self.optimizers:
            for group in optimizer.param_groups:
                group["params"][:] = [replacements.get(parameter, parameter) for parameter in group["params"]]
        for optimizer, old, new, state in states:
            del optimizer.state[old]
            optimizer.state[new] = state
        for action in self._publications:
            action()
        self._committed = True
