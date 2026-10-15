"""Generic runtime contracts for model forward inputs."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, cast

import pydantic
import torch
from tensordict import TensorDict

from relflow.data.arrow import Geometry
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.tree import Address
from relflow.tensorfields.base import TENSORFIELDS, TensorFieldBase

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class ForwardContractError(ValueError):
    """Raised when a forward batch violates a model input contract."""


INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


ContractSignature = tuple[object, ...]
ContractScope = tuple[int, str, int, int, ContractSignature]
CONTRACT_SCOPES = 128


class ContractScheduler(pydantic.BaseModel):
    """Deterministic backoff scheduler for expensive forward contract checks."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    periodic_interval: int = 1024
    _counts: OrderedDict[ContractScope, int] = pydantic.PrivateAttr(default_factory=OrderedDict)

    def reset(self) -> None:
        self._counts.clear()

    def should_check(
        self,
        module: "Model",
        inputs: object,
        *,
        strata: Strata,
        dataloader_idx: int,
    ) -> bool:
        generation = int(getattr(module, "_contract_generation", 0))
        scope = (
            int(getattr(module, "global_rank", 0)),
            str(strata),
            dataloader_idx,
            generation,
            batch_signature(module, inputs),
        )
        count = self._counts.get(scope, 0)
        self._counts[scope] = count + 1
        self._counts.move_to_end(scope)
        if len(self._counts) > CONTRACT_SCOPES:
            self._counts.popitem(last=False)
        return is_backoff_index(count, periodic_interval=self.periodic_interval)


def sanitize(
    module: "Model",
    inputs: TensorDict,
    *,
    strata: Strata | str,
    dataloader_idx: int = 0,
) -> None:
    """Validate the generic forward-input contract before model execution."""
    normalized = Strata.normalize(strata)
    if not isinstance(inputs, TensorDict):
        raise TypeError(f"forward inputs must be a TensorDict, got {type(inputs).__name__}")

    require_forward_addresses(module, inputs, strata=normalized)
    for address in module.schema.active_requests:
        require_registered_tensorfield(module, address, inputs[address])
    shapes = require_geometry(module, inputs)
    for address in module.schema.active_requests:
        tensorfield = cast(TensorFieldBase, inputs[address])
        require_core_tensors(module, address, tensorfield, shape=shapes[address])
        require_tensor_devices(module, address, tensorfield)

    scheduler = getattr(module, "_contract_scheduler", None)
    if isinstance(scheduler, ContractScheduler) and not scheduler.should_check(
        module,
        inputs,
        strata=normalized,
        dataloader_idx=dataloader_idx,
    ):
        return

    require_geometry_values(module, inputs)
    for address in module.schema.active_requests:
        tensorfield = cast(TensorFieldBase, inputs[address])
        require_token_values(address, TensorKey.state, tensorfield.state)
        if TensorKey.state in tensorfield.targets:
            require_token_values(
                address,
                f"{TensorKey.targets}[{TensorKey.state}]",
                cast(torch.Tensor, tensorfield.targets[TensorKey.state]),
            )
        require_objective_contract(module, address, tensorfield)
        require_routing_contract(module, address, tensorfield, strata=normalized)


def is_backoff_index(index: int, *, periodic_interval: int) -> bool:
    if index == 0:
        return True

    if (index & (index - 1)) == 0:
        return True

    return periodic_interval > 0 and index % periodic_interval == 0


def batch_signature(module: "Model", inputs: object) -> ContractSignature:
    if not isinstance(inputs, TensorDict):
        return ("inputs", qualified_name(type(inputs)))

    input_keys = tuple(sorted(str(key) for key in inputs.keys()))
    fields: list[tuple[object, ...]] = []
    for address in sorted(module.schema.active_requests, key=str):
        if address not in inputs.keys():
            fields.append((str(address), "missing"))
            continue

        tensorfield = inputs[address]
        fields.append(
            (
                str(address),
                qualified_name(type(tensorfield)),
                tensor_signature(getattr(tensorfield, TensorKey.state, None)),
                tensor_signature(getattr(tensorfield, TensorKey.present, None)),
                tensor_signature(getattr(tensorfield, TensorKey.trainable, None)),
                tensor_signature(getattr(tensorfield, TensorKey.inferred, None)),
                tensor_tree_signature(getattr(tensorfield, TensorKey.content, None)),
                tensor_tree_signature(getattr(tensorfield, TensorKey.targets, None)),
            )
        )

    return (input_keys, tuple(fields))


def tensor_signature(value: object) -> tuple[object, ...]:
    if not torch.is_tensor(value):
        return ("object", qualified_name(type(value)))

    return (
        "tensor",
        value.ndim,
        str(value.dtype),
        str(value.device),
    )


def tensor_tree_signature(value: object) -> tuple[object, ...]:
    if torch.is_tensor(value):
        return tensor_signature(value)

    if isinstance(value, TensorDict):
        return (
            "tensordict",
            tuple((str(key), tensor_tree_signature(value[key])) for key in sorted(value.keys(), key=str)),
        )

    if isinstance(value, Mapping):
        return (
            "mapping",
            tuple(
                (str(key), tensor_tree_signature(item)) for key, item in sorted(value.items(), key=lambda x: str(x[0]))
            ),
        )

    return ("object", qualified_name(type(value)))


def require_forward_addresses(
    module: "Model",
    inputs: TensorDict,
    *,
    strata: Strata,
) -> None:
    addresses = {Address(str(key)) for key in inputs.keys()}
    expected = set(module.schema.active_requests)
    if isinstance(inputs.get(Address()), TensorDict):
        expected.add(Address())

    missing = expected - addresses
    if missing:
        raise ForwardContractError(f"forward input is missing active request address(es): {format_addresses(missing)}")

    extra = addresses - expected
    if not extra:
        return

    branches = extra & set(module.schema.branches)
    if branches:
        raise ForwardContractError(
            f"forward input contains branch address(es); only active leaf request addresses are allowed: "
            f"{format_addresses(branches)}"
        )

    inactive = {address for address in extra if address in module.schema.requests}
    if inactive:
        raise ForwardContractError(
            "forward input contains inactive request address(es): "
            f"{format_addresses(inactive)}. Inactive fields remain in the schema but must not be present in runtime input."
        )

    raise ForwardContractError(f"forward input contains unknown address(es): {format_addresses(extra)}")


def require_registered_tensorfield(module: "Model", address: Address, value: object) -> None:
    if not isinstance(value, TensorFieldBase):
        raise TypeError(f"forward input '{address}' must be a TensorFieldBase, got {type(value).__name__}")

    request = module.schema.requests[address]
    expected = TENSORFIELDS[request.type].TensorField
    if not isinstance(value, expected):
        raise TypeError(
            f"forward input '{address}' must use tensorfield class {qualified_name(expected)}, "
            f"got {qualified_name(type(value))}"
        )


def require_geometry(module: "Model", inputs: TensorDict) -> dict[Address, tuple[int, ...]]:
    """Validate shared branch extents without reading tensor values on the device."""

    metadata = inputs.get(Address())
    paths = {
        address: tuple(node.address for node in request.path if node.type == "branch")
        for address, request in module.schema.active_requests.items()
    }
    if metadata is None:
        if any(size is None for address in paths for size in module.schema.shapes[address]):
            raise ForwardContractError("forward inputs with automatic branch lengths require branch geometry at '/'")
        first = next(iter(paths), None)
        state = (
            None
            if first is None
            else require_tensor_attribute(first, cast(TensorFieldBase, inputs[first]), TensorKey.state)
        )
        batch = 0 if state is None or not state.ndim else state.shape[0]
        return {address: (batch, *cast(tuple[int, ...], module.schema.shapes[address])) for address in paths}

    if not isinstance(metadata, TensorDict):
        raise TypeError(f"forward branch geometry at '/' must be a TensorDict, got {type(metadata).__name__}")
    expected = {Address(), *(parent for path in paths.values() for parent in path)}
    actual = {Address(str(address)) for address in metadata.keys()}
    if actual != expected:
        raise ForwardContractError(
            f"forward branch geometry addresses must match active branches: "
            f"expected {format_addresses(expected)}, got {format_addresses(actual)}"
        )

    extents: dict[Address, int] = {}
    batch = 0
    for address, branch in module.schema.branches.items():
        if address not in expected:
            continue
        geometry = metadata[address]
        if not isinstance(geometry, Geometry):
            raise TypeError(f"forward branch geometry '{address}' must be a Geometry, got {type(geometry).__name__}")
        tensors = {name: getattr(geometry, name, None) for name in ("lengths", "present", "valid", "slots")}
        devices: set[torch.device] = set()
        for name, tensor in tensors.items():
            if not isinstance(tensor, torch.Tensor):
                raise TypeError(f"forward branch geometry '{address}' {name} must be a torch.Tensor")
            dtype = torch.int64 if name == "lengths" else torch.bool
            if tensor.dtype != dtype:
                raise TypeError(f"forward branch geometry '{address}' {name} must use {dtype}, got {tensor.dtype}")
            devices.add(tensor.device)
        if address == Address():
            if geometry.lengths.ndim != 1:
                raise ForwardContractError("forward root geometry lengths must have rank one")
            batch = geometry.lengths.shape[0]
        parents = tuple(node.address for node in branch.path if node.type == "branch" and node is not branch)
        parent_shape = (batch, *(extents[parent] for parent in parents))
        for name in ("lengths", "present", "valid"):
            tensor = getattr(geometry, name)
            if tuple(tensor.shape) != parent_shape:
                raise ForwardContractError(
                    f"forward branch geometry '{address}' {name} must have shape {parent_shape}, "
                    f"got {tuple(tensor.shape)}"
                )
        if geometry.slots.ndim != len(parent_shape) + 1 or tuple(geometry.slots.shape[:-1]) != parent_shape:
            raise ForwardContractError(
                f"forward branch geometry '{address}' slots must start with parent shape {parent_shape}, "
                f"got {tuple(geometry.slots.shape)}"
            )
        extent = geometry.slots.shape[-1]
        if (address == Address() and extent != 1) or (branch.length is not None and extent > branch.length):
            raise ForwardContractError(
                f"forward branch geometry '{address}' extent must be "
                f"{'one' if address == Address() else f'at most {branch.length}'}, got {extent}"
            )
        module_device = getattr(module, "device", None)
        if len(devices) != 1 or (
            isinstance(module_device, torch.device) and not same_device(next(iter(devices)), module_device)
        ):
            raise ForwardContractError(
                f"forward branch geometry '{address}' tensors must share module device {module_device}, "
                f"got {', '.join(sorted(map(str, devices)))}"
            )
        extents[address] = extent
    return {address: (batch, *(extents[parent] for parent in path)) for address, path in paths.items()}


def require_geometry_values(module: "Model", inputs: TensorDict) -> None:
    """Check collection validity and offsets on the expensive validation schedule."""

    metadata = inputs.get(Address())
    if not isinstance(metadata, TensorDict):
        return
    for address in metadata.keys():
        geometry = cast(Geometry, metadata[address])
        if (geometry.lengths < 0).any() or (geometry.lengths > geometry.slots.shape[-1]).any():
            raise ForwardContractError(f"forward branch geometry '{address}' lengths must be within its extent")
        if (geometry.valid & ~geometry.present).any() or (geometry.lengths.masked_select(~geometry.valid) != 0).any():
            raise ForwardContractError(
                f"forward branch geometry '{address}' validity must agree with collection lengths"
            )
        positions = torch.arange(geometry.slots.shape[-1], device=geometry.slots.device)
        if (geometry.slots & (positions >= geometry.lengths.unsqueeze(-1))).any():
            raise ForwardContractError(f"forward branch geometry '{address}' slots must lie within collection lengths")
        if address == Address():
            if not geometry.present.all() or not geometry.valid.all() or (geometry.lengths != 1).any():
                raise ForwardContractError("forward root geometry must preserve one collection slot per batch row")
        else:
            parent = cast(Geometry, metadata[module.schema.branches[Address(address)].parent.address])
            if (geometry.present & ~parent.slots).any():
                raise ForwardContractError(f"forward branch geometry '{address}' cannot exist outside its parent slots")


def require_core_tensors(
    module: "Model", address: Address, tensorfield: TensorFieldBase, *, shape: tuple[int, ...]
) -> None:
    state = require_tensor_attribute(address, tensorfield, TensorKey.state)
    present = require_tensor_attribute(address, tensorfield, TensorKey.present)
    trainable = require_tensor_attribute(address, tensorfield, TensorKey.trainable)
    inferred = require_tensor_attribute(address, tensorfield, TensorKey.inferred)
    content = require_tensor_tree(
        address,
        TensorKey.content,
        getattr(tensorfield, TensorKey.content, None),
    )
    targets = require_targets(address, tensorfield)

    field_shape = module.schema.shapes[address]
    if state.ndim != len(field_shape) + 1:
        raise ForwardContractError(
            f"forward input '{address}' state must have rank {len(field_shape) + 1}, got {state.ndim}"
        )

    if tuple(state.shape) != shape:
        raise ForwardContractError(f"forward input '{address}' state must have shape {shape}, got {tuple(state.shape)}")

    if state.dtype not in INTEGER_DTYPES:
        raise TypeError(f"forward input '{address}' state must use an integer dtype, got {state.dtype}")

    for name, values in (
        (TensorKey.present, present),
        (TensorKey.trainable, trainable),
        (TensorKey.inferred, inferred),
    ):
        if tuple(values.shape) != tuple(state.shape):
            raise ForwardContractError(
                f"forward input '{address}' {name} must have shape {tuple(state.shape)}, got {tuple(values.shape)}"
            )
        if values.dtype != torch.bool:
            raise TypeError(f"forward input '{address}' {name} must use bool dtype, got {values.dtype}")

    require_content_prefix_shapes(address, content, state)

    if TensorKey.state in targets.keys():
        target_state_name = f"{TensorKey.targets}[{TensorKey.state}]"
        target_state = require_tensor_tree(address, target_state_name, targets[TensorKey.state])
        require_matching_tree_shapes(
            address,
            actual_name=target_state_name,
            actual=target_state,
            expected_name=TensorKey.state,
            expected={(): state},
        )
        require_integer_tensors(address, target_state_name, target_state)

    if TensorKey.content in targets.keys():
        target_content_name = f"{TensorKey.targets}[{TensorKey.content}]"
        target_content = require_tensor_tree(address, target_content_name, targets[TensorKey.content])
        require_matching_tree_shapes(
            address,
            actual_name=target_content_name,
            actual=target_content,
            expected_name=TensorKey.content,
            expected=content,
        )


def require_tensor_devices(module: "Model", address: Address, tensorfield: TensorFieldBase) -> None:
    tensors = list(iter_tensor_leaves(tensorfield))
    devices = {tensor.device for _, tensor in tensors}
    if len(devices) > 1:
        formatted = ", ".join(sorted(str(device) for device in devices))
        raise ForwardContractError(f"forward input '{address}' tensors must share one device, got {formatted}")

    module_device = getattr(module, "device", None)
    if isinstance(module_device, torch.device) and devices and not same_device(next(iter(devices)), module_device):
        raise ForwardContractError(
            f"forward input '{address}' tensors must be on module device {module_device}, got {next(iter(devices))}"
        )


def canonical_device(device: torch.device) -> tuple[str, int | None]:
    if device.type == "mps":
        return (device.type, 0 if device.index is None else device.index)

    if device.type == "cuda":
        index = device.index
        if index is None and torch.cuda.is_available():
            index = torch.cuda.current_device()
        return (device.type, index)

    return (device.type, device.index)


def same_device(left: torch.device, right: torch.device) -> bool:
    return canonical_device(left) == canonical_device(right)


def require_routing_contract(
    module: "Model",
    address: Address,
    tensorfield: TensorFieldBase,
    *,
    strata: Strata,
) -> None:
    state = tensorfield.state
    present = tensorfield.present
    trainable = tensorfield.trainable
    inferred = tensorfield.inferred
    padded = state.eq(Tokens.padded.value)

    if not torch.equal(present, ~padded):
        raise ForwardContractError(f"forward input '{address}' present must be true exactly where state is not padded")

    if (trainable & inferred).any():
        raise ForwardContractError(f"forward input '{address}' cannot be trainable and inferred at one coordinate")

    if strata == Strata.predict and trainable.any():
        raise ForwardContractError(f"forward input '{address}' cannot be trainable during prediction")
    if strata != Strata.predict and inferred.any():
        raise ForwardContractError(f"forward input '{address}' cannot be inferred during {strata}")

    if not trainable.any():
        return

    targets = tensorfield.targets
    for key in (TensorKey.state, TensorKey.content):
        if key not in targets.keys():
            raise ForwardContractError(f"forward input '{address}' has trainable positions but lacks targets[{key}]")

    target_state = cast(torch.Tensor, targets[TensorKey.state])
    unavailable = target_state.masked_select(trainable)
    if unavailable.eq(Tokens.masked.value).any() or unavailable.eq(Tokens.padded.value).any():
        raise ForwardContractError(
            f"forward input '{address}' targets[{TensorKey.state}] must contain original trainable state"
        )


def require_objective_contract(
    module: "Model",
    address: Address,
    tensorfield: TensorFieldBase,
) -> None:
    objective = address in module.schema.objectives
    if tensorfield.trainable.any() and not objective:
        raise ForwardContractError(f"forward input '{address}' is trainable but is not a reconstruction objective")
    if tensorfield.inferred.any() and address not in module.schema.decodes:
        raise ForwardContractError(f"forward input '{address}' is inferred but is not a prediction decode")


def require_tensor_attribute(address: Address, tensorfield: TensorFieldBase, name: str) -> torch.Tensor:
    value = getattr(tensorfield, name, None)
    if not torch.is_tensor(value):
        raise TypeError(f"forward input '{address}' {name} must be a torch.Tensor, got {type(value).__name__}")

    return value


def require_targets(address: Address, tensorfield: TensorFieldBase) -> TensorDict:
    value = getattr(tensorfield, TensorKey.targets, None)
    if not isinstance(value, TensorDict):
        raise TypeError(
            f"forward input '{address}' {TensorKey.targets} must be a TensorDict, got {type(value).__name__}"
        )

    require_tensor_tree(address, TensorKey.targets, value, allow_empty=True)
    return value


def require_tensor_tree(
    address: Address,
    name: str,
    value: object,
    *,
    allow_empty: bool = False,
) -> dict[tuple[str, ...], torch.Tensor]:
    tensors = dict(iter_tensor_leaves(value))
    if not tensors and not allow_empty:
        raise TypeError(f"forward input '{address}' {name} must contain at least one tensor")

    return tensors


def require_matching_tree_shapes(
    address: Address,
    *,
    actual_name: str,
    actual: dict[tuple[str, ...], torch.Tensor],
    expected_name: str,
    expected: dict[tuple[str, ...], torch.Tensor],
) -> None:
    if set(actual) != set(expected):
        raise ForwardContractError(
            f"forward input '{address}' {actual_name} keys must match {expected_name} keys: "
            f"expected {format_paths(expected)}, got {format_paths(actual)}"
        )

    for path, actual_tensor in actual.items():
        expected_tensor = expected[path]
        if tuple(actual_tensor.shape) != tuple(expected_tensor.shape):
            suffix = format_path(path)
            raise ForwardContractError(
                f"forward input '{address}' {actual_name}{suffix} must have shape "
                f"{tuple(expected_tensor.shape)}, got {tuple(actual_tensor.shape)}"
            )


def require_content_prefix_shapes(
    address: Address,
    content: dict[tuple[str, ...], torch.Tensor],
    state: torch.Tensor,
) -> None:
    state_shape = tuple(state.shape)
    state_rank = len(state_shape)
    for path, tensor in content.items():
        if len(tensor.shape) < state_rank or tuple(tensor.shape[:state_rank]) != state_shape:
            suffix = format_path(path)
            raise ForwardContractError(
                f"forward input '{address}' {TensorKey.content}{suffix} must start with {TensorKey.state} "
                f"shape {state_shape}, "
                f"got {tuple(tensor.shape)}"
            )


def require_integer_tensors(
    address: Address,
    name: str,
    tensors: dict[tuple[str, ...], torch.Tensor],
) -> None:
    for path, tensor in tensors.items():
        if tensor.dtype not in INTEGER_DTYPES:
            suffix = format_path(path)
            raise TypeError(f"forward input '{address}' {name}{suffix} must use an integer dtype, got {tensor.dtype}")


def require_token_values(address: Address, name: str, values: torch.Tensor) -> None:
    valid = torch.tensor([token.value for token in Tokens], device=values.device, dtype=values.dtype)
    invalid = ~torch.isin(values, valid)
    if invalid.any():
        value = values.masked_select(invalid).reshape(-1)[0].item()
        raise ForwardContractError(f"forward input '{address}' {name} contains invalid token id {value}")


def iter_tensor_leaves(value: object, path: tuple[str, ...] = ()) -> Iterator[tuple[tuple[str, ...], torch.Tensor]]:
    if torch.is_tensor(value):
        yield path, value
        return

    if isinstance(value, TensorFieldBase):
        for name in (
            TensorKey.state,
            TensorKey.present,
            TensorKey.trainable,
            TensorKey.inferred,
            TensorKey.content,
            TensorKey.targets,
        ):
            yield from iter_tensor_leaves(getattr(value, name, None), (*path, name))
        return

    if isinstance(value, TensorDict):
        for key in value.keys():
            yield from iter_tensor_leaves(value[key], (*path, str(key)))
        return

    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from iter_tensor_leaves(item, (*path, str(key)))
        return

    raise TypeError(f"expected tensor tree at {format_path(path) or '<root>'}, got {type(value).__name__}")


def format_addresses(addresses: set[Address]) -> str:
    return ", ".join(sorted(str(address) for address in addresses))


def format_paths(values: Mapping[tuple[str, ...], object]) -> str:
    return ", ".join(format_path(path) or "<tensor>" for path in sorted(values))


def format_path(path: tuple[str, ...]) -> str:
    return "".join(f"[{part}]" for part in path)


def qualified_name(cls: type[object]) -> str:
    return f"{cls.__module__}.{cls.__qualname__}"
