"""Runtime graph construction for schema-backed models."""

from __future__ import annotations

from copy import deepcopy
from typing import TYPE_CHECKING, TypedDict, cast

import torch

from relflow.architecture.node import NodeModule
from relflow.data.datasets.base import EncodedInput
from relflow.structs.enums import Strata
from relflow.structs.experiment import Schema
from relflow.structs.tree import Address, Node
from relflow.tensorfields.base import TENSORFIELDS

if TYPE_CHECKING:
    from relflow.architecture.root import Model


class ForwardInputs(TypedDict):
    """Keyword arguments used by Lightning to trace an example model forward."""

    inputs: EncodedInput
    strata: Strata


class ModelGraph:
    """Build and rebuild runtime modules from a schema tree."""

    @staticmethod
    def example_forward_kwargs(schema: Schema, batch_size: int) -> ForwardInputs:
        from relflow.data.iterables import mock

        return {
            "inputs": mock(schema=schema, batch_size=batch_size),
            "strata": Strata.predict,
        }

    @staticmethod
    def build(
        schema: Schema,
        batch_size: int,
    ) -> tuple[torch.nn.ModuleDict, ForwardInputs]:
        checked: set[str] = set()
        for address, request in schema.requests.items():
            if request.type in checked:
                continue
            TENSORFIELDS[request.type].require(address=address)
            checked.add(request.type)

        nodes = torch.nn.ModuleDict()

        for address in schema.requests | schema.branches:
            nodes[address] = NodeModule(
                schema=schema,
                address=address,
            )

        return nodes, ModelGraph.example_forward_kwargs(schema=schema, batch_size=batch_size)

    @staticmethod
    def install(module: "Model") -> None:
        module.nodes, example = ModelGraph.build(
            schema=module.schema,
            batch_size=module.batch_size,
        )
        # Lightning accepts a concrete dict but does not type its keyword fields.
        module.example_input_array = cast(dict[str, EncodedInput | Strata], example)

    @staticmethod
    def rebuild(module: "Model") -> None:
        module.schema.clear_tree_caches()
        was_training = module.training
        device = module.device
        previous = {
            name: value.detach().clone() if isinstance(value, torch.Tensor) else deepcopy(value)
            for name, value in module.state_dict().items()
        }
        ModelGraph.install(module)
        if isinstance(device, torch.device):
            module.to(device=device)
        current = module.state_dict()
        compatible = {}
        for name, value in previous.items():
            if name not in current:
                continue

            current_value = current[name]
            if isinstance(current_value, torch.Tensor) and isinstance(value, torch.Tensor):
                if current_value.shape != value.shape:
                    continue
            elif type(current_value) is not type(value):
                continue

            compatible[name] = value

        module.load_state_dict(compatible, strict=False)
        module.train(was_training)

    @staticmethod
    def reset_selected(module: "Model", selected: list[Node], *, descendants: bool = False) -> None:
        selected_by_address: dict[Address, Node] = {}
        for node in selected:
            if node.address in module.nodes:
                selected_by_address[Address(str(node.address))] = node

            if descendants:
                for descendant in getattr(node, "descendants", ()):
                    if descendant.address in module.nodes:
                        selected_by_address[Address(str(descendant.address))] = descendant

        if not selected_by_address:
            raise ValueError("reset matched no runtime nodes")

        for address in selected_by_address:
            module.nodes[address] = NodeModule(
                schema=module.schema,
                address=address,
            )

        module.example_input_array = cast(
            dict[str, EncodedInput | Strata],
            ModelGraph.example_forward_kwargs(schema=module.schema, batch_size=module.batch_size),
        )
        device = module.device
        if isinstance(device, torch.device):
            module.to(device=device)
