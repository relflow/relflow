"""Transient compilation of the model's prepared tensor regions."""

from collections.abc import Callable
from copy import deepcopy
from types import FunctionType, MethodType
from typing import TYPE_CHECKING, Any

import torch

from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.packed import customized
from relflow.architecture.pool import CrossAttentionBlock, LearnedQueryCrossAttention

if TYPE_CHECKING:
    from relflow.architecture.root import Model

__all__ = ["clear", "compile"]


class Region:
    """Bind one compiler cache to one compute method, without serializing it."""

    def __init__(
        self,
        eager: MethodType,
        local: bool,
        *,
        retain_graph: bool,
        backend: str | Callable,
        dynamic: bool | None,
        options: dict[str, Any],
    ) -> None:
        self.eager = eager
        self.local = local
        self.retain_graph = retain_graph
        original = eager.__func__
        # Dynamo caches by code object. Independent regions must not exhaust a
        # shared method's specialization limit as the number of nodes grows.
        function = FunctionType(
            original.__code__.replace(),
            original.__globals__,
            original.__name__,
            original.__defaults__,
            original.__closure__,
        )
        function.__kwdefaults__ = original.__kwdefaults__
        function.__annotations__ = original.__annotations__
        function.__dict__.update(original.__dict__)
        function.__module__ = original.__module__
        function.__qualname__ = original.__qualname__
        self.compiled = torch.compile(
            MethodType(function, eager.__self__),
            backend=backend,
            fullgraph=True,
            dynamic=dynamic,
            options=options or None,
        )

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        owner = self.eager.__self__
        layers = (
            owner.encoder
            if isinstance(owner, BranchEncoder)
            else [*owner.blocks, owner.norm, *([owner.mass_projection] if owner.mass_projection is not None else [])]
        )
        # Torch skips guards for hooks by default. Check outside the graph so
        # callbacks added after warmup and custom replacements still run eager.
        if customized(layers, additional=(CrossAttentionBlock,)):
            return self.eager(*args, **kwargs)
        if self.retain_graph:
            from torch._functorch import config

            # AOTAutograd must preserve saved tensors for objective probes.
            # Scope this around lazy tracing; ordinary models keep defaults.
            with config.patch(donated_buffer=False, backward_pass_autocast="off"):
                return self.compiled(*args, **kwargs)
        return self.compiled(*args, **kwargs)

    def __deepcopy__(self, memo: dict[int, Any]) -> MethodType:
        return deepcopy(self.eager, memo)

    def __reduce__(self):
        return self.eager.__reduce__()


def clear(model: torch.nn.Module) -> None:
    """Restore eager methods still owned by RelFlow's compiler bindings."""
    for module in model.modules():
        region = module.__dict__.get("compute")
        if isinstance(region, Region):
            if region.local:
                module.compute = region.eager
            else:
                del module.compute


def compile(
    model: "Model",
    *,
    encoders: bool,
    pools: bool,
    backend: str | Callable,
    dynamic: bool | None,
    options: dict[str, Any] | None,
) -> None:
    """Prepare all replacements before changing the model's active selection."""
    settings = {}
    if backend == "inductor":
        settings.update(
            {
                "fallback_random": True,
                "triton.cudagraphs": False,
                "shape_padding": False,
                "comprehensive_padding": False,
                "inplace_padding": False,
            }
        )
    settings.update(options or {})
    replacements = []
    for module in model.modules():
        if not (
            (encoders and type(module) is BranchEncoder and len(module.encoder))
            or (pools and type(module) is LearnedQueryCrossAttention)
        ):
            continue
        current = module.compute
        eager = current.eager if isinstance(current, Region) else current
        # Custom compute implementations retain their own execution contract.
        if (
            not isinstance(eager, MethodType)
            or eager.__self__ is not module
            or eager.__func__ is not type(module).compute
        ):
            continue
        local = current.local if isinstance(current, Region) else "compute" in module.__dict__
        region = Region(
            eager, local, retain_graph=model.jacobian, backend=backend, dynamic=dynamic, options=settings.copy()
        )
        replacements.append((module, region))
    clear(model)
    for module, region in replacements:
        module.compute = region
