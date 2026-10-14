"""Model-facing schema mutation orchestration."""

from __future__ import annotations

from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from copy import deepcopy
from functools import partialmethod, wraps
from typing import TYPE_CHECKING, Any, cast

import lightning.pytorch as lit
import torch
from lightning.pytorch import Callback

from relflow.architecture.graph import ModelGraph
from relflow.logging import logger
from relflow.presets.base import subtree
from relflow.structs.enums import Strata
from relflow.structs.experiment import NodeAttribute, NodePredicate, TreeFieldInput, bind_fields
from relflow.structs.tree import Node

if TYPE_CHECKING:
    from relflow.architecture.root import Model


def immutable[F: Callable[..., object]](name: str | Strata) -> Callable[[F], F]:
    """Hold a model mutation lock without changing the decorated call signature."""

    def decorator(method: F) -> F:
        @wraps(method)
        def wrapped(self: Model, *args: object, **kwargs: object) -> object:
            locks = self.locks
            locks[name] += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                if locks[name] <= 1:
                    locks.pop(name, None)
                else:
                    locks[name] -= 1

        return cast(F, wrapped)

    return decorator


class MutationLockCallback(Callback):
    """Prevent runtime schema mutations while Lightning owns an active loop."""

    locks: tuple[Strata, ...] = (Strata.train, Strata.validate, Strata.test, Strata.predict)

    def on_loop_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        module = cast("Model", pl_module)
        if strata == Strata.predict:
            module.output_plans.clear()
        module.locks[strata] += 1

    def on_loop_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        locks = cast("Model", pl_module).locks
        if locks[strata] <= 1:
            locks.pop(strata, None)
        else:
            locks[strata] -= 1

    def on_exception(
        self,
        trainer: lit.Trainer,
        pl_module: lit.LightningModule,
        exception: BaseException,
    ) -> None:
        for lock in self.locks:
            cast("Model", pl_module).locks.pop(lock, None)

    # Preserve Lightning hook signatures for the runtime partialmethod descriptors.
    if TYPE_CHECKING:
        on_train_start = Callback.on_train_start
        on_train_end = Callback.on_train_end
        on_validation_start = Callback.on_validation_start
        on_validation_end = Callback.on_validation_end
        on_test_start = Callback.on_test_start
        on_test_end = Callback.on_test_end
        on_predict_start = Callback.on_predict_start
        on_predict_end = Callback.on_predict_end
    else:
        on_train_start = partialmethod(on_loop_start, strata=Strata.train)
        on_train_end = partialmethod(on_loop_end, strata=Strata.train)
        on_validation_start = partialmethod(on_loop_start, strata=Strata.validate)
        on_validation_end = partialmethod(on_loop_end, strata=Strata.validate)
        on_test_start = partialmethod(on_loop_start, strata=Strata.test)
        on_test_end = partialmethod(on_loop_end, strata=Strata.test)
        on_predict_start = partialmethod(on_loop_start, strata=Strata.predict)
        on_predict_end = partialmethod(on_loop_end, strata=Strata.predict)


class RuntimePlacementCallback(Callback):
    """Move late-created modules onto the Lightning module's active device."""

    def on_loop_start(self, trainer: lit.Trainer, pl_module: lit.LightningModule, strata: Strata) -> None:
        device = getattr(pl_module, "device", None)
        if isinstance(device, torch.device):
            pl_module.to(device=device)

    # Preserve Lightning hook signatures for the runtime partialmethod descriptors.
    if TYPE_CHECKING:
        on_train_start = Callback.on_train_start
        on_validation_start = Callback.on_validation_start
        on_test_start = Callback.on_test_start
        on_predict_start = Callback.on_predict_start
    else:
        on_train_start = partialmethod(on_loop_start, strata=Strata.train)
        on_validation_start = partialmethod(on_loop_start, strata=Strata.validate)
        on_test_start = partialmethod(on_loop_start, strata=Strata.test)
        on_predict_start = partialmethod(on_loop_start, strata=Strata.predict)


class SchemaEditor:
    """Coordinate schema mutations with runtime graph rebuilds."""

    def __init__(self, module: "Model") -> None:
        self.module = module

    @contextmanager
    def transaction(self) -> Generator[None, None, None]:
        """Restore the schema and runtime graph when a mutation cannot rebuild."""

        schema = deepcopy(self.module.schema)
        nodes = self.module.nodes
        example = self.module.example_input_array
        try:
            yield
        except Exception:
            self.module.schema = schema
            self.module.nodes = nodes
            self.module.example_input_array = example
            raise

    def assert_mutation_allowed(self, action: str) -> None:
        active = tuple(name for name, count in self.module.locks.items() if count > 0)
        if active:
            labels = ", ".join(active)
            raise RuntimeError(f"model.{action}(...) cannot run while the model is in an active loop: {labels}")

    def select(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
    ) -> list[Node]:
        return self.module.schema.select(
            *predicates,
            include_root=include_root,
        )

    def update(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> None:
        self.assert_mutation_allowed("update")
        with self.transaction():
            self.module.schema.update(
                *predicates,
                strict=strict,
                include_root=include_root,
                validate=validate,
                **values,
            )
            ModelGraph.rebuild(self.module)
        self.module.reset_contracts()
        logger.bind(component="schema_mutation", action="update").info("updated model schema")

    def extend(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        fields: Mapping[str, TreeFieldInput] | None = None,
        include_root: bool = True,
        **children: TreeFieldInput,
    ) -> None:
        self.assert_mutation_allowed("extend")
        if self.module.preset is not None:
            definitions = bind_fields(fields, children)
            fields = {
                cast(str, field.name): subtree(self.module.preset, cast(str, field.name), field)
                for field in definitions
            }
            children = {}
        with self.transaction():
            self.module.schema.extend(*predicates, fields=fields, include_root=include_root, **children)
            ModelGraph.rebuild(self.module)
        self.module.reset_contracts()
        logger.bind(component="schema_mutation", action="extend").info("extended model schema")

    def delete(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = False,
    ) -> None:
        self.assert_mutation_allowed("delete")
        with self.transaction():
            self.module.schema.delete(*predicates, include_root=include_root)
            ModelGraph.rebuild(self.module)
        self.module.reset_contracts()
        logger.bind(component="schema_mutation", action="delete").info("deleted model schema nodes")

    def reset(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        include_root: bool = True,
        descendants: bool = False,
    ) -> None:
        self.assert_mutation_allowed("reset")
        selected = self.module.schema.select(
            *predicates,
            include_root=include_root,
        )
        if not selected:
            raise ValueError("reset matched no nodes")

        ModelGraph.reset_selected(self.module, selected, descendants=descendants)
        self.module.reset_contracts()
        logger.bind(component="schema_mutation", action="reset").info("reset model nodes")

    @contextmanager
    def override(
        self,
        *predicates: NodePredicate | NodeAttribute | Callable[[Node], bool],
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: Any,
    ) -> Generator[None, None, None]:
        self.assert_mutation_allowed("override")
        entered = False
        try:
            with self.module.schema.override(
                *predicates,
                strict=strict,
                include_root=include_root,
                validate=validate,
                **values,
            ):
                entered = True
                ModelGraph.rebuild(self.module)
                self.module.reset_contracts()
                logger.bind(component="schema_mutation", action="override").info("overrode model schema")
                yield
        finally:
            ModelGraph.rebuild(self.module)
            self.module.reset_contracts()
            if entered:
                logger.bind(component="schema_mutation", action="override_restore").info("restored model schema")
