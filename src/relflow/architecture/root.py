"""Public Lightning model facade for `relflow` schemas."""

from collections import Counter
from collections.abc import Callable, Generator, Mapping
from contextlib import contextmanager
from functools import partialmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Required, Self, TypedDict, TypeVar, cast, overload

import lightning.pytorch as lit
import pyarrow as pa
import torch
from beartype import beartype
from lightning.pytorch import Callback
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from rich.console import Console, ConsoleOptions
from rich.text import Text
from torchmetrics import Metric as TorchMetric

from relflow._version import __version__
from relflow.architecture import compiler
from relflow.architecture.checkpoint import CheckpointState, RollbackCheckpoint
from relflow.architecture.contracts import ContractScheduler
from relflow.architecture.graph import ModelGraph
from relflow.architecture.mutations import (
    MutationLockCallback,
    RuntimePlacementCallback,
    SchemaEditor,
    immutable,
)
from relflow.architecture.node import NodeModule
from relflow.architecture.runtime import (
    ExecutionPlan,
    ModelRuntime,
    Output,
    OutputPlan,
    PredictionInput,
    Retain,
    step,
)
from relflow.data.arrow import Encoded
from relflow.data.datasets.base import EncodedInput, InterprocessEncodingContext
from relflow.data.processors import PostprocessorInput, PreprocessorInput
from relflow.logging import logger
from relflow.logging.throughput import ThroughputLogger
from relflow.structs.enums import AttentionInput, AttentionMode, Strata, StrataInput
from relflow.structs.experiment import (
    NodeSelector,
    Schema,
    TreeFieldInput,
)
from relflow.structs.packages import Prediction
from relflow.structs.reduction import Attention, ReductionConfig
from relflow.structs.tree import Address, Leaf, MaskInput, Node, Rate, Renderable
from relflow.tensorfields.base import TENSORFIELDS, Extension

OptimizerConfig = torch.optim.Optimizer | Callable[["Model"], torch.optim.Optimizer]


class Scheduler(Protocol):
    """Stateful scheduler attached to an optimizer.

    Torch schedulers implement this contract. For a custom scheduler, override
    ``Model.lr_scheduler_step`` to perform its update, or use manual optimization.
    """

    @property
    def optimizer(self) -> torch.optim.Optimizer: ...

    def state_dict(self) -> dict[str, Any]: ...
    def load_state_dict(self, state_dict: dict[str, Any]) -> None: ...


class SchedulerOptions(TypedDict, total=False):
    """Lightning scheduling controls; only ``scheduler`` is required."""

    scheduler: Required[Scheduler]
    name: str | None
    interval: Literal["step", "epoch"]
    frequency: int
    reduce_on_plateau: bool
    monitor: str | None
    strict: bool


SchedulerConfig = (
    Scheduler
    | SchedulerOptions
    | LRSchedulerConfigType
    | Callable[["Model", torch.optim.Optimizer], Scheduler | SchedulerOptions | LRSchedulerConfigType]
)
MetricValue = TypeVar("MetricValue", torch.Tensor, TorchMetric)
Pathname = TypeVar("Pathname", str, Path)

__all__ = [
    "Model",
    "MutationLockCallback",
    "RollbackCheckpoint",
    "RuntimePlacementCallback",
]

_DEFAULT_REDUCTION = Attention()


class Model(lit.LightningModule, Renderable):
    """Neural model generated from a `relflow` schema tree.

    `Model` owns the schema tree, tensorfield embedders, branch
    encoders, decoders, and convenience methods for prediction, checkpointing,
    schema display and mutation.

    Example:
        ```python
        import relflow as rf

        model = rf.Model(
            segment=rf.Category(size=32),
            label=rf.Category(mask=True, size=4),
            d_model=16,
            n_layers=1,
            n_heads=4,
            batch_size=8,
            embed=True,
        )
        ```
    """

    @overload
    def __init__(
        self,
        *,
        d_model: int,
        n_layers: int,
        n_heads: int,
        batch_size: int = 1,
        fields: Mapping[str, TreeFieldInput] | None = None,
        name: str = "record",
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionInput = AttentionMode.mha,
        reduction: ReductionConfig | None = _DEFAULT_REDUCTION,
        dropout: Rate | None = None,
        mask: MaskInput = False,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        **field_kwargs: TreeFieldInput,
    ) -> None: ...

    @overload
    def __init__(
        self,
        schema: Schema,
        *,
        batch_size: int = 1,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
    ) -> None: ...

    @beartype
    def __init__(
        self,
        schema: Schema | None = None,
        *,
        d_model: int | None = None,
        n_layers: int | None = None,
        n_heads: int | None = None,
        batch_size: int = 1,
        fields: Mapping[str, TreeFieldInput] | None = None,
        name: str = "record",
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionMode | str = AttentionMode.mha,
        reduction: ReductionConfig | None = _DEFAULT_REDUCTION,
        dropout: Rate | None = None,
        mask: MaskInput = False,
        optimizer: OptimizerConfig | None = None,
        scheduler: Any = None,  # Lightning validates custom schedulers at fit time.
        **field_kwargs: Any,
    ) -> None:
        """Build a model from tree fields, or from an existing ``schema``.

        Provide ``d_model``, ``n_layers``, and ``n_heads`` with tree fields.
        Keyword field names bind their data keys; field classes such as
        ``rf.Number`` are instantiated with defaults. Child ``rf.Branch``
        nodes describe repeated objects; the generated root is a singleton.

        ``mask=True`` makes a field a supervised target. ``embed=True`` emits
        its embedding during prediction. ``attention`` selects the root
        encoder mode and ``reduction`` selects its output representation.

        ``optimizer`` accepts a Torch optimizer or a factory receiving this
        model. ``scheduler`` accepts a Torch scheduler, a Lightning scheduler
        configuration dictionary, or a factory receiving model and optimizer.
        Custom schedulers also work with an overridden ``lr_scheduler_step``.
        Configure an optimizer before fitting; neither factory is checkpointed.

        A positional ``Schema`` or ``schema=...`` restores an existing
        architecture and cannot be combined with tree fields or dimensions.
        """
        if "n_linear" in field_kwargs and not (
            isinstance(field_kwargs["n_linear"], Node)
            or (isinstance(field_kwargs["n_linear"], type) and issubclass(field_kwargs["n_linear"], Leaf))
        ):
            raise ValueError("n_linear was removed from Model; use reduction=Attention(n_layers=...)")
        if "n_outputs" in field_kwargs and not (
            isinstance(field_kwargs["n_outputs"], Node)
            or (isinstance(field_kwargs["n_outputs"], type) and issubclass(field_kwargs["n_outputs"], Leaf))
        ):
            raise ValueError("n_outputs belongs to a reduction; use reduction=Attention(n_outputs=...)")

        if schema is not None:
            if fields is not None or field_kwargs or mask is not False:
                raise TypeError("schema cannot be combined with tree fields")
            if d_model is not None or n_layers is not None or n_heads is not None:
                raise TypeError("schema cannot be combined with d_model, n_layers, or n_heads")
            if reduction is not _DEFAULT_REDUCTION:
                raise TypeError("schema cannot be combined with a root reduction")
        else:
            required = {"d_model": d_model, "n_layers": n_layers, "n_heads": n_heads}
            missing = [key for key, value in required.items() if value is None]
            if missing:
                names = ", ".join(missing)
                raise TypeError(f"Model requires {names} when constructed from tree fields")

            schema = Schema.from_tree(
                d_model=cast(int, d_model),
                n_layers=cast(int, n_layers),
                n_heads=cast(int, n_heads),
                fields=fields,
                name=name,
                query=query,
                description=description,
                embed=embed,
                attention=cast(AttentionInput, attention),
                reduction=reduction,
                dropout=dropout,
                mask=mask,
                **field_kwargs,
            )

        super().__init__()
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")

        # Model provenance intentionally uses a string rather than nn.Module's
        # internal integer state-dict revision; preserve the existing artifact contract.
        self._version: str = __version__  # pyrefly: ignore[bad-override-mutable-attribute]
        self.schema: Schema = schema
        self.batch_size: int = batch_size
        self.optimizer: OptimizerConfig | None = optimizer
        self.scheduler: SchedulerConfig | None = scheduler
        self.locks: Counter[str | Strata] = Counter()
        self.nodes: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._schema_editor: SchemaEditor = SchemaEditor(self)
        self._contract_generation: int = 0
        self._contract_scheduler: ContractScheduler = ContractScheduler()
        self.output_plans: dict[tuple[int, Retain], OutputPlan] = {}
        self.execution_plan: ExecutionPlan | None = None

        ModelGraph.install(self)

        logger.bind(
            component="model",
            batch_size=self.batch_size,
            requests=len(self.schema.active_requests),
            branches=len(self.schema.branches),
            embeds=len(self.schema.embed),
        ).info("initialized Model module")

    @property
    def version(self) -> str:
        """RelFlow version associated with this model's checkpoint provenance."""
        return self._version

    def reset_contracts(self) -> None:
        compiler.clear(self)
        self._contract_generation += 1
        self._contract_scheduler.reset()
        self.output_plans.clear()
        self.execution_plan = None

    # RelFlow's existing region-selection API is chainable, unlike nn.Module.compile.
    def compile(  # pyrefly: ignore[bad-override]
        self,
        *,
        encoders: bool = True,
        pools: bool = True,
        backend: str | Callable = "inductor",
        dynamic: bool | None = None,
        options: dict[str, Any] | None = None,
    ) -> Self:
        """Compile selected tensor regions in place and return this model.

        ``encoders`` selects branch encoder stacks; ``pools`` selects learned
        attention pools in branches and decoders. Routing, coordinate encoders,
        tensorfield operations, and losses keep their existing execution paths.
        Each region uses ``torch.compile(fullgraph=True)`` lazily on real inputs.

        Calls replace the previous selection. Set both switches to ``False``
        to restore eager execution. Schema mutations and checkpoint rebuilds
        clear compilation; checkpoints and model copies do not retain it.
        Custom compute overrides remain eager, as do regions with customized
        internals or nested hooks, including hooks added after compilation.

        ``backend``, ``dynamic``, and ``options`` are PyTorch compiler settings.
        Inductor defaults preserve eager random draws and disable CUDA graphs
        and automatic tensor padding. Floating-point results may still differ.
        """
        self._schema_editor.assert_mutation_allowed("compile")
        if not isinstance(encoders, bool) or not isinstance(pools, bool):
            raise TypeError(
                f"model.compile requires Boolean region switches, got encoders={encoders!r}, pools={pools!r}"
            )
        compiler.compile(self, encoders=encoders, pools=pools, backend=backend, dynamic=dynamic, options=options)
        return self

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> Generator[Text, None, None]:
        parameters = sum(parameter.numel() for parameter in self.parameters())
        heading = Text()
        heading.append(type(self).__name__, style=self.RICH_NAME_STYLE)
        heading.append(" ")
        heading.append("[model]", style=self.RICH_TYPE_STYLE)
        for name, value in (
            ("batch_size", self.batch_size),
            ("d_model", self.schema.d_model),
            ("parameters", f"{parameters:,}"),
            ("branches", len(self.schema.branches)),
            ("fields", len(self.schema.active_requests)),
            ("reconstruct", len(self.schema.reconstruct)),
            ("embeds", len(self.schema.embed)),
        ):
            heading.append(" ")
            heading.append(f"{name}=", style="dim")
            heading.append(str(value), style="cyan")
        yield heading

        lines = list(self.schema.fields.__rich_console__(console, options))
        if not lines:
            return
        first = Text()
        first.append("`-- ", style=self.RICH_TREE_STYLE)
        if isinstance(lines[0], Text):
            first.append_text(lines[0])
        else:
            first.append(str(lines[0]))
        yield first
        for line in lines[1:]:
            nested = Text()
            nested.append("    ", style=self.RICH_TREE_STYLE)
            if isinstance(line, Text):
                nested.append_text(line)
            else:
                nested.append(str(line))
            yield nested

    def select(
        self,
        *predicates: NodeSelector,
        include_root: bool = True,
        use_cache: bool = True,
    ) -> list[Node]:
        """Return schema nodes that satisfy every predicate."""
        return self._schema_editor.select(*predicates, include_root=include_root, use_cache=use_cache)

    def update(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: object,
    ) -> None:
        """Mutate selected schema nodes and rebuild compatible modules.

        Args:
            *predicates: Predicates used to select nodes.
            strict: Raise when a selected node cannot accept one of `values`.
            include_root: Include the root node in predicate matching.
            validate: Validate each node after applying candidate values.
            use_cache: Permit cached selector results. Mutations default this to
                `False` so updates always evaluate against current schema state.
            **values: Declared schema attributes to update. Use `description` for notes.
        """
        self._schema_editor.update(
            *predicates,
            strict=strict,
            include_root=include_root,
            validate=validate,
            use_cache=use_cache,
            **values,
        )

    def extend(
        self,
        *predicates: NodeSelector,
        fields: Mapping[str, TreeFieldInput] | None = None,
        include_root: bool = True,
        use_cache: bool = True,
        **children: TreeFieldInput,
    ) -> None:
        """Append keyword or mapping-named children under one selected branch."""
        self._schema_editor.extend(
            *predicates, fields=fields, include_root=include_root, use_cache=use_cache, **children
        )

    def delete(
        self,
        *predicates: NodeSelector,
        include_root: bool = False,
        use_cache: bool = True,
    ) -> None:
        """Permanently remove selected schema nodes and rebuild modules."""
        self._schema_editor.delete(*predicates, include_root=include_root, use_cache=use_cache)

    def reset(
        self,
        *predicates: NodeSelector,
        include_root: bool = True,
        use_cache: bool = True,
        descendants: bool = False,
    ) -> None:
        """Reinitialize selected runtime node modules while preserving schema values."""
        self._schema_editor.reset(
            *predicates,
            include_root=include_root,
            use_cache=use_cache,
            descendants=descendants,
        )

    @contextmanager
    def override(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        use_cache: bool = False,
        **values: object,
    ) -> Generator[None, None, None]:
        """Temporarily mutate selected schema nodes and keep runtime modules synchronized."""
        with self._schema_editor.override(
            *predicates,
            strict=strict,
            include_root=include_root,
            validate=validate,
            use_cache=use_cache,
            **values,
        ):
            yield

    def configure_callbacks(self) -> list[Callback]:
        callbacks: list[Callback] = []
        factories: set[Callable[[], Callback]] = set()
        trainer = getattr(self, "_trainer", None)
        attached_callback_types = {type(callback) for callback in getattr(trainer, "callbacks", ())}

        if RuntimePlacementCallback not in attached_callback_types:
            callbacks.append(RuntimePlacementCallback())
        if MutationLockCallback not in attached_callback_types:
            callbacks.append(MutationLockCallback())
        if ThroughputLogger not in attached_callback_types:
            callbacks.append(ThroughputLogger())

        for request in self.schema.active_requests.values():
            extension: Extension = TENSORFIELDS[request.type]
            for factory in extension.callback_factories:
                if factory in factories:
                    continue

                factories.add(factory)
                callback = factory()
                if type(callback) not in attached_callback_types:
                    callbacks.append(callback)

        # Callbacks may perform distributed work, so register them in a
        # deterministic order on every rank. Use class paths instead of Python's
        # salted hash or schema traversal order.
        callbacks.sort(
            key=lambda callback: (
                type(callback).__module__,
                type(callback).__qualname__,
            )
        )

        return callbacks

    def on_fit_start(self) -> None:
        if not self.schema.objectives:
            raise RuntimeError(
                "model has no reconstruction objectives; configure at least one active node "
                "with mask=True or Mask(reconstruct=True) before fitting"
            )

    def track(self, names: tuple[str, ...], /, value: MetricValue) -> MetricValue:
        """Log an epoch metric and return the original tensor or metric object."""

        def groupname(names: tuple[str, ...]) -> str:
            assert len(names) > 1

            group, *keys = tuple(map(lambda x: x.replace("/", ".").lower(), names))
            key = ".".join(list(keys))

            return f"{group}/{key}"

        # Scalar metrics are emitted from data-dependent branches, so DDP ranks cannot
        # safely synchronize every scalar log call as a collective. Stateful
        # TorchMetrics are updated/logged on every rank and can aggregate their state.
        stateful = isinstance(value, TorchMetric)
        self.log(
            name=groupname(names),
            value=value.detach() if isinstance(value, torch.Tensor) else value,
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            rank_zero_only=not stateful,
            batch_size=self.batch_size,
        )

        return value

    @property
    def interprocess_encoding_context(self) -> InterprocessEncodingContext:
        """Return extension-owned resources shared with encoding workers."""
        contexts: InterprocessEncodingContext = {}
        for address in self.schema.active_requests:
            state = cast(NodeModule, self.nodes[address]).embedder.context
            if state is not None:
                contexts[Address(str(address))] = state
        return contexts

    @beartype
    def save(self, pathname: Pathname) -> Pathname:
        """Write weights and schema, returning the supplied path unchanged."""
        CheckpointState.save(self, pathname)

        return pathname

    @immutable("forward")
    @beartype
    def forward(
        self,
        inputs: EncodedInput,
        *,
        strata: Strata | str,
        dataloader_idx: int = 0,
    ) -> list[Prediction]:
        return ModelRuntime.forward(self, inputs, strata=strata, dataloader_idx=dataloader_idx)

    if TYPE_CHECKING:
        __call__ = forward

    @beartype
    def configure_optimizers(self) -> torch.optim.Optimizer | OptimizerLRSchedulerConfig:
        if self.optimizer is None:
            raise ValueError("optimizer must be passed to Model before fitting")

        if isinstance(self.optimizer, torch.optim.Optimizer):
            optimizer = self.optimizer
        else:
            optimizer = self.optimizer(self)

        scheduler = self.scheduler(self, optimizer) if callable(self.scheduler) else self.scheduler

        if scheduler is None:
            return optimizer

        # Lightning permits custom stateful schedulers with an overridden step
        # hook, although its return annotation lists only Torch schedulers.
        return cast(OptimizerLRSchedulerConfig, {"optimizer": optimizer, "lr_scheduler": scheduler})

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        CheckpointState.dump(self, checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        CheckpointState.restore_version(self, checkpoint)

    def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore this model in place from a `relflow` checkpoint dictionary."""
        CheckpointState.restore(self, checkpoint)

    @classmethod
    def load(cls, checkpoint: str | Path) -> Self:
        """Load a `Model` checkpoint written by `Model.save(...)`."""
        return CheckpointState.load(cls, checkpoint)

    from_checkpoint = load

    def write(
        self,
        predictions: list[Prediction],
        *,
        source: pa.Table,
        retain: Retain = (),
    ) -> pa.Table:
        """Convert tensor predictions into the canonical Arrow output table."""

        return ModelRuntime.write(self, predictions, source=source, retain=retain)

    @immutable("inference")
    def encode(
        self,
        batch: pa.Table | pa.RecordBatch,
        preprocess: PreprocessorInput = (),
        strata: StrataInput = Strata.predict,
        seed: int = 0,
        epoch: int = 0,
    ) -> EncodedInput:
        """Tensorize an Arrow table or record batch using this model's schema.

        Preprocessors run on eager Polars frames before structural queries.
        ``strata`` selects training, validation, test, or prediction masking;
        ``seed`` and ``epoch`` control deterministic batch randomness.
        """
        return ModelRuntime.encode(
            self,
            batch=batch,
            preprocess=preprocess,
            strata=strata,
            seed=seed,
            epoch=epoch,
        )

    @immutable("inference")
    def predict(
        self,
        batch: PredictionInput,
        preprocess: PreprocessorInput = (),
        postprocess: PostprocessorInput = (),
        retain: Retain = (),
    ) -> pa.Table:
        """Predict an Arrow table, record batch, or nonempty sequence of mappings.

        Returns one Arrow row per processed observation. The ``predictions``
        column contains decoded values and requested embeddings. ``retain``
        keeps selected input columns, or all columns with ``"*"``. Use a typed
        empty Arrow object when there are no observations.

        Preprocessors run before encoding; postprocessors receive the final
        eager Polars frame and may reshape the output before conversion to Arrow.
        """

        return ModelRuntime.predict(
            self,
            batch=batch,
            preprocess=preprocess,
            postprocess=postprocess,
            retain=retain,
        )

    def transfer_batch_to_device(
        self,
        batch: Any,
        device: torch.device,
        dataloader_idx: int,
    ) -> Any:
        """Move encoded tensors while retaining Arrow data on CPU."""

        if isinstance(batch, Encoded):
            return Encoded(
                tensors=batch.tensors.to(device),
                source=batch.source,
                retain=batch.retain,
                observations={address: value.to(device) for address, value in batch.observations.items()},
            )
        return super().transfer_batch_to_device(batch, device, dataloader_idx)

    if TYPE_CHECKING:
        # partialmethod keeps Lightning's runtime hooks compact; declare their
        # bound signatures for editors without replacing the descriptors.
        def training_step(
            self, batch: Encoded | EncodedInput, batch_idx: int, dataloader_idx: int = 0
        ) -> Output | None: ...
        def validation_step(self, batch: Encoded | EncodedInput, batch_idx: int, dataloader_idx: int = 0) -> Output: ...
        def test_step(self, batch: Encoded | EncodedInput, batch_idx: int, dataloader_idx: int = 0) -> Output: ...
        def predict_step(self, batch: Encoded, batch_idx: int, dataloader_idx: int = 0) -> pa.Table: ...
    else:
        training_step = partialmethod(step, strata=Strata.train)
        validation_step = partialmethod(step, strata=Strata.validate)
        test_step = partialmethod(step, strata=Strata.test)
        predict_step = partialmethod(step, strata=Strata.predict)
