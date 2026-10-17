"""Public Lightning model facade for `relflow` schemas."""

from collections import Counter
from collections.abc import Callable, Generator, Mapping, Sequence
from contextlib import contextmanager
from functools import partialmethod
from math import isfinite
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Protocol, Required, Self, TypedDict, TypeVar, Unpack, cast, overload

import lightning.pytorch as lit
import pyarrow as pa
import torch
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import BasePredictionWriter, EarlyStopping
from lightning.pytorch.utilities.types import LRSchedulerConfigType, OptimizerLRSchedulerConfig
from torchmetrics import Metric as TorchMetric

from relflow import presets
from relflow._version import __version__
from relflow.architecture import compiler, template, training
from relflow.architecture.binding import bind
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
    step,
)
from relflow.data.arrow import Encoded, Retain
from relflow.data.datasets.base import EncodedInput, InterprocessEncodingContext
from relflow.data.processors import PostprocessorInput, PreprocessorInput
from relflow.helpers.optimizers import adamw
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
from relflow.structs.structure import OMITTED
from relflow.structs.tree import Address, MaskInput, Node, Rate
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


class Model(lit.LightningModule):
    """Neural model generated from a `relflow` schema tree.

    `Model` owns the schema tree, tensorfield embedders, branch
    encoders, decoders, and convenience methods for prediction, checkpointing,
    schema display and mutation.

    Example:
        ```python
        import relflow as rf

        model = rf.Model(
            segment=rf.Category(),
            label=rf.Category(mask=True),
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
        ffnx: int = 8,
        batch_size: int = 1,
        fields: Mapping[str, TreeFieldInput] | None = None,
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
        ffnx: int = cast(int, OMITTED),
        batch_size: int = 1,
        fields: Mapping[str, TreeFieldInput] | None = None,
        query: str | None = None,
        description: str | None = None,
        embed: bool = False,
        attention: AttentionInput = AttentionMode.mha,
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
        nodes describe repeated objects; the anonymous root is a singleton
        at ``/``, with child addresses such as ``/amount`` and ``/items/sku``.

        ``mask=True`` makes a field a supervised target. ``embed=True`` emits
        its embedding during prediction. ``attention`` selects the root
        encoder mode; ``None`` disables its coordinate and sequence attention.
        ``reduction`` independently selects its output representation.
        ``ffnx`` multiplies the root feed-forward width; each child has its own
        value, defaulting to 8. Activation and normalization are fixed.

        ``optimizer`` accepts a Torch optimizer or a factory receiving this
        model. ``scheduler`` accepts a Torch scheduler, a Lightning scheduler
        configuration dictionary, or a factory receiving model and optimizer.
        Custom schedulers also work with an overridden ``lr_scheduler_step``.
        ``Model.fit`` supplies AdamW when no optimizer is configured. Direct
        Trainer fitting requires an optimizer; neither factory is checkpointed.

        A positional ``Schema`` or ``schema=...`` restores an existing
        architecture and cannot be combined with tree fields or dimensions.
        """
        if schema is not None:
            if ffnx is not OMITTED:
                raise TypeError("schema cannot be combined with a root ffnx override")
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
                ffnx=8 if ffnx is OMITTED else ffnx,
                fields=fields,
                query=query,
                description=description,
                embed=embed,
                attention=attention,
                reduction=reduction,
                dropout=dropout,
                mask=mask,
                **field_kwargs,
            )

        super().__init__()

        # Model provenance intentionally uses a string rather than nn.Module's
        # internal integer state-dict revision; preserve the existing artifact contract.
        self._version: str = __version__  # pyrefly: ignore[bad-override-mutable-attribute]
        self.schema: Schema = schema
        self.preset: presets.Preset | None = None
        self.batch_size = batch_size
        self.optimizer: OptimizerConfig | None
        self.scheduler: SchedulerConfig | None
        # Training recipes remain configuration even when their callables are Modules.
        object.__setattr__(self, "optimizer", optimizer)
        object.__setattr__(self, "scheduler", scheduler)
        self.locks: Counter[str | Strata] = Counter()
        self.nodes: torch.nn.ModuleDict = torch.nn.ModuleDict()
        self._schema_editor: SchemaEditor = SchemaEditor(self)
        self._contract_generation: int = 0
        self._length_warning_generation: int = -1
        self._contract_scheduler: ContractScheduler = ContractScheduler()
        self.output_plans: dict[tuple[int, Retain], OutputPlan] = {}
        self.execution_plan: ExecutionPlan | None = None

        ModelGraph.install(self)
        self.warn_unbounded()

        logger.bind(
            component="model",
            batch_size=self.batch_size,
            requests=len(self.schema.active_requests),
            branches=len(self.schema.branches),
            embeds=len(self.schema.embed),
        ).info("initialized Model module")

    xs = presets.factory(presets.XS)
    sm = presets.factory(presets.SM)
    md = presets.factory(presets.MD)
    lg = presets.factory(presets.LG)
    xl = presets.factory(presets.XL)

    @classmethod
    def from_yaml(cls, pathname: str | Path, /, **overrides: Unpack[template.Options]) -> Self:
        """Construct a fresh model from a named YAML schema and its optional preset.

        Explicit root and runtime overrides replace file values. Fields and the
        built-in preset name come from the file. Loading retains ordinary schema
        validation, registered datatype options, and subclass construction.
        """
        options, profile = template.read(pathname, overrides)
        constructor = cls if profile is None else presets.factory(profile).__get__(None, cls)
        try:
            return constructor(**options)
        except (TypeError, ValueError, BeartypeCallHintParamViolation) as error:
            raise ValueError(f"model template {pathname}: {error}") from error

    @property
    def version(self) -> str:
        """RelFlow version associated with this model's checkpoint provenance."""
        return self._version

    @property
    def batch_size(self) -> int:
        """Current per-device batch size for model-owned data loaders."""
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value: int) -> None:
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"model batch_size must be a positive integer, got {value!r}")
        self._batch_size = value

    def reset_contracts(self) -> None:
        compiler.clear(self)
        self._contract_generation += 1
        self._contract_scheduler.reset()
        self.output_plans.clear()
        self.execution_plan = None
        self.warn_unbounded()

    def warn_unbounded(self) -> None:
        """Report unbounded active collection paths once per model generation."""

        if self._length_warning_generation == self._contract_generation:
            return
        self._length_warning_generation = self._contract_generation
        ancestors = {
            owner.address
            for request in self.schema.active_requests.values()
            for owner in request.path
            if getattr(owner, "type", None) == "branch"
        }
        addresses = [
            str(address)
            for address, branch in self.schema.branches.items()
            if branch.length is None and address in ancestors
        ]
        if addresses:
            logger.bind(component="model", branches=tuple(addresses)).warning(
                "Unbounded branches "
                + ", ".join(repr(address) for address in addresses)
                + " allow untrusted records to determine batch widths and exhaust memory or compute, "
                "causing denial of service. Configure finite length limits and request-size limits before public serving."
            )

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
        Omitted ``dynamic`` enables dynamic shapes. Packed grouping stays eager;
        dense tensor regions remain eligible for compilation.
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

    def select(
        self,
        *predicates: NodeSelector,
        include_root: bool = True,
    ) -> list[Node]:
        """Return schema nodes that satisfy every predicate."""
        return self._schema_editor.select(*predicates, include_root=include_root)

    def update(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: object,
    ) -> None:
        """Mutate selected schema nodes and rebuild compatible modules.

        Args:
            *predicates: Predicates used to select nodes.
            strict: Raise when a selected node cannot accept one of `values`.
            include_root: Include the root node in predicate matching.
            validate: Validate each node after applying candidate values.
            **values: Declared schema attributes to update. Use `description` for notes.
        """
        self._schema_editor.update(
            *predicates,
            strict=strict,
            include_root=include_root,
            validate=validate,
            **values,
        )

    def extend(
        self,
        *predicates: NodeSelector,
        fields: Mapping[str, TreeFieldInput] | None = None,
        include_root: bool = True,
        **children: TreeFieldInput,
    ) -> None:
        """Append keyword or mapping-named children under one selected branch."""
        self._schema_editor.extend(*predicates, fields=fields, include_root=include_root, **children)

    def delete(
        self,
        *predicates: NodeSelector,
        include_root: bool = False,
    ) -> None:
        """Permanently remove selected schema nodes and rebuild modules."""
        self._schema_editor.delete(*predicates, include_root=include_root)

    def reset(
        self,
        *predicates: NodeSelector,
        include_root: bool = True,
        descendants: bool = False,
    ) -> None:
        """Reinitialize selected runtime node modules while preserving schema values."""
        self._schema_editor.reset(
            *predicates,
            include_root=include_root,
            descendants=descendants,
        )

    @contextmanager
    def override(
        self,
        *predicates: NodeSelector,
        strict: bool = True,
        include_root: bool = True,
        validate: bool = True,
        **values: object,
    ) -> Generator[None, None, None]:
        """Temporarily mutate selected schema nodes and keep runtime modules synchronized."""
        with self._schema_editor.override(
            *predicates,
            strict=strict,
            include_root=include_root,
            validate=validate,
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

    def track(
        self,
        names: tuple[str, ...],
        /,
        value: MetricValue,
        *,
        on_step: bool | None = None,
        on_epoch: bool | None = None,
        logger: bool | None = None,
    ) -> MetricValue:
        """Log training steps or evaluation aggregates, using dotted address groups."""
        assert len(names) > 1
        group, *keys = names
        group = ("." if group.startswith("/") else "") + group.strip("/").replace("/", ".")
        key = ".".join(part.replace("/", ".").lower() for part in keys)

        # Scalar metrics are emitted from data-dependent branches, so DDP ranks cannot
        # safely synchronize every scalar log call as a collective. Stateful
        # TorchMetrics are updated/logged on every rank and can aggregate their state.
        stateful = isinstance(value, TorchMetric)
        self.log(
            name=f"{group}/{key}",
            value=value.detach() if isinstance(value, torch.Tensor) else value,
            on_step=on_step,
            on_epoch=not self.training if on_epoch is None else on_epoch,
            logger=logger,
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

        optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
        parameters = {id(parameter) for parameter in self.parameters()}
        if not optimized <= parameters:
            raise ValueError(
                "optimizer references a previous or different model graph; "
                "supply a factory that binds the current model.parameters()"
            )

        scheduler = self.scheduler(self, optimizer) if callable(self.scheduler) else self.scheduler

        if scheduler is None:
            return optimizer

        scheduled = scheduler.get("scheduler") if isinstance(scheduler, Mapping) else scheduler
        if scheduled is not None and scheduled.optimizer is not optimizer:
            raise ValueError(
                "scheduler references a different optimizer; supply a factory receiving the current model and optimizer"
            )

        # Lightning permits custom stateful schedulers with an overridden step
        # hook, although its return annotation lists only Torch schedulers.
        return cast(OptimizerLRSchedulerConfig, {"optimizer": optimizer, "lr_scheduler": scheduler})

    def fit(
        self,
        datamodule: lit.LightningDataModule,
        *,
        batch_size: int | None = None,
        optimizer: OptimizerConfig | None = None,
        scheduler: SchedulerConfig | None = None,
        learning_rate: float | None = None,
        weight_decay: float | None = None,
        betas: tuple[float, float] | None = None,
        eps: float | None = None,
        fused: bool | None = cast(bool | None, OMITTED),
        decay_bias: bool | None = None,
        decay_1d: bool | None = None,
        no_decay_name_fragments: tuple[str, ...] | None = None,
        callbacks: Sequence[Callback] | None = None,
        early_stopping: bool | EarlyStopping = True,
        rollback: bool | RollbackCheckpoint = True,
        monitor: str = "loss/validate",
        mode: Literal["min", "max"] = "min",
        patience: int = 10,
        min_delta: float = 0.0,
        checkpoint_dir: str | Path | None = None,
        ckpt_path: str | Path | None = None,
        **trainer_options: Any,
    ) -> Self:
        """Fit a data module and automatically restore its best saved model state.

        Unspecified hardware uses all available devices, DDP with unused-parameter
        detection across multiple devices or nodes, and BF16 mixed precision where supported. Native training-step
        metrics reach attached loggers every ``log_every_n_steps`` optimizer
        updates (default 50); training loss averages remain internal to callbacks.
        ``val_check_interval`` controls validation independently. Explicit
        Lightning batch limits can shorten a loop; no automatic limits apply.
        ``batch_size`` sets the current per-device size for model-owned data
        loaders, including validation. Omission retains the current size.

        Early stopping and rollback accept booleans or configured callbacks.
        Their default monitor requires a validation split. Train-only fitting
        can disable both policies or explicitly monitor ``loss/train_epoch``.
        Full training continuation requires ``ckpt_path``; another ordinary
        call starts a new run from the current model state.

        Configure AdamW directly with ``learning_rate``, ``weight_decay``,
        ``betas``, ``eps``, and ``fused``. Defaults are 1e-3, 0.01,
        (0.9, 0.95), 1e-8, and True. Biases, one-dimensional parameters,
        and names containing "norm" receive no decay unless their policies
        are overridden. Explicit ``fused=None`` lets Torch select its backend.

        These options and ``scheduler`` belong to this run. The optimizer is
        constructed from the current graph after device and checkpoint setup;
        repeated calls can use different options after rollback or mutation.
        An explicit optimizer configuration cannot be combined with AdamW
        options. Model-level configurations remain defaults for advanced use.
        """
        selected = optimizer if optimizer is not None else self.optimizer
        adamw_options: dict[str, Any] = {
            name: value
            for name, value in {
                "weight_decay": weight_decay,
                "betas": betas,
                "eps": eps,
                "decay_bias": decay_bias,
                "decay_1d": decay_1d,
                "no_decay_name_fragments": no_decay_name_fragments,
            }.items()
            if value is not None
        }
        if fused is not OMITTED:
            adamw_options["fused"] = fused
        if selected is not None and (learning_rate is not None or adamw_options):
            raise ValueError(
                "AdamW options conflict with a configured optimizer; "
                "supply options directly to fit without a separate optimizer, or configure its factory"
            )
        if learning_rate is not None and (not isfinite(learning_rate) or learning_rate <= 0):
            raise ValueError(f"learning_rate must be finite and positive, got {learning_rate!r}")
        configured = training.policies(
            callbacks,
            early_stopping=early_stopping,
            rollback=rollback,
            monitor=monitor,
            mode=mode,
            patience=patience,
            min_delta=min_delta,
            checkpoint_dir=checkpoint_dir,
            options=trainer_options,
        )
        monitors = tuple(
            callback.monitor
            for callback in configured
            if isinstance(callback, (EarlyStopping, RollbackCheckpoint)) and callback.monitor is not None
        )
        if monitors:
            configured.append(training.Selection(monitors))
        runner = training.trainer(
            self,
            datamodule,
            strata=Strata.train,
            batch_size=batch_size,
            callbacks=configured,
            options=trainer_options,
        )
        previous_optimizer, previous_scheduler = self.optimizer, self.scheduler
        object.__setattr__(
            self,
            "optimizer",
            selected
            if selected is not None
            else adamw(learning_rate=1e-3 if learning_rate is None else learning_rate, **adamw_options),
        )
        object.__setattr__(self, "scheduler", scheduler if scheduler is not None else previous_scheduler)
        try:
            runner.fit(self, datamodule=datamodule, ckpt_path=ckpt_path)
        finally:
            object.__setattr__(self, "optimizer", previous_optimizer)
            object.__setattr__(self, "scheduler", previous_scheduler)
        return self

    def validate(
        self,
        datamodule: lit.LightningDataModule,
        *,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        **trainer_options: Any,
    ) -> list[dict[str, float]]:
        """Evaluate validation metrics, optionally updating the per-device batch size."""
        runner = training.trainer(
            self,
            datamodule,
            strata=Strata.validate,
            batch_size=batch_size,
            callbacks=callbacks,
            options=trainer_options,
        )
        return [
            dict(metrics)
            for metrics in runner.validate(self, datamodule=datamodule, ckpt_path=ckpt_path, verbose=verbose)
        ]

    def test(
        self,
        datamodule: lit.LightningDataModule,
        *,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        verbose: bool = True,
        **trainer_options: Any,
    ) -> list[dict[str, float]]:
        """Evaluate test metrics, optionally updating the per-device batch size."""
        runner = training.trainer(
            self,
            datamodule,
            strata=Strata.test,
            batch_size=batch_size,
            callbacks=callbacks,
            options=trainer_options,
        )
        return [
            dict(metrics) for metrics in runner.test(self, datamodule=datamodule, ckpt_path=ckpt_path, verbose=verbose)
        ]

    def on_save_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        CheckpointState.dump(self, checkpoint)

    def on_load_checkpoint(self, checkpoint: dict[str, Any]) -> None:
        CheckpointState.restore_metadata(self, checkpoint)

    def restore_checkpoint_state(self, checkpoint: dict[str, Any]) -> None:
        """Restore this model in place from a `relflow` checkpoint dictionary."""
        CheckpointState.restore(self, checkpoint)

    @classmethod
    def load(cls, checkpoint: str | Path) -> Self:
        """Load a `Model` checkpoint written by `Model.save(...)`."""
        return CheckpointState.load(cls, checkpoint)

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

    @overload
    def predict(
        self,
        batch: PredictionInput,
        preprocess: PreprocessorInput = (),
        postprocess: PostprocessorInput = (),
        retain: Retain = (),
        *,
        batch_size: int | None = None,
    ) -> pa.Table: ...

    @overload
    def predict(
        self,
        *,
        datamodule: lit.LightningDataModule,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        return_predictions: Literal[True] = True,
        **trainer_options: Any,
    ) -> pa.Table: ...

    @overload
    def predict(
        self,
        *,
        datamodule: lit.LightningDataModule,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        return_predictions: Literal[False],
        **trainer_options: Any,
    ) -> None: ...

    @overload
    def predict(
        self,
        batch: lit.LightningDataModule,
        *,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        return_predictions: Literal[True] = True,
        **trainer_options: Any,
    ) -> pa.Table: ...

    @overload
    def predict(
        self,
        batch: lit.LightningDataModule,
        *,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        return_predictions: Literal[False],
        **trainer_options: Any,
    ) -> None: ...

    @immutable("inference")
    def predict(
        self,
        batch: PredictionInput | lit.LightningDataModule | None = None,
        preprocess: PreprocessorInput = (),
        postprocess: PostprocessorInput = (),
        retain: Retain = (),
        *,
        datamodule: lit.LightningDataModule | None = None,
        batch_size: int | None = None,
        callbacks: Sequence[Callback] | None = None,
        ckpt_path: str | Path | None = None,
        return_predictions: bool = True,
        **trainer_options: Any,
    ) -> pa.Table | None:
        """Predict direct observations or a data module's batched prediction split.

        Returns one Arrow row per processed observation. The ``predictions``
        column nests rich decoded results under schema field names and record
        lists. Requested embeddings appear separately in ``representations``;
        either column is omitted when unconfigured. ``retain`` adds selected
        input columns under ``inputs``, or all columns with ``"*"``. Use a typed
        empty Arrow object when there are no observations.

        Preprocessors run before encoding; postprocessors receive the final
        eager Polars frame and may reshape the output before conversion to Arrow.
        For direct inputs, ``batch_size`` bounds each encoded and forwarded
        batch after preprocessing. Postprocessing runs once on the combined
        output. A supplied size also becomes the model's current batch size.
        Omission processes the direct input as one batch.

        Data-module inference accepts Lightning callbacks and Trainer options.
        ``batch_size`` updates the current per-device size for model-owned data
        loaders; omission retains the current size.
        Configure preprocessing/retention on the data module and postprocessing
        on a Writer. Distributed prediction requires a prediction writer and
        ``return_predictions=False``; ``devices=1`` collects one Arrow table.
        """
        if batch is not None and datamodule is not None:
            raise TypeError("predict accepts either observations or a datamodule, not both")
        if isinstance(batch, lit.LightningDataModule):
            datamodule = batch
            batch = None
        if datamodule is None:
            if batch is None:
                raise TypeError("predict requires observations or a datamodule")
            if callbacks is not None or ckpt_path is not None or not return_predictions or trainer_options:
                raise TypeError("callbacks, checkpoint loading, and Trainer options require predict(datamodule=...)")
            if batch_size is not None:
                self.batch_size = batch_size
            return ModelRuntime.predict(
                self, batch=batch, preprocess=preprocess, postprocess=postprocess, retain=retain, batch_size=batch_size
            )
        if preprocess != () or postprocess != () or retain != ():
            raise TypeError(
                "data-module prediction configures preprocessing/retention on the data module and postprocessing on Writer"
            )
        if not return_predictions and not any(
            isinstance(callback, BasePredictionWriter) for callback in callbacks or ()
        ):
            raise ValueError("return_predictions=False requires a prediction writer in callbacks")
        runner = training.trainer(
            self,
            datamodule,
            strata=Strata.predict,
            batch_size=batch_size,
            callbacks=callbacks,
            options=trainer_options,
        )
        if return_predictions and runner.num_devices * runner.num_nodes > 1:
            raise ValueError(
                "distributed prediction requires a writer and return_predictions=False; use devices=1 to collect an Arrow table"
            )
        outputs = runner.predict(
            self, datamodule=datamodule, ckpt_path=ckpt_path, return_predictions=return_predictions
        )
        if not return_predictions:
            return None
        tables = [table for output in outputs or () for table in (output if isinstance(output, list) else [output])]
        if any(not isinstance(table, pa.Table) for table in tables):
            raise TypeError("data-module predict_step must return pyarrow.Table batches")
        return pa.concat_tables(tables) if tables else ModelRuntime.write(self, [], source=pa.table({}))

    def on_before_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        """Commit extension resources before DDP starts this batch's graph."""
        if isinstance(batch, Encoded):
            trainer = self.trainer
            strata = (
                Strata.train
                if trainer.training
                else Strata.validate
                if trainer.validating or trainer.sanity_checking
                else Strata.test
                if trainer.testing
                else Strata.predict
            )
            return bind(self, batch, strata)
        return batch

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
                bindings=batch.bindings,
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
