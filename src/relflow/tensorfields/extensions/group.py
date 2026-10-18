from __future__ import annotations

import math
from collections.abc import Generator, Mapping, Sequence
from contextlib import AbstractContextManager, contextmanager
from dataclasses import InitVar
from typing import TYPE_CHECKING, Annotated, Any, Literal, Unpack, cast
from weakref import ReferenceType, ref

import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from lightning.pytorch import Callback, LightningModule, Trainer
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.distributed import all_reduce_sum, broadcast_object
from relflow.helpers.resize import Resize
from relflow.logging import logger
from relflow.structs.enums import Component, Metric, Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel, Prediction
from relflow.structs.tree import Address, FieldOptions
from relflow.tensorfields.base import (
    Context,
    DecoderBase,
    EmbedderBase,
    Extension,
    RequestBase,
    TensorFieldBase,
    TensorInput,
)
from relflow.tensorfields.grouping import Refinement
from relflow.tensorfields.output import array, labels, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.reconstruction import Metrics, Reconstruction, Scores
from relflow.tensorfields.shared.rows import Embedding
from relflow.tensorfields.shared.vocabulary import (
    OnlineVocabularyModel,
    VocabularyBatch,
    VocabularyState,
    VocabularySyncCallback,
    capacity,
)

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch

group: Extension = Extension(name="group", types=(bool, int, float, str, bytes))

group.callback(VocabularySyncCallback, CounterUpdateCallback, Metrics)

# Algorithmic constants: fixed values from the SwAV/DINO literature or symmetry-breaking
# magnitudes that have no reason to be user-tuned.
_BALANCE_EPSILON: float = 0.05
_BALANCE_ITERS: int = 3
_REVIVE_NOISE: float = 0.02
_REVIVE_WARMUP: int = 1
_OVERRIDE_LOCK: str = "group assignment overrides"

Assignment = int | torch.Tensor | list[float] | tuple[float, ...]


@torch.no_grad()
def initialize(weight: torch.Tensor, start: int, stop: int, retired: torch.Tensor) -> None:
    """Break assignment symmetry weakly and evenly without changing learned rows."""
    available = (~retired).nonzero(as_tuple=True)[0]
    rows = weight[start:stop]
    rows.zero_()
    selected = available[torch.arange(start, stop, device=weight.device) % available.numel()]
    rows.scatter_(1, selected.unsqueeze(-1), 0.1)


@group.register(Component.Request)
class Request(RequestBase):
    """Scalar labels represented through shared, learned latent groups.

    Training grows label storage automatically. ``n_groups=K`` normalizes to ``(K, K)``;
    ``n_groups=(lower, upper)`` allows an adaptive group count within that
    range. ``p_unavailable`` routes known training labels through shared unavailable content.
    ``ema_decay`` smooths usage statistics; ``revive_temperature=0`` disables
    inactive-group revival. Assignments initialize weakly and evenly and route
    deterministically with straight-through gradients. A reconstructing mask
    trains the group head; identity likelihoods normalize within each group.
    When Group has a reconstruction objective, fitting can refine memberships
    against its own training identity likelihood, accepting only improvement
    on its own validation likelihood. Other losses guide ordinary backpropagation
    without defining Group's objective. Predictions expose the best group and identity.
    """

    type: Literal["group"] = "group"
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01

    ema_decay: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.99)] = 0.99
    revive_temperature: Annotated[float, pydantic.Field(ge=0.0, default=10.0)] = 10.0

    n_groups: Annotated[
        tuple[int, int],
        pydantic.Field(
            description="(lower_bound, upper_bound) for the number of groups. A single int is broadcast to (K, K) to specify an exact number of groups.",
        ),
    ]

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            n_groups: int | tuple[int, int],
            p_unavailable: float = 0.01,
            ema_decay: float = 0.99,
            revive_temperature: float = 10.0,
            type: Literal["group"] = "group",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.field_validator("n_groups", mode="before")
    @classmethod
    def check_n_groups_input(cls, value: Any) -> Any:
        if isinstance(value, int) and not isinstance(value, bool):
            return (value, value)
        return value

    @pydantic.model_validator(mode="after")
    def check_n_groups(self):
        if len(self.n_groups) != 2:
            raise ValueError("n_groups tuple must have exactly 2 elements (lower_bound, upper_bound)")
        lower, upper = self.n_groups
        if lower <= 0:
            raise ValueError("n_groups lower bound must be greater than 0")
        if lower > upper:
            raise ValueError("n_groups lower bound must be less than or equal to upper bound")
        return self

    @classmethod
    def assign(
        cls,
        model: "Model",
        address: Address | str,
        token: Any,
        assignment: int | torch.Tensor | list[float] | tuple[float, ...],
        /,
    ) -> int:
        """Persistently assign one token to a group or group distribution."""
        return GroupRuntime(model=model, address=address).assign(token, assignment)

    @classmethod
    def vocabulary(
        cls,
        source: "Model | InterprocessEncodingContext",
        address: Address | str,
        /,
    ) -> tuple[Any, ...]:
        """Return an immutable snapshot of one Group vocabulary."""
        from relflow.architecture.root import Model

        normalized = Address(str(address))
        if isinstance(source, Model):
            embedder = resolve_group_embedder(source, normalized)
            with embedder.vocab.lock:
                return tuple(embedder.vocab.master)
        elif isinstance(source, Mapping):
            if normalized not in source:
                raise KeyError(f"no encoding context at address {str(normalized)!r}")

            state = source[normalized]
            if not isinstance(state, VocabularyState):
                raise TypeError(
                    f"encoding context at {str(normalized)!r} is not a VocabularyState (got {type(state).__name__})"
                )
        else:
            raise TypeError(
                f"Group.vocabulary source must be a Model or InterprocessEncodingContext, got {type(source).__name__}"
            )

        with state.lock:
            return tuple(state.master)

    @classmethod
    def assignments(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[Any, dict[str, int | tuple[float, ...]]]:
        """Return evaluation-time assignments for every populated token."""
        embedder = resolve_group_embedder(model, address)
        with embedder.vocab.lock:
            vocabulary = tuple(embedder.vocab.master)
            logits = (
                cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)[: len(vocabulary)].detach().clone()
            )

        if not vocabulary:
            return {}

        logits = embedder.support(logits)
        probabilities = torch.softmax(logits, dim=-1)
        groups: list[int] = logits.argmax(dim=-1).cpu().tolist()
        rows: list[list[float]] = probabilities.cpu().tolist()
        return {
            token: {
                "group": group_id,
                "probabilities": tuple(float(probability) for probability in row),
            }
            for token, group_id, row in zip(vocabulary, groups, rows, strict=True)
        }

    @classmethod
    def status(
        cls,
        model: "Model",
        address: Address | str,
        /,
    ) -> dict[str, tuple[int, ...] | tuple[float, ...]]:
        """Return the balancing subset, retired IDs, and reconstruction-head usage."""
        embedder = resolve_group_embedder(model, address)
        committed = embedder.committed.detach().cpu().clone()
        usage = embedder.usage_ema.detach().cpu().clone()

        return {
            "committed": tuple(index for index, value in enumerate(committed.tolist()) if value),
            "retired": tuple(embedder.retired.nonzero(as_tuple=True)[0].cpu().tolist()),
            "usage": tuple(float(value) for value in usage.tolist()),
        }

    @classmethod
    def override(
        cls,
        model: "Model",
        address: Address | str,
        assignments: Mapping[Any, int | torch.Tensor | list[float] | tuple[float, ...]],
        /,
    ) -> AbstractContextManager[None]:
        """Temporarily override token assignments within a context."""
        return GroupRuntime(model=model, address=address).override(assignments)


@group.register(Component.observe)
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Reserve and count the complete pristine grouping exposure."""

    if not learn:
        return None
    if not isinstance(state, VocabularyState):
        raise RuntimeError(f"group field at '{address}' requires a vocabulary encoding context")

    indices = state.indices(field.values, learn=True)
    content = torch.from_numpy(indices.copy())
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(content.masked_fill(content < 0, state.size), state.size + 1),
        },
        batch_size=[],
    )


@group.register(Component.TensorField)
@tensorclass
class TensorField(TensorFieldBase[torch.Tensor]):
    state: torch.Tensor
    content: torch.Tensor
    present: torch.Tensor
    trainable: torch.Tensor
    inferred: torch.Tensor
    targets: TensorDict

    if TYPE_CHECKING:
        # TensorClass metadata is accepted by its generated initializer.
        batch_size: InitVar[int | torch.Size | list[int] | tuple[int, ...] | None] = None
        device: InitVar[torch.device | str | int | None] = None
        names: InitVar[list[str | None] | None] = None

    @classmethod
    def new(
        cls,
        input: RaggedField,
        target: RaggedField,
        present: torch.Tensor,
        trainable: torch.Tensor,
        inferred: torch.Tensor,
        address: Address,
        schema: Schema,
        strata: Strata,
        context: Context,
    ) -> TensorField:
        state = context.state
        if state is not None and not isinstance(state, VocabularyState):
            raise TypeError(f"group field at '{address}' requires VocabularyState context, got {type(state).__name__}")

        def encode(field: RaggedField) -> torch.Tensor:
            if not len(field.values):
                return torch.zeros(field.shape, dtype=torch.int64)
            if state is None:
                raise RuntimeError(f"group field at '{address}' requires a vocabulary encoding context")
            tokens = state.indices(field.values, learn=False)
            return torch.from_numpy(field.place(tokens, fill=0))

        content = encode(input)
        target_content = encode(target)

        state_tensor = torch.from_numpy(input.dense)
        target_state = torch.from_numpy(target.dense)
        if strata == Strata.train:
            p_unavailable: float = cast(Request, schema.requests[address]).p_unavailable
            unavailable_index = -1

            if p_unavailable > 0.0:
                # OOV simulation is input corruption, not a change to answers.
                known = state_tensor.eq(Tokens.valued.value) & content.ne(unavailable_index)
                selected = torch.rand_like(state_tensor, dtype=torch.float).lt(p_unavailable) & known
                content = content.masked_fill(selected, unavailable_index)

        return cls(
            state=state_tensor,
            content=content,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: target_state,
                    TensorKey.content: target_content,
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@group.register(Component.Embedder)
class Embedder(EmbedderBase):
    usage_ema: torch.Tensor
    committed: torch.Tensor
    retired: torch.Tensor
    adherence_ema: torch.Tensor

    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address
        self.size: int = request.n_groups[-1]

        if address not in schema.reconstruct:
            # Group loss fires only for reconstructed rows; a plain-input
            # Group silently freezes n_committed at initialization.
            logger.bind(component="tensorfield", field_type="group", address=str(address)).warning(
                f"Group field {address!s} has no reconstructing Mask; dynamic K-selection "
                "will not engage. Add Mask(reconstruct=True) to train the group head."
            )

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel()
        self._override_depth: int = 0

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.group.name: Embedding(
                    num_embeddings=self.capacity + 1,
                    embedding_dim=self.size,
                ),
                TensorKey.content.name: torch.nn.Linear(
                    in_features=self.size,
                    out_features=schema.d_model,
                    bias=False,
                ),
            }
        )
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
                TensorKey.content.name: Counter(address=address, size=self.capacity + 1),
            }
        )

        lower: int = request.n_groups[0]
        usage_init = torch.zeros(self.size)
        usage_init[:lower] = 1.0 / lower
        committed_init = torch.zeros(self.size, dtype=torch.bool)
        committed_init[:lower] = True
        self.register_buffer("usage_ema", usage_init)
        self.register_buffer("committed", committed_init)
        self.register_buffer("retired", torch.zeros(self.size, dtype=torch.bool))
        self.register_buffer("adherence_ema", torch.zeros(()))
        self.register_buffer("information_ema", torch.tensor(math.log(lower)))
        self.register_buffer("statistics", torch.zeros(self.size + 2), persistent=False)
        with torch.no_grad():
            weight = cast(torch.Tensor, self.embeddings[TensorKey.group.name].weight)
            weight.zero_()
            initialize(weight, 0, self.capacity, self.retired)

    @property
    def capacity(self) -> int:
        return self.vocab.size

    def support(self, logits: torch.Tensor) -> torch.Tensor:
        """Retired groups have no assignment or prediction probability until revived."""
        return logits.masked_fill(self.retired, float("-inf"))

    def resize(self, size: int, resize: Resize) -> None:
        """Expand identity rows without changing group columns or their state."""
        previous = self.embeddings[TensorKey.group.name].weight
        resize.embedding(cast(torch.nn.Embedding, self.embeddings[TensorKey.group.name]), size + 1, tail=True)
        for old, new in resize.parameters:
            if old is previous:
                initialize(new, old.shape[0] - 1, new.shape[0] - 1, self.retired)
        cast(Counter, self.counters[TensorKey.content.name]).resize(size + 1, resize, tail=True)
        if self.vocab.size != size:
            resize.attribute(self.vocab, "size", size)

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N: int
        dims: list[int]

        N, *dims = inputs.state.shape
        state = inputs.state.reshape(-1)
        content = cast(torch.Tensor, inputs.content).reshape(-1)
        valued = state.eq(Tokens.valued.value)

        if valued.any():
            valid_content = content.masked_select(valued)
            if ((valid_content < -1) | (valid_content >= self.capacity)).any().item():
                raise ValueError(f"Token in address {self.origin} outside [-1, {self.capacity})")

        safe_content = content.masked_fill(content.eq(-1), self.capacity).masked_fill(~valued, 0)
        assign_logits = self.support(self.embeddings[TensorKey.group.name](safe_content))
        if self.training:
            soft = assign_logits.softmax(dim=-1)
            hard = torch.zeros_like(soft).scatter_(1, soft.argmax(dim=-1, keepdim=True), 1.0)
            group_assignments = hard - soft.detach() + soft
        else:
            hard = torch.zeros_like(assign_logits)
            hard.scatter_(-1, assign_logits.argmax(dim=-1, keepdim=True), 1.0)
            group_assignments = hard
        group_assignments = group_assignments * valued.unsqueeze(-1)
        group_embeddings = self.embeddings[TensorKey.content.name](group_assignments)

        embeddings: torch.Tensor = (self.embeddings[TensorKey.state.name](state) + group_embeddings).reshape(
            N, *dims, -1
        )

        return Parcel(
            payload=embeddings,
            present=torch.ones(N, dtype=torch.bool, device=embeddings.device),
            origin=self.origin,
            destination=self.destination,
            batch_size=N,
        )

    @property
    def context(self) -> VocabularyState:
        return self.vocab.state

    def assignment_logits(self, assignment: Assignment) -> torch.Tensor:
        weight = cast(torch.Tensor, self.embeddings[TensorKey.group.name].weight)
        if isinstance(assignment, bool):
            raise TypeError("assignment must be a group index or probability vector")
        if isinstance(assignment, int):
            if not 0 <= assignment < self.size:
                raise ValueError(f"group must be in [0, {self.size}); got {assignment}")
            if bool(self.retired[assignment]):
                raise ValueError(f"group {assignment} at {self.origin} is retired; choose an available group")
            distribution = torch.zeros(self.size, dtype=weight.dtype, device=weight.device)
            distribution[assignment] = 1.0
            return torch.log(distribution.clamp_min(1e-8))
        distribution = torch.as_tensor(assignment, dtype=weight.dtype, device=weight.device)
        if distribution.shape != (self.size,):
            raise ValueError(
                f"assignment probabilities must have shape ({self.size},); got {tuple(distribution.shape)}"
            )
        if bool((distribution < 0).any().item()):
            raise ValueError("assignment probabilities must be non-negative")
        total = float(distribution.sum().item())
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(f"assignment probabilities must sum to 1; got {total}")
        if bool(distribution[self.retired].gt(0).any()):
            raise ValueError(
                f"assignment at {self.origin} gives probability to a retired group; choose available groups"
            )
        return torch.log(distribution.clamp_min(1e-8))

    def assign_token(self, token: Any, assignment: Assignment) -> int:
        logits = self.assignment_logits(assignment)
        weight = cast(torch.Tensor, self.embeddings[TensorKey.group.name].weight)
        index = self.vocab.state.index.get(token)
        if index is None:
            if len(self.vocab.master) >= self.capacity:
                raise ValueError(f"group field {self.origin} at capacity ({self.capacity}); cannot assign {token!r}")
            self.vocab.master.append(token)
            self.vocab._snapshot_cache = None
            self.vocab._snapshot_size = -1
            index = len(self.vocab.master) - 1
        with torch.no_grad():
            weight[index].copy_(logits)
        return index

    @contextmanager
    def override_assignments(self, assignments: Mapping[Any, Assignment]) -> Generator[None, None, None]:
        if self._override_depth:
            raise RuntimeError("Group assignment overrides cannot be nested")

        self._override_depth += 1
        weight = cast(torch.Tensor, self.embeddings[TensorKey.group.name].weight)
        saved: list[tuple[int, torch.Tensor, bool]] = []
        try:
            for token, assignment in assignments.items():
                existing = self.vocab.state.index.get(token)
                if existing is not None:
                    original = weight[existing].detach().clone()
                    self.assign_token(token, assignment)
                    saved.append((existing, original, False))
                    continue

                if len(self.vocab.master) >= self.capacity:
                    raise ValueError(
                        f"group field {self.origin} at capacity ({self.capacity}); cannot override {token!r}"
                    )
                new_index = len(self.vocab.master)
                original = weight[new_index].detach().clone()
                self.assign_token(token, assignment)
                saved.append((new_index, original, True))

            yield
        finally:
            try:
                with torch.no_grad():
                    for index, original, was_new in reversed(saved):
                        weight[index].copy_(original)
                        if was_new and self.vocab.master:
                            self.vocab.master.pop()
                            self.vocab._snapshot_cache = None
                            self.vocab._snapshot_size = -1
            finally:
                self._override_depth -= 1

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        if self._override_depth:
            raise RuntimeError("cannot save or rebuild a model while Group assignment overrides are active")
        super()._save_to_state_dict(destination, prefix, keep_vars)


@group.register(Component.bind)
def bind(
    module: Model,
    field: TensorFieldBase,
    observation: TensorDict | None,
    binding: object,
    *,
    address: Address,
    strata: Strata,
    resize: Resize,
) -> None:
    """Grow source identities while preserving the unavailable row and group count."""
    if not isinstance(binding, VocabularyBatch):
        raise TypeError(f"group binding at '{address}' requires VocabularyBatch, got {type(binding).__name__}")
    embedder = cast(Embedder, module.nodes[address].embedder)
    mapping, size = binding.resolve(embedder.vocab, learn=strata == Strata.train, device=module.device, resize=resize)
    embedder.resize(size, resize)
    field = cast(TensorField, field)
    field.content = mapping.to(field.content.device)[field.content]
    values = cast(torch.Tensor, field.targets[TensorKey.content])
    field.targets[TensorKey.content] = mapping.to(values.device)[values]
    if observation is not None:
        counter_mapping = mapping.clone()
        counter_mapping[-1] = size
        observation[TensorKey.content] = binding.counts(
            cast(torch.Tensor, observation[TensorKey.content]), counter_mapping, size + 1
        )


@group.register(Component.learn)
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine grouping counts to model-owned resources."""

    if strata != Strata.train:
        raise ValueError(f"group learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    cast(Counter, embedder.counters[TensorKey.state.name]).learn(cast(torch.Tensor, observation[TensorKey.state]))
    cast(Counter, embedder.counters[TensorKey.content.name]).learn(cast(torch.Tensor, observation[TensorKey.content]))


def resolve_group_embedder(model: "Model", address: Address | str) -> Embedder:
    """Resolve a live Group embedder without importing Model at module load."""
    from relflow.architecture.root import Model

    if not isinstance(model, Model):
        raise TypeError(f"Group binding model must be a Model, got {type(model).__name__}")

    normalized = Address(str(address))
    if normalized not in model.nodes:
        raise KeyError(f"no field at address {str(normalized)!r}")

    embedder = getattr(model.nodes[normalized], "embedder", None)
    if not isinstance(embedder, Embedder):
        raise TypeError(f"address {str(normalized)!r} is not a Group field (got {type(embedder).__name__})")
    return embedder


class GroupRuntime:
    """Resolve and mutate one Group field's model-owned runtime state."""

    def __init__(self, model: "Model", address: Address | str):
        self._model: ReferenceType[Model] = ref(model)
        self.address: Address = Address(str(address))
        self.resolve()

    def resolve(self) -> tuple[Model, Embedder]:
        model = self._model()
        if model is None:
            raise RuntimeError("the model bound to this Group field no longer exists")
        if self.address not in model.nodes:
            raise KeyError(f"no field at address {str(self.address)!r}")

        embedder = getattr(model.nodes[self.address], "embedder", None)
        if not isinstance(embedder, Embedder):
            raise TypeError(f"address {str(self.address)!r} is not a Group field (got {type(embedder).__name__})")
        return model, embedder

    @staticmethod
    def assert_writable(model: Model, embedder: Embedder) -> None:
        if embedder.vocab.is_shared:
            raise RuntimeError("Group assignments cannot change while vocabulary state is shared")
        active = tuple(str(name) for name, count in model.locks.items() if count > 0)
        if active:
            raise RuntimeError(f"Group assignments cannot change while the model is active: {', '.join(active)}")

    def assign(self, token: Any, assignment: Assignment) -> int:
        from relflow.architecture.binding import optimizers

        model, embedder = self.resolve()
        with embedder.vocab.lock:
            self.assert_writable(model, embedder)
            embedder.assignment_logits(assignment)
            required = len(embedder.vocab.master) + int(token not in embedder.vocab.state.index)
            resize = Resize(optimizers(model))
            embedder.resize(capacity(embedder.capacity, required), resize)
            resize.commit()
            return embedder.assign_token(token, assignment)

    @contextmanager
    def override(self, assignments: Mapping[Any, Assignment]) -> Generator[None, None, None]:
        from relflow.architecture.binding import optimizers

        model, embedder = self.resolve()
        with embedder.vocab.lock:
            self.assert_writable(model, embedder)
            index = embedder.vocab.state.index
            for assignment in assignments.values():
                embedder.assignment_logits(assignment)
            required = len(index) + sum(token not in index for token in assignments)
            resize = Resize(optimizers(model))
            embedder.resize(capacity(embedder.capacity, required), resize)
            resize.commit()
            model.locks[_OVERRIDE_LOCK] += 1
            try:
                with embedder.override_assignments(assignments):
                    yield
            finally:
                if model.locks[_OVERRIDE_LOCK] <= 1:
                    model.locks.pop(_OVERRIDE_LOCK, None)
                else:
                    model.locks[_OVERRIDE_LOCK] -= 1


@group.register(Component.Decoder)
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        # Group assignments belong to stable identities. Row-varying sibling
        # values may supervise them through downstream objectives, but must not
        # alter the query used to reconstruct the identity itself.
        super().__init__(schema=schema, address=address, conditioned=False)

        request: Request = cast(Request, schema.requests[address])
        size: int = request.n_groups[-1]
        self.metrics = Scores(address, Reconstruction)

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.group.name: torch.nn.Linear(in_features=schema.d_model, out_features=size, bias=False),
            }
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        return TensorDict(
            source={
                TensorKey.state: self.linears[TensorKey.state.name](pooled),
                TensorKey.group: self.linears[TensorKey.group.name](pooled),
            }
        )


@group.register(Component.loss)
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    embedder: Embedder = cast(Embedder, module.nodes[prediction.address].embedder)
    request: Request = cast(Request, module.schema.requests[prediction.address])
    N: int = cast(torch.Tensor, batch.targets[TensorKey.state]).numel()
    trainable = batch.trainable.reshape(N)

    state_inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(N, -1)
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(N)

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=(
            torch.nn.functional.cross_entropy(
                input=state_inputs,
                target=state_targets,
                reduction="none",
            )
            .masked_select(trainable)
            .mean()
        ),
    )

    module.track(
        (prediction.address, strata, Metric.accuracy, TensorKey.state),
        value=state_inputs.argmax(dim=1).eq(state_targets).masked_select(trainable).float().mean(),
    )
    module.track(
        (prediction.address, strata, "vocabulary", "size"),
        value=state_inputs.new_tensor(len(embedder.vocab.master), dtype=torch.float32),
    )

    valued: torch.Tensor = trainable & state_targets.eq(Tokens.valued.value)
    group_logits = embedder.support(cast(torch.Tensor, prediction.payload[TensorKey.group]).reshape(N, -1))
    content_targets = cast(torch.Tensor, batch.targets[TensorKey.content]).reshape(N)

    vocab_size = len(embedder.vocab.master)
    known = valued & content_targets.ge(0) & content_targets.lt(vocab_size)
    # The reconstruction support matches the writer. Neither the unavailable
    # input embedding nor unallocated label rows are output candidates.
    group_probs = group_logits[known].float().softmax(dim=-1)
    vocab_logits = vocabulary_logits(embedder, group_probs)
    objective = torch.nn.functional.cross_entropy(vocab_logits, content_targets[known], reduction="sum")
    decoder = cast(Decoder, module.nodes[prediction.address].decoder)
    with torch.no_grad():
        correct = known.sum().new_zeros(())
        if vocab_size:
            correct = vocab_logits.argmax(dim=-1).eq(content_targets[known]).sum()
        decoder.metrics.record(
            module, strata, torch.stack((known.sum(), valued.sum(), correct)), torch.stack((objective, objective))
        )
    loss = loss + objective / known.sum().clamp_min(1)

    if strata != Strata.train:
        return loss

    # Unknown identities provide no assignment evidence: they must not
    # move the usage EMA, committed count, or balancing targets.
    valued_logits = group_logits[known].float()
    n_valued: int = valued_logits.shape[0]
    lower: int = request.n_groups[0]

    valued_probs = valued_logits.softmax(dim=-1)
    with torch.no_grad():
        entropy = -(valued_probs * valued_probs.clamp_min(1e-12).log()).sum()
        embedder.statistics.add_(
            torch.cat((valued_probs.sum(dim=0), entropy.reshape(1), valued_probs.new_tensor([n_valued])))
        )

    if n_valued < max(lower, 2):
        return loss

    with torch.no_grad():
        usage_normalized = embedder.usage_ema / embedder.usage_ema.sum().clamp_min(1e-12)
        usage_safe = usage_normalized.clamp_min(1e-12)
        usage_entropy = -(usage_safe * usage_safe.log()).sum()
        perplexity = embedder.information_ema.exp()
        committed_idx = embedder.committed.nonzero(as_tuple=True)[0]
        n_committed = committed_idx.numel()

        # Preserve finite-logit support under sharp assignments. Exponentiating
        # before balancing can underflow entire columns, leaving no recovery
        # gradient even though those columns remain committed.
        log_q = valued_logits[:, committed_idx] / _BALANCE_EPSILON
        for _ in range(_BALANCE_ITERS):
            log_q = log_q - torch.logsumexp(log_q, dim=1, keepdim=True) - math.log(n_valued)
            log_q = log_q - torch.logsumexp(log_q, dim=0, keepdim=True) - math.log(n_committed)
        Q = log_q.softmax(dim=1)

    full_log_probs = torch.log_softmax(valued_logits, dim=-1)
    committed_log_probs = full_log_probs[:, committed_idx]
    balance_loss = -(Q * committed_log_probs).sum(dim=-1).mean()

    loss += module.track(
        (prediction.address, strata, Metric.loss, TensorKey.group),
        value=balance_loss,
    )
    module.track(
        (prediction.address, strata, "group", "active"),
        value=perplexity,
    )
    module.track(
        (prediction.address, strata, "group", "committed"),
        value=state_inputs.new_tensor(float(n_committed)),
    )
    module.track(
        (prediction.address, strata, "group", "usage_entropy"),
        value=usage_entropy,
    )

    return loss


@torch.no_grad()
def update_usage(embedder: Embedder, request: Request, statistics: torch.Tensor) -> None:
    """Update grouping evidence once from globally weighted training observations.

    Marginal entropy alone mistakes identical uncertain predictions for many
    distinct groups. Subtracting the mean prediction entropy measures variation
    between observations instead. Empty batches provide no evidence.
    """
    count = statistics[-1]
    if not bool(count > 0):
        return
    usage = statistics[:-2] / count
    entropy = -(usage * usage.clamp_min(1e-12).log()).sum()
    information = (entropy - statistics[-2] / count).clamp_min(0)
    decay = request.ema_decay
    embedder.usage_ema.mul_(decay).add_(usage * (1 - decay))
    embedder.information_ema.mul_(decay).add_(information * (1 - decay))
    lower, upper = request.n_groups
    available = int((~embedder.retired).sum().item())
    count = max(lower, min(upper, available, round(embedder.information_ema.exp().item())))
    indices = embedder.usage_ema.masked_fill(embedder.retired, -1.0).topk(count).indices
    embedder.committed.zero_()
    embedder.committed[indices] = True
    if lower != upper:
        mass = usage[~embedder.committed].sum()
        embedder.adherence_ema.mul_(decay).add_(mass * (1 - decay))


def vocabulary_logits(embedder: Embedder, probabilities: torch.Tensor) -> torch.Tensor:
    """Mix identity likelihoods normalized within each available group.

    Normalizing over identities preserves learned frequency priors, including
    when only one group exists. Spare rows and unavailable input are not labels.
    """
    weight = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)
    available = ~embedder.retired
    assignments = weight[: len(embedder.vocab.master), available].float()
    conditional = assignments.log_softmax(dim=0)
    predicted = probabilities[:, available].float().clamp_min(1e-30).log()
    # Iterate the small group axis rather than materializing batch x vocabulary
    # x groups. Work remains vectorized over every identity and observation.
    result = predicted.new_full((predicted.shape[0], assignments.shape[0]), float("-inf"))
    for context, identity in zip(predicted.unbind(-1), conditional.unbind(-1), strict=True):
        result = torch.logaddexp(result, context.unsqueeze(-1) + identity.unsqueeze(0))
    return result


@group.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    group_type = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.int32(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    content_type = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string()),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    return pa.struct(
        [
            pa.field(TensorKey.group.name, group_type, nullable=False),
            pa.field(TensorKey.content.name, content_type, nullable=False),
        ]
    )


@group.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    group_type = datatype.field(TensorKey.group.name).type
    content_type = datatype.field(TensorKey.content.name).type
    embedder: Embedder = cast(Embedder, module.nodes[prediction.address].embedder)
    logits = cast(torch.Tensor, prediction.payload[TensorKey.group])
    group_probabilities = embedder.support(logits.reshape(-1, logits.shape[-1])).float().softmax(dim=-1)
    group_probability, group_ids = group_probabilities.max(dim=-1)
    group_values = struct(
        {
            TensorKey.value.name: array(group_ids, pa.int32()),
            TensorKey.probability.name: array(group_probability, pa.float32()),
        },
        group_type,
    )

    logits = vocabulary_logits(embedder, group_probabilities)
    vocabulary = labels(embedder.vocab)
    size = len(vocabulary)
    if size > logits.shape[-1]:
        raise ValueError(
            f"group vocabulary at {prediction.address!s} has {size} values but assignment width is {logits.shape[-1]}"
        )

    count = logits.shape[0]
    content_labels: pa.Array = pa.nulls(count, type=pa.large_string())
    content_probability = torch.zeros(count, dtype=torch.float32, device=logits.device)
    if size:
        probabilities = logits.softmax(dim=-1)
        content_probability, indices = probabilities.max(dim=-1)
        content_labels = pc.take(vocabulary, array(indices, pa.int64()))

    content_values = struct(
        {
            TensorKey.value.name: content_labels,
            TensorKey.probability.name: array(content_probability, pa.float32()),
        },
        content_type,
    )
    return struct(
        {
            TensorKey.group.name: group_values,
            TensorKey.content.name: content_values,
        },
        datatype,
    )


@torch.no_grad()
def reset_history(
    parameter: torch.Tensor,
    axis: int,
    index: int,
    optimizers: Sequence[torch.optim.Optimizer],
) -> None:
    """Forget replaced coordinates without changing scalar steps or neighboring history."""
    if parameter.grad is not None:
        parameter.grad.select(axis, index).zero_()
    for optimizer in optimizers:
        for value in optimizer.state.get(parameter, {}).values():
            if isinstance(value, torch.Tensor) and value.shape == parameter.shape:
                value.select(axis, index).zero_()


def adaptable(embedder: Embedder, decoder: Decoder) -> bool:
    """An adaptive transition must preserve every frozen assignment or representation."""
    return all(
        parameter.requires_grad
        for parameter in (
            embedder.embeddings[TensorKey.group.name].weight,
            embedder.embeddings[TensorKey.content.name].weight,
            decoder.linears[TensorKey.group.name].weight,
        )
    )


class GroupUsageCallback(Callback):
    """Commit globally weighted assignment evidence after every training batch."""

    @torch.no_grad()
    def on_train_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        outputs: Any,
        batch: Any,
        batch_idx: int,
    ) -> None:
        module = cast("Model", pl_module)
        # An objective-empty rank still joins every reduction. Loss functions
        # cannot provide that guarantee because the shared loop skips them.
        for address in sorted(module.schema.reconstruct):
            request = module.schema.requests[address]
            if not isinstance(request, Request):
                continue
            embedder = cast(Embedder, module.nodes[address].embedder)
            totals = all_reduce_sum(embedder.statistics.clone())
            embedder.statistics.zero_()
            update_usage(embedder, request, totals)


class GroupReviveCallback(Callback):
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module = cast("Model", pl_module)
        epoch = trainer.current_epoch
        targets: dict[Address, tuple[Embedder, Decoder, Request]] = {}
        for address, node in pl_module.nodes.items():
            request = pl_module.schema.requests.get(cast(Address, address))
            if not isinstance(request, Request) or request.revive_temperature <= 0.0:
                continue
            if request.n_groups[0] == request.n_groups[-1]:
                continue
            embedder = getattr(node, "embedder", None)
            decoder = getattr(node, "decoder", None)
            if isinstance(embedder, Embedder) and isinstance(decoder, Decoder) and adaptable(embedder, decoder):
                if decoder.metrics.compute(Strata.train)["targets.known"] > 0:
                    targets[cast(Address, address)] = (embedder, decoder, request)

        if not targets:
            return

        plans: dict[Address, dict[str, Any] | None] = {}
        if trainer.is_global_zero:
            for address, (embedder, _, request) in targets.items():
                plans[address] = self.plan(embedder, request, epoch=epoch)
        plans = broadcast_object(plans, src=0)

        for address, (embedder, decoder, request) in targets.items():
            self.apply(embedder, decoder, plans.get(address), optimizers=getattr(trainer, "optimizers", ()))

    @staticmethod
    def plan(embedder: "Embedder", request: "Request", *, epoch: int) -> dict[str, Any] | None:
        # Uncommitted candidates can still encode known identities. Revival may
        # replace only columns with no actual members, including retired slots.
        occupied = torch.zeros_like(embedder.committed)
        with embedder.vocab.lock:
            assignments = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)[
                : len(embedder.vocab.master)
            ]
            if assignments.numel():
                occupied[embedder.support(assignments).argmax(dim=-1)] = True
        dead = (~embedder.committed & ~occupied).nonzero(as_tuple=True)[0]
        n_dead = dead.numel()
        if n_dead == 0:
            return None

        committed_idx = embedder.committed.nonzero(as_tuple=True)[0]
        n_committed = committed_idx.numel()
        adherence = float(embedder.adherence_ema.item())
        warmup = epoch < _REVIVE_WARMUP

        signal = max(adherence, 1.0 if warmup else 0.0)
        if signal <= 0.0:
            return None

        base = math.exp(-epoch / request.revive_temperature)
        cap = 1.0 / n_dead
        p = min(base * signal, cap)
        if p <= 0.0:
            return None

        trials = torch.rand(n_dead)
        chosen_mask = trials < p
        if not bool(chosen_mask.any().item()):
            return None

        chosen = dead[chosen_mask].tolist()

        if n_committed > 0:
            committed_order = committed_idx[embedder.usage_ema[committed_idx].argsort(descending=True)]
            donors = [int(committed_order[i % n_committed].item()) for i in range(len(chosen))]
        else:
            donor = int(embedder.usage_ema.argmax().item())
            donors = [donor] * len(chosen)

        assign_w: torch.Tensor = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)
        content_w: torch.Tensor = cast(torch.Tensor, embedder.embeddings[TensorKey.content.name].weight)
        v_plus_1 = assign_w.shape[0]
        d_model = content_w.shape[0]
        noise = _REVIVE_NOISE
        n_chosen = len(chosen)
        return {
            "dead_ids": chosen,
            "donors": donors,
            "noise_assign": noise * torch.randn(n_chosen, v_plus_1),
            "noise_content": noise * torch.randn(n_chosen, d_model),
            "noise_group": noise * torch.randn(n_chosen, d_model),
        }

    @staticmethod
    @torch.no_grad()
    def apply(
        embedder: "Embedder",
        decoder: "Decoder",
        plan: dict[str, Any] | None,
        *,
        optimizers: Sequence[torch.optim.Optimizer] = (),
    ) -> None:
        if plan is None or not adaptable(embedder, decoder):
            return

        assign_w: torch.Tensor = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)
        content_w: torch.Tensor = cast(torch.Tensor, embedder.embeddings[TensorKey.content.name].weight)
        group_w: torch.Tensor = cast(torch.Tensor, decoder.linears[TensorKey.group.name].weight)
        donors = plan["donors"]
        device = assign_w.device
        dtype = assign_w.dtype

        for i, k in enumerate(plan["dead_ids"]):
            donor = donors[i]
            assign_w[:, k].copy_(assign_w[:, donor] + plan["noise_assign"][i].to(device=device, dtype=dtype))
            content_w[:, k].copy_(content_w[:, donor] + plan["noise_content"][i].to(device=device, dtype=dtype))
            group_w[k].copy_(group_w[donor] + plan["noise_group"][i].to(device=group_w.device, dtype=group_w.dtype))
            embedder.usage_ema[k] = embedder.usage_ema[donor] * 0.5
            embedder.usage_ema[donor] *= 0.5
            embedder.committed[k] = True
            embedder.retired[k] = False
            reset_history(assign_w, 1, k, optimizers)
            reset_history(content_w, 1, k, optimizers)
            reset_history(group_w, 0, k, optimizers)

        embedder.adherence_ema.zero_()


_MERGE_THRESHOLD: float = 0.95


class GroupMergeCallback(Callback):
    @torch.no_grad()
    def on_train_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        pl_module = cast("Model", pl_module)
        targets: dict[Address, tuple[Embedder, Decoder, Request]] = {}
        for address, node in pl_module.nodes.items():
            request = pl_module.schema.requests.get(cast(Address, address))
            if not isinstance(request, Request):
                continue
            if request.n_groups[0] == request.n_groups[-1]:
                continue
            embedder = getattr(node, "embedder", None)
            decoder = getattr(node, "decoder", None)
            if isinstance(embedder, Embedder) and isinstance(decoder, Decoder) and adaptable(embedder, decoder):
                if decoder.metrics.compute(Strata.train)["targets.known"] > 0:
                    targets[cast(Address, address)] = (embedder, decoder, request)

        if not targets:
            return

        plans: dict[Address, dict[str, int] | None] = {}
        if trainer.is_global_zero:
            for address, (embedder, decoder, request) in targets.items():
                plans[address] = self.plan(embedder, decoder, request)
        plans = broadcast_object(plans, src=0)

        for address, (embedder, decoder, _) in targets.items():
            self.apply(embedder, decoder, plans.get(address), optimizers=getattr(trainer, "optimizers", ()))

    @staticmethod
    def plan(embedder: "Embedder", decoder: "Decoder", request: "Request") -> dict[str, int] | None:
        committed_idx = embedder.committed.nonzero(as_tuple=True)[0]
        n_committed = committed_idx.numel()
        lower = request.n_groups[0]
        if n_committed <= lower or n_committed < 2:
            return None

        content_w = cast(torch.Tensor, embedder.embeddings[TensorKey.content.name].weight)  # (d_model, K)
        group_w = cast(torch.Tensor, decoder.linears[TensorKey.group.name].weight)  # (K, d_model)

        content_cols = content_w[:, committed_idx]
        group_rows = group_w[committed_idx, :]
        content_norm = content_cols / content_cols.norm(dim=0).clamp_min(1e-12)
        group_norm = group_rows / group_rows.norm(dim=1, keepdim=True).clamp_min(1e-12)

        content_sim = content_norm.T @ content_norm
        group_sim = group_norm @ group_norm.T
        joint_sim = torch.minimum(content_sim, group_sim)
        joint_sim.fill_diagonal_(-1.0)

        max_sim, max_j = joint_sim.max(dim=1)
        overall_max, overall_i = max_sim.max(dim=0)
        if float(overall_max.item()) <= _MERGE_THRESHOLD:
            return None

        i_local = int(overall_i.item())
        j_local = int(max_j[i_local].item())
        i_global = int(committed_idx[i_local].item())
        j_global = int(committed_idx[j_local].item())

        if embedder.usage_ema[i_global] < embedder.usage_ema[j_global]:
            loser, winner = i_global, j_global
        else:
            loser, winner = j_global, i_global
        return {"loser": loser, "winner": winner}

    @staticmethod
    @torch.no_grad()
    def apply(
        embedder: "Embedder",
        decoder: "Decoder",
        plan: dict[str, int] | None,
        *,
        optimizers: Sequence[torch.optim.Optimizer] = (),
    ) -> None:
        if plan is None or not adaptable(embedder, decoder):
            return

        loser = plan["loser"]
        winner = plan["winner"]
        assign_w = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)
        content_w = cast(torch.Tensor, embedder.embeddings[TensorKey.content.name].weight)
        group_w = cast(torch.Tensor, decoder.linears[TensorKey.group.name].weight)
        assign_w[:, winner].copy_(torch.logaddexp(assign_w[:, winner], assign_w[:, loser]))
        # Keep stored logits finite so a zero retired probability cannot produce 0 * -inf.
        floor = (assign_w.amin(dim=1) - 30).clamp_min(torch.finfo(assign_w.dtype).min)
        assign_w[:, loser].copy_(floor)
        content_w[:, loser].zero_()
        group_w[loser].zero_()
        embedder.usage_ema[winner] = embedder.usage_ema[winner] + embedder.usage_ema[loser]
        embedder.usage_ema[loser] = 0.0
        embedder.committed[loser] = False
        embedder.retired[loser] = True
        reset_history(assign_w, 1, winner, optimizers)
        reset_history(assign_w, 1, loser, optimizers)
        reset_history(content_w, 1, loser, optimizers)
        reset_history(group_w, 0, loser, optimizers)


group.callback(GroupUsageCallback, GroupReviveCallback, GroupMergeCallback, Refinement)
