"""
Groups is motivated as a minimal parameter implementation which enables RF to learn
patterns and associations across many UUIDs. This is most easily conceived to be used
to model an entity identifier, and recommended for use over comparable online-
vocabulary module implementations such as `category` if and when there are too
many UUIDs in the vocabulary to justify modeling each separately, as it relates to
parameter explosion. Secondarily, `group` becomes more and more valuable when there is
not enough data per unique ID or field value otherwise to learn perfectly customized
representations; instead, `group` attempts to identify which groupings of the observed
values are most effectively able to improve downstream modeling through a shared group
representation.
"""

from __future__ import annotations

from collections.abc import Generator, Mapping
from contextlib import AbstractContextManager, contextmanager
from dataclasses import InitVar
from typing import TYPE_CHECKING, Annotated, Any, Literal, Unpack, cast
from weakref import ReferenceType, ref

import numpy as np
import pyarrow as pa
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
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
from relflow.tensorfields.output import array, struct
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.resize import Resize
from relflow.tensorfields.shared.rows import Embedding
from relflow.tensorfields.shared.vocabulary import (
    OnlineVocabularyModel,
    VocabularyBatch,
    VocabularyState,
    VocabularySyncCallback,
    admissible,
    capacity,
    states,
)

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch


group: Extension = Extension(
    name="group",
    types=(bool, int, float, str, bytes),
    callbacks=(VocabularySyncCallback, CounterUpdateCallback),
)


_OVERRIDE_LOCK: str = "group assignment overrides"

Assignment = int


@group.register(Component.Request)
class Request(RequestBase):
    """Scalar labels represented through shared, learned latent groups.

    Training grows label storage automatically. ``p_unavailable``
    routes known training labels through shared unavailable content.
    Each identity stores one logit per group. Embedding selects the argmax as a
    one-hot vector, with straight-through softmax gradients during training and
    no sampling noise. Assignment logits can learn through downstream objectives
    when the field is visible as context. A reconstructing mask trains the state
    head only; the group head has no built-in reconstruction objective.
    Predictions expose the argmax of the decoder's group softmax and its
    probability, without vocabulary labels.
    """

    model_config = pydantic.ConfigDict(populate_by_name=True, serialize_by_alias=True)

    type: Literal["group"] = "group"
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01

    n_groups: Annotated[
        int,
        pydantic.Field(
            description="A single int to specify an exact number of groups.",
        ),
    ]

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            n_groups: int,
            p_unavailable: float = 0.01,
            type: Literal["group"] = "group",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @pydantic.model_validator(mode="after")
    def check_n_groups(self):
        if self.n_groups <= 1:
            raise ValueError("n_groups must be greater than 1")
        return self

    @classmethod
    def assign(
        cls,
        model: "Model",
        address: Address | str,
        token: Any,
        assignment: int,
        /,
    ) -> int:
        """Persistently assign one token to an integer group index."""
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

        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        probabilities = logits.softmax(dim=-1)
        groups: list[int] = probabilities.argmax(dim=-1).cpu().tolist()
        rows: list[list[float]] = probabilities.cpu().tolist()
        return {
            token: {
                "group": group_id,
                "probabilities": tuple(float(probability) for probability in row),
            }
            for token, group_id, row in zip(vocabulary, groups, rows, strict=True)
        }

    @classmethod
    def override(
        cls,
        model: "Model",
        address: Address | str,
        assignments: Mapping[Any, int],
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

    values, nan = admissible(field)
    content = torch.from_numpy(state.indices(values, learn=True))
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(states(field, nan)), len(Tokens)),
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

        def encode(field: RaggedField) -> tuple[torch.Tensor, torch.Tensor]:
            if not len(field.values):
                return torch.zeros(field.shape, dtype=torch.int64), torch.from_numpy(field.dense.copy())
            if state is None:
                raise RuntimeError(f"group field at '{address}' requires a vocabulary encoding context")
            values, nan = admissible(field)
            tokens = np.full(len(nan), -1, dtype=np.int64)
            tokens[~nan] = state.indices(values, learn=False)
            return torch.from_numpy(field.place(tokens, fill=0)), torch.from_numpy(states(field, nan))

        content, state_tensor = encode(input)
        targets = TensorDict({}, batch_size=input.shape)
        if address in schema.objectives:
            target_content, target_state = encode(target)
            targets = TensorDict(
                {TensorKey.state: target_state, TensorKey.content: target_content}, batch_size=input.shape
            )

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
            targets=targets,
            batch_size=input.batch_size,
        )


@group.register(Component.Embedder)
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address
        self.size: int = request.n_groups

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

    @property
    def capacity(self) -> int:
        return self.vocab.size

    def resize(self, size: int, resize: Resize) -> None:
        """Expand identity rows without changing group columns or their state."""
        resize.embedding(cast(torch.nn.Embedding, self.embeddings[TensorKey.group.name]), size + 1, tail=True)
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

        safe_content = content.masked_fill(content.eq(-1), self.capacity).masked_fill(~valued, 0)
        embedding_values: torch.Tensor = self.embeddings[TensorKey.group.name](safe_content)
        hard = torch.zeros_like(embedding_values)
        hard.scatter_(-1, embedding_values.argmax(dim=-1, keepdim=True), 1.0)
        if self.training:
            soft = torch.softmax(embedding_values, dim=-1)
            group_assignments = hard.detach() - soft.detach() + soft
        else:
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

    def assignment(self, assignment: Assignment) -> torch.Tensor:
        weight = cast(torch.Tensor, self.embeddings[TensorKey.group.name].weight)
        if isinstance(assignment, bool) or not isinstance(assignment, int):
            raise TypeError(
                f"group assignment at {self.origin!s} must be an integer group index, got {type(assignment).__name__}"
            )
        if not 0 <= assignment < self.size:
            raise ValueError(f"group must be in [0, {self.size}); got {assignment}")
        values = weight.new_zeros(self.size)
        values[assignment] = 1.0
        return values

    def assign_token(self, token: Any, assignment: Assignment) -> int:
        values = self.assignment(assignment)
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
            weight[index].copy_(values)
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
    """Grow identity vectors while preserving group columns and the unavailable row."""
    if not isinstance(binding, VocabularyBatch):
        raise TypeError(f"group binding at '{address}' requires VocabularyBatch, got {type(binding).__name__}")
    embedder = cast(Embedder, module.nodes[address].embedder)
    mapping, size = binding.resolve(embedder.vocab, learn=strata == Strata.train, device=module.device, resize=resize)
    embedder.resize(size, resize)
    field = cast(TensorField, field)
    field.content = binding.translate(field.content, mapping, address=address)
    if TensorKey.content in field.targets:
        values = cast(torch.Tensor, field.targets[TensorKey.content])
        field.targets[TensorKey.content] = binding.translate(values, mapping, address=address)
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
            embedder.assignment(assignment)
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
                embedder.assignment(assignment)
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
    coordinatewise = True

    def __init__(self, schema: Schema, address: Address):
        # Group assignments belong to stable identities. Row-varying sibling
        # values may supervise them through downstream objectives, but must not
        # alter the query used to reconstruct the identity itself.
        super().__init__(schema=schema, address=address, conditioned=False)

        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.group.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=cast(Request, schema.requests[address]).n_groups,
                    bias=False,
                ),
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
    N: int = cast(torch.Tensor, batch.targets[TensorKey.state]).numel()
    trainable = batch.trainable.reshape(N)

    state_inputs = cast(torch.Tensor, prediction.payload[TensorKey.state]).reshape(-1, len(Tokens))
    state_targets = cast(torch.Tensor, batch.targets[TensorKey.state]).reshape(N)
    selected_inputs = state_inputs[trainable]
    selected_targets = state_targets[trainable]
    state_losses = torch.nn.functional.cross_entropy(selected_inputs, selected_targets, reduction="none")

    loss: torch.Tensor = module.track(
        (prediction.address, strata, Metric.loss, TensorKey.state),
        value=state_losses.mean() if state_losses.numel() else state_losses.sum(),
    )

    if state_losses.numel():
        module.track(
            (prediction.address, strata, Metric.accuracy, TensorKey.state),
            value=selected_inputs.argmax(dim=1).eq(selected_targets).float().mean(),
        )
    module.track(
        (prediction.address, strata, "vocabulary", "size"),
        value=state_inputs.new_tensor(len(embedder.vocab.master), dtype=torch.float32),
    )

    # content_targets = cast(torch.Tensor, batch.targets[TensorKey.group]).reshape(N)

    # group_prediction = cast(torch.Tensor, prediction.payload[TensorKey.group]).reshape(N, -1)

    # TODO: Consider whether MSE enforces semantic clustering. If so:
    # TODO: set up the only content loss to be a MSE. Document that MSE is chosen to encourage semantic clustering

    # TODO: Log the frequency of the majority class' frequency across batches
    # TODO: Log distribution statistics which encapsulate how spread out // concentrated the clusters are, overall

    # valued: torch.Tensor = trainable & state_targets.eq(Tokens.valued.value)

    # assign_weight: torch.Tensor = cast(torch.Tensor, embedder.embeddings[TensorKey.group.name].weight)
    # vocab_size = len(embedder.vocab.master)
    # known = valued & content_targets.ge(0) & content_targets.lt(vocab_size)

    # safe_content = content_inputs.masked_fill(content_inputs.eq(-1), self.capacity).masked_fill(~valued, 0)

    # embedding_values = embedder.embeddings[TensorKey.group.name](safe_content)
    # radial_distance: torch.Tensor = (
    #     embedding_values.fmod(2*torch.pi)
    #     -
    #     torch.arange(0, 2*torch.pi, step=2*torch.pi / embedder.size).reshape(1, -1)
    # ).abs()

    # group_probs = group_logits[known].float().softmax(dim=-1)
    # vocab_logits = group_probs @ assign_weight[:vocab_size].float().T

    # objective = torch.nn.functional.cross_entropy(vocab_logits, content_targets[known], reduction="sum")
    # loss = loss + objective / known.sum().clamp_min(1)

    return loss


@group.register(Component.output)
def output(module: Model, address: Address) -> pa.StructType:
    group_type = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.int32(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    return pa.struct([pa.field(TensorKey.group.name, group_type, nullable=False)])


@group.register(Component.write)
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    group_type = datatype.field(TensorKey.group.name).type
    embedder: Embedder = cast(Embedder, module.nodes[prediction.address].embedder)
    values = cast(torch.Tensor, prediction.payload[TensorKey.group]).detach()
    if values.ndim == 0 or values.shape[-1] != embedder.size:
        raise ValueError(
            f"group prediction at {prediction.address!s} requires {embedder.size} logits per coordinate; "
            f"got shape {tuple(values.shape)}"
        )
    if values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    group_probabilities = values.reshape(-1, embedder.size).softmax(dim=-1)
    group_probability, group_ids = group_probabilities.max(dim=-1)
    group_values = struct(
        {
            TensorKey.value.name: array(group_ids, pa.int32()),
            TensorKey.probability.name: array(group_probability, pa.float32()),
        },
        group_type,
    )

    return struct({TensorKey.group.name: group_values}, datatype)
