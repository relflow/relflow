from __future__ import annotations

from collections.abc import Mapping
from dataclasses import InitVar
from typing import TYPE_CHECKING, Annotated, Any, Literal, Unpack, cast

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pydantic
import torch
from beartype import beartype
from tensordict import TensorDict, tensorclass

from relflow.data.ragged import RaggedField
from relflow.structs.enums import Metric, Strata, TensorKey, Tokens
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
from relflow.tensorfields.output import array, labels, struct, variable
from relflow.tensorfields.shared.counter import Counter, CounterUpdateCallback, tally
from relflow.tensorfields.shared.reconstruction import Metrics, Scores, Totals
from relflow.tensorfields.shared.rows import Embedding, Linear
from relflow.tensorfields.shared.vocabulary import (
    INITIAL_CAPACITY,
    OnlineVocabularyModel,
    VocabularyBatch,
    VocabularyState,
    VocabularySyncCallback,
)

if TYPE_CHECKING:
    from relflow.architecture.root import Model
    from relflow.data.datasets.base import InterprocessEncodingContext
    from relflow.helpers.resize import Resize
    from relflow.structs.experiment import Schema
    from relflow.structs.structure import Branch

sets: Extension = Extension(name="set", types=(bool, int, float, str, bytes))
sets.callback(VocabularySyncCallback, CounterUpdateCallback, Metrics)


@sets.register
class Request(RequestBase):
    """Unordered label membership represented by an online vocabulary.

    Training grows vocabulary storage automatically. Empty
    sets are valued; nulls have a separate state. Unknown members share a
    learned unavailable representation; their identities cannot be recovered.
    ``p_unavailable`` independently drops known positive input labels during
    training without changing targets. ``mask=True`` makes the set a supervised reconstruction target.
    Predictions contain label/probability pairs for populated vocabulary
    entries. ``threshold`` in ``[0, 1]`` filters those pairs; ``None`` keeps
    all entries. This filter leaves the training loss and metrics unchanged.
    """

    type: Literal["set"] = "set"
    p_unavailable: Annotated[float, pydantic.Field(ge=0.0, le=1.0, default=0.01)] = 0.01
    threshold: Annotated[float | None, pydantic.Field(ge=0.0, le=1.0, default=None)] = None

    if TYPE_CHECKING:

        def __init__(
            self,
            *,
            p_unavailable: float = 0.01,
            threshold: float | None = None,
            type: Literal["set"] = "set",
            **options: Unpack[FieldOptions],
        ) -> None: ...

    @classmethod
    def vocabulary(
        cls,
        source: "Model | InterprocessEncodingContext",
        address: Address | str,
        /,
    ) -> tuple[Any, ...]:
        """Return an immutable snapshot of one set vocabulary."""
        from relflow.architecture.root import Model

        address = Address(str(address))
        if isinstance(source, Model):
            if address not in source.nodes:
                raise KeyError(f"no field at address {str(address)!r}")

            embedder = getattr(source.nodes[address], "embedder", None)
            if not isinstance(embedder, Embedder):
                raise TypeError(f"address {str(address)!r} is not a Set field (got {type(embedder).__name__})")
            with embedder.vocab.lock:
                return tuple(embedder.vocab.master)
        elif isinstance(source, Mapping):
            if address not in source:
                raise KeyError(f"no encoding context at address {str(address)!r}")

            state = source[address]
            if not isinstance(state, VocabularyState):
                raise TypeError(
                    f"encoding context at {str(address)!r} is not a VocabularyState (got {type(state).__name__})"
                )
        else:
            raise TypeError(
                f"Set.vocabulary source must be a Model or InterprocessEncodingContext, got {type(source).__name__}"
            )

        with state.lock:
            return tuple(state.master)


@sets.register
def observe(
    field: RaggedField,
    *,
    address: Address,
    schema: Schema,
    state: object | None,
    learn: bool,
) -> TensorDict | None:
    """Reserve and count the complete pristine set exposure."""

    if not learn:
        return None
    if not isinstance(state, VocabularyState):
        raise RuntimeError(f"set field at '{address}' requires a vocabulary encoding context")

    values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
    if pa.types.is_list(values.type) or pa.types.is_large_list(values.type) or pa.types.is_fixed_size_list(values.type):
        values = pc.list_flatten(values)
    if values.null_count:
        values = pc.filter(values, pc.is_valid(values))
    indices = state.indices(values, learn=True)
    return TensorDict(
        {
            TensorKey.state: tally(torch.from_numpy(field.dense.copy()), len(Tokens)),
            TensorKey.content: tally(torch.from_numpy(indices.copy()), state.size),
        },
        batch_size=[],
    )


@sets.register
@tensorclass
class TensorField(TensorFieldBase[TensorDict]):
    state: torch.Tensor
    content: TensorDict
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
        request: Request = cast(Request, schema.requests[address])
        state = context.state
        if state is not None and not isinstance(state, VocabularyState):
            raise TypeError(f"set field at '{address}' requires VocabularyState context, got {type(state).__name__}")
        n_tokens = state.size if state is not None else INITIAL_CAPACITY

        def encode(field: RaggedField) -> TensorDict:
            values = field.values.combine_chunks() if isinstance(field.values, pa.ChunkedArray) else field.values
            encoded = np.zeros((len(values), n_tokens), dtype=np.float32)
            unavailable = np.zeros(len(values), dtype=np.int64)
            if len(values) and state is None:
                raise RuntimeError(f"set field at '{address}' requires a vocabulary encoding context")
            if (
                pa.types.is_list(values.type)
                or pa.types.is_large_list(values.type)
                or pa.types.is_fixed_size_list(values.type)
            ):
                parents = pc.list_parent_indices(values)
                flattened = pc.list_flatten(values)
            else:
                parents = pa.array(np.arange(len(values), dtype=np.int64))
                flattened = values
            if flattened.null_count:
                valid = pc.is_valid(flattened)
                flattened = pc.filter(flattened, valid)
                parents = pc.filter(parents, valid)
            indices = state.indices(flattened, learn=False) if state is not None else np.empty(0, dtype=np.int64)
            if len(indices):
                rows = parents.to_numpy(zero_copy_only=False)
                known = (indices >= 0) & (indices < n_tokens)
                encoded[rows[known], indices[known]] = 1.0
                if not known.all():
                    # Count distinct unknown members, not occurrences: order
                    # and duplicates must not change a set's representation.
                    unknown = cast(
                        pa.DictionaryArray, pc.dictionary_encode(pc.filter(flattened, pa.array(~known)))
                    ).indices.to_numpy()
                    pairs = np.unique(np.column_stack((rows[~known], unknown)), axis=0)
                    unavailable = np.bincount(pairs[:, 0], minlength=len(values))
            return TensorDict(
                {
                    "membership": torch.from_numpy(field.place(encoded, fill=0.0, value_shape=(n_tokens,))),
                    "unavailable": torch.from_numpy(field.place(unavailable, fill=0)),
                },
                batch_size=field.shape,
            )

        state_tensor = torch.from_numpy(input.dense)
        content = encode(input)
        target_content = encode(target)

        if strata == Strata.train and request.p_unavailable > 0.0:
            membership = cast(torch.Tensor, content["membership"])
            selected = torch.rand_like(membership).lt(request.p_unavailable) & membership.bool()
            content["membership"] = membership.masked_fill(selected, 0.0)
            content["unavailable"] += selected.sum(dim=-1)

        return cls(
            state=state_tensor,
            content=content,
            present=present,
            trainable=trainable,
            inferred=inferred,
            targets=TensorDict(
                {
                    TensorKey.state: torch.from_numpy(target.dense),
                    TensorKey.content: target_content,
                },
                batch_size=input.shape,
            ),
            batch_size=input.batch_size,
        )


@sets.register
class Embedder(EmbedderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        request: Request = cast(Request, schema.requests[address])
        self.origin: Address = address
        self.destination: Address = cast("Branch", request.parent).address

        self.vocab: OnlineVocabularyModel = OnlineVocabularyModel()
        # A neutral fallback until real or simulated unavailable inputs teach
        # it; absence is not a randomly initialized pseudo-label.
        self.unavailable = torch.nn.Parameter(torch.zeros(schema.d_model))

        self.embeddings = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Embedding(
                    num_embeddings=len(Tokens),
                    embedding_dim=schema.d_model,
                ),
                TensorKey.content.name: Embedding(
                    num_embeddings=self.vocab.size,
                    embedding_dim=schema.d_model,
                ),
            }
        )
        self.counters = torch.nn.ModuleDict(
            {
                TensorKey.state.name: Counter(address=address, size=len(Tokens)),
                TensorKey.content.name: Counter(address=address, size=self.vocab.size),
            }
        )

    @property
    def size(self) -> int:
        return self.vocab.size

    @beartype
    def forward(self, inputs: TensorInput) -> Parcel:
        N: int
        dims: list[int]

        content = cast(TensorDict, inputs.content)
        membership = cast(torch.Tensor, content["membership"])
        N, *dims, n_tokens = membership.shape
        if n_tokens > self.size:
            raise ValueError(f"Set in address {self.origin} has invalid vocabulary width")

        state = inputs.state.reshape(-1)
        membership = membership.reshape(-1, n_tokens)
        valued = state.eq(Tokens.valued.value)

        weights = cast(torch.nn.Embedding, self.embeddings[TensorKey.content.name]).weight[:n_tokens]
        counts = membership.sum(dim=-1, keepdim=True).clamp_min(1.0)
        content_embedding = membership.to(dtype=weights.dtype).matmul(weights) / counts
        unavailable = cast(torch.Tensor, content["unavailable"]).reshape(-1, 1).gt(0)
        content_embedding = content_embedding + unavailable * self.unavailable

        embeddings: torch.Tensor = (
            self.embeddings[TensorKey.state.name](state) + content_embedding * valued.unsqueeze(-1)
        ).reshape(N, *dims, -1)

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


@sets.register
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
    """Move membership columns into the vocabulary current at batch consumption."""
    if not isinstance(binding, VocabularyBatch):
        raise TypeError(f"set binding at '{address}' requires VocabularyBatch, got {type(binding).__name__}")
    node = module.nodes[address]
    embedder = cast(Embedder, node.embedder)
    mapping, size = binding.resolve(embedder.vocab, learn=strata == Strata.train, device=module.device, resize=resize)
    resize.embedding(cast(torch.nn.Embedding, embedder.embeddings[TensorKey.content.name]), size)
    cast(Counter, embedder.counters[TensorKey.content.name]).resize(size, resize)
    decoder = getattr(node, "decoder", None)
    if decoder is not None:
        resize.linear(decoder.linears[TensorKey.content.name], size)
    field = cast(TensorField, field)
    for content in (field.content, cast(TensorDict, field.targets[TensorKey.content])):
        previous = cast(torch.Tensor, content["membership"])
        indices = mapping[:-1].to(previous.device)
        known = indices.ge(0) & indices.lt(size)
        membership = previous.new_zeros((*previous.shape[:-1], size))
        membership.index_add_(-1, indices[known], previous[..., known])
        content["membership"] = membership
        content["unavailable"] = cast(torch.Tensor, content["unavailable"]) + previous[..., ~known].sum(dim=-1).long()
    if observation is not None:
        observation[TensorKey.content] = binding.counts(
            cast(torch.Tensor, observation[TensorKey.content]), mapping, size
        )


@sets.register
def learn(
    module: Model,
    observation: TensorDict,
    *,
    address: Address,
    strata: Strata,
) -> None:
    """Apply pristine set-state counts to the model resource."""

    if strata != Strata.train:
        raise ValueError(f"set learner at '{address}' requires train strata, got {strata}")
    embedder: Embedder = cast(Embedder, module.nodes[address].embedder)
    cast(Counter, embedder.counters[TensorKey.state.name]).learn(cast(torch.Tensor, observation[TensorKey.state]))
    cast(Counter, embedder.counters[TensorKey.content.name]).learn(cast(torch.Tensor, observation[TensorKey.content]))


class Membership(Totals):
    """Set-member coverage and micro scores over populated vocabulary entries."""

    def __init__(self):
        super().__init__(counts=9, sums=1)

    def compute(self) -> dict[str, torch.Tensor]:
        known, unavailable, sets, complete, bits, correct, exact, tp, predicted = self.counts.unbind()
        return {
            "targets.known": known,
            "targets.unavailable": unavailable,
            "coverage.content": known / (known + unavailable),
            "sets.valued": sets,
            "sets.complete": complete,
            "coverage.set": complete / sets,
            "accuracy.content": correct / bits,
            "accuracy.set": exact / sets,
            "precision.content": tp / predicted,
            "recall.content": tp / known,
            "recall.all": tp / (known + unavailable),
            "f1.content": 2 * tp / (predicted + known),
            "f1.all": 2 * tp / (predicted + known + unavailable),
            "loss.content": self.sums[0] / bits,
        }


@sets.register
class Decoder(DecoderBase):
    def __init__(self, schema: Schema, address: Address):
        super().__init__(schema=schema, address=address)

        self.metrics = Scores(address, Membership)
        self.linears = torch.nn.ModuleDict(
            {
                TensorKey.state.name: torch.nn.Linear(
                    in_features=schema.d_model,
                    out_features=len(Tokens),
                ),
                TensorKey.content.name: Linear(
                    in_features=schema.d_model,
                    out_features=INITIAL_CAPACITY,
                ),
            }
        )

    @beartype
    def decode(self, pooled: torch.Tensor) -> TensorDict:
        return TensorDict(
            source={
                TensorKey.state: self.linears[TensorKey.state.name](pooled),
                TensorKey.content: self.linears[TensorKey.content.name](pooled),
            }
        )


@sets.register
def loss(
    module: Model,
    prediction: Prediction,
    batch: TensorFieldBase,
    strata: Strata,
) -> torch.Tensor:
    embedder: Embedder = cast(Embedder, module.nodes[prediction.address].embedder)
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

    valued = trainable & state_targets.eq(Tokens.valued.value)
    size = len(embedder.vocab.master)
    content_inputs = cast(torch.Tensor, prediction.payload[TensorKey.content]).reshape(N, -1)[valued, :size].float()
    targets = cast(TensorDict, batch.targets[TensorKey.content])
    memberships = cast(torch.Tensor, targets["membership"]).reshape(N, -1)
    if memberships.shape[-1] < size:
        memberships = torch.nn.functional.pad(memberships, (0, size - memberships.shape[-1]))
    content_targets = memberships[valued, :size]
    unavailable = cast(torch.Tensor, targets["unavailable"]).reshape(N)[valued]
    # Complete set annotations supervise all populated memberships, including
    # known negatives in partially/all-OOV sets. Unallocated slots are not
    # negative examples of future labels.
    objective = torch.nn.functional.binary_cross_entropy_with_logits(content_inputs, content_targets, reduction="sum")
    decoder = cast(Decoder, module.nodes[prediction.address].decoder)
    metric = cast(Membership, decoder.metrics[f"{strata.value}_metrics"])
    with torch.no_grad():
        expected = content_targets.bool()
        predicted = content_inputs.ge(0)
        correct = predicted.eq(expected)
        complete = unavailable.eq(0)
        counts = torch.stack(
            (
                expected.sum(),
                unavailable.sum(),
                valued.sum(),
                complete.sum(),
                valued.sum() * size,
                correct.sum(),
                (correct.all(dim=-1) & complete).sum(),
                (predicted & expected).sum(),
                predicted.sum(),
            )
        )
        metric.update(counts, objective.unsqueeze(0))
    module.track(
        (prediction.address, strata, "vocabulary", "size"), value=state_inputs.new_tensor(size, dtype=torch.float32)
    )
    return loss + objective / max(content_inputs.numel(), 1)


@sets.register
def output(module: Model, address: Address) -> pa.StructType:
    candidate = pa.struct(
        [
            pa.field(TensorKey.value.name, pa.large_string(), nullable=False),
            pa.field(TensorKey.probability.name, pa.float32(), nullable=False),
        ]
    )
    return pa.struct([pa.field(TensorKey.content.name, pa.list_(candidate), nullable=False)])


@sets.register
def write(module: Model, prediction: Prediction, datatype: pa.StructType) -> pa.StructArray:
    content_type = datatype.field(TensorKey.content.name).type
    candidate_type = content_type.value_type
    logits = cast(torch.Tensor, prediction.payload[TensorKey.content])
    coordinates = logits.reshape(-1, logits.shape[-1])
    vocabulary = labels(cast(Embedder, module.nodes[prediction.address].embedder).vocab)
    size = len(vocabulary)
    if size > coordinates.shape[-1]:
        raise ValueError(
            f"set vocabulary at {prediction.address!s} has {size} values but prediction width is "
            f"{coordinates.shape[-1]}"
        )

    probabilities = coordinates[:, :size].sigmoid()
    request: Request = cast(Request, module.schema.requests[prediction.address])
    if request.threshold is None:
        counts = torch.full((coordinates.shape[0],), size, dtype=torch.int64)
        indices = torch.arange(size, device=logits.device).expand(coordinates.shape[0], size).reshape(-1)
        selected = probabilities.reshape(-1)
    else:
        keep = probabilities.ge(request.threshold)
        counts = keep.sum(dim=-1, dtype=torch.int64)
        indices = keep.nonzero(as_tuple=False)[:, 1]
        selected = probabilities[keep]

    candidates = struct(
        {
            TensorKey.value.name: pc.take(vocabulary, array(indices, pa.int64())),
            TensorKey.probability.name: array(selected, pa.float32()),
        },
        candidate_type,
    )
    return struct({TensorKey.content.name: variable(candidates, counts)}, datatype)
