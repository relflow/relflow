from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.helpers.resize import Resize
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Prediction
from relflow.structs.tree import Mask
from relflow.tensorfields.base import TENSORFIELDS, Context
from relflow.tensorfields.extensions.group import (
    Decoder,
    Embedder,
    GroupMergeCallback,
    GroupReviveCallback,
    GroupUsageCallback,
    TensorField,
    loss,
    write,
)
from relflow.tensorfields.extensions.group import (
    output as output_type,
)
from relflow.tensorfields.shared.counter import Counter
from relflow.tensorfields.shared.vocabulary import OnlineVocabularyModel, VocabularyState
from tests.arrow import table as arrow_table
from tests.tensorfields.helpers import tensorize

ADDRESS = "/items/group"


@pytest.mark.parametrize("n_groups, normalized", [(3, (3, 3)), ((2, 4), (2, 4))])
def test_group_configuration_has_one_normalized_n_groups_field(n_groups, normalized):
    field = rf.Group(n_groups=n_groups)
    assert field.n_groups == normalized
    assert field.model_dump()["n_groups"] == normalized
    assert not hasattr(field, "n_clusters")
    assert not hasattr(field, "bounds")
    assert set(rf.Group.model_fields).isdisjoint({"n_clusters", "bounds"})
    for obsolete in ("n_clusters", "bounds"):
        with pytest.raises(pydantic.ValidationError, match=obsolete):
            rf.Group.model_validate({"n_groups": n_groups, obsolete: n_groups})
        with pytest.raises(pydantic.ValidationError, match=obsolete):
            rf.Group.model_validate({obsolete: n_groups})


def test_group_checkpoint_and_output_preserve_user_field_names(tmp_path):
    model = rf.Model.xs(d_model=8, fields={"cluster": rf.Group(n_groups=2, mask=True)})
    inputs = pa.table({"cluster": ["a", "b"]})
    model.encode(inputs, strata="train")
    path = model.save(tmp_path / "group.ckpt")
    restored = rf.Model.load(path)
    assert restored.schema.model_dump() == model.schema.model_dump()
    assert restored.schema.requests["/cluster"].n_groups == (2, 2)
    assert restored.schema.requests["/cluster"].type == "group"
    predictions = restored.predict(inputs)["predictions"].combine_chunks().field("cluster")
    assert {field.name for field in predictions.type} == {"state", "group", "content", "inferred"}
    assert set(rf.Group.assignments(restored, "/cluster")["a"]) == {"group", "probabilities"}

    checkpoint = torch.load(path, weights_only=False)
    field = checkpoint["schema"]["fields"]["fields"][0]
    assert field["n_groups"] == [2, 2]
    assert set(field).isdisjoint({"n_clusters", "bounds"})
    field["bounds"] = field.pop("n_groups")
    torch.save(checkpoint, path)
    with pytest.raises(pydantic.ValidationError, match="bounds"):
        rf.Model.load(path)
    field["n_groups"] = field.pop("bounds")
    checkpoint["schema"]["fields"]["fields"][0]["type"] = "cluster"
    torch.save(checkpoint, path)
    with pytest.raises(pydantic.ValidationError, match="unknown tensor field type: cluster"):
        rf.Model.load(path)


def build_embedder(schema):
    """Allocate rows for the synthetic token IDs used by component-level tests."""
    embedder = Embedder(schema=schema, address=ADDRESS)
    resize = Resize()
    embedder.resize(8, resize)
    resize.commit()
    return embedder


def build_schema(
    *,
    n_groups: int | list[int] | tuple[int, int] = (2, 4),
    p_unavailable: float | None = None,
    revive_temperature: float | None = None,
    ema_decay: float | None = None,
    mask: bool | Mask = False,
) -> Schema:
    options = {}
    if p_unavailable is not None:
        options["p_unavailable"] = p_unavailable
    if revive_temperature is not None:
        options["revive_temperature"] = revive_temperature
    if ema_decay is not None:
        options["ema_decay"] = ema_decay
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.1,
        items=rf.Branch(length=2, group=rf.Group(n_groups=n_groups, mask=mask, **options)),
    )


def vocabulary_state(size: int = 8):
    return OnlineVocabularyModel(size=size).state


def tensorfield(
    rows: list[list[Any]],
    *,
    schema: Schema,
    strata: Strata,
    state: VocabularyState,
) -> TensorField:
    batch = arrow_table([{"items": [{"group": value} for value in row]} for row in rows])
    projection = coalesce(batch, schema=schema, strata=strata).fields[ADDRESS]
    return tensorize(
        TensorField,
        projection,
        TENSORFIELDS["group"],
        address=ADDRESS,
        schema=schema,
        strata=strata,
        context=Context(state=state),
    )


# ---------- TensorField ----------


def test_group_tensorfield_routes_oov_to_sentinel_at_validate():
    schema = build_schema(p_unavailable=0.0)
    state = vocabulary_state()

    tensorfield(
        rows=[["ALPHA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )
    field = tensorfield(
        rows=[["OMEGA"]],
        schema=schema,
        strata=Strata.validate,
        state=state,
    )

    sentinel = -1
    assert torch.equal(
        field.state,
        torch.tensor([[[Tokens.valued.value]]], dtype=torch.int64),
    )
    assert torch.equal(
        field.content,
        torch.tensor([[[sentinel]]], dtype=torch.int64),
    )


def test_group_tensorfield_simulates_unavailable_during_training():
    schema = build_schema(p_unavailable=1.0)
    state = vocabulary_state()

    field = tensorfield(
        rows=[["ALPHA", None], ["BETA"]],
        schema=schema,
        strata=Strata.train,
        state=state,
    )

    sentinel = -1
    assert torch.equal(
        field.content,
        torch.tensor(
            [
                [[sentinel, 0]],
                [[sentinel, 0]],
            ],
            dtype=torch.int64,
        ),
    )


# ---------- Embedder ----------


def test_group_embedder_train_forward_uses_the_same_identity_assignment():
    schema = build_schema(n_groups=4, p_unavailable=0.0)
    embedder = build_embedder(schema)
    embedder.train()

    with torch.no_grad():
        # Flat logits still choose the same group for the same identity.
        embedder.embeddings[TensorKey.group.name].weight.zero_()

    field = TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[3]], dtype=torch.int64),
        present=torch.ones((1, 1), dtype=torch.bool),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        inferred=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    torch.manual_seed(1)
    a = embedder.embed(field).payload.clone()
    torch.manual_seed(2)
    b = embedder.embed(field).payload.clone()
    assert torch.equal(a, b)


def test_group_embedder_routes_sentinel_row_for_oov_tokens():
    schema = build_schema(n_groups=4, p_unavailable=0.0)
    embedder = build_embedder(schema)
    embedder.eval()
    sentinel = embedder.capacity

    field = TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[-1]], dtype=torch.int64),
        present=torch.ones((1, 1), dtype=torch.bool),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        inferred=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )
    output = embedder.embed(field).payload
    assert torch.isfinite(output).all()

    assign_logits = embedder.embeddings[TensorKey.group.name](torch.tensor([sentinel]))
    hard = torch.zeros_like(assign_logits)
    hard.scatter_(-1, assign_logits.argmax(dim=-1, keepdim=True), 1.0)
    expected_state = embedder.embeddings[TensorKey.state.name](field.state)
    expected_group = embedder.embeddings[TensorKey.content.name](hard)
    assert torch.allclose(output, expected_state + expected_group.reshape(1, 1, -1))


def test_group_embedder_zeros_group_contribution_for_non_valued_state():
    schema = build_schema(n_groups=4, p_unavailable=0.0)
    embedder = build_embedder(schema)
    embedder.eval()

    field = TensorField(
        state=torch.tensor(
            [
                [
                    Tokens.valued.value,
                    Tokens.padded.value,
                    Tokens.null.value,
                    Tokens.masked.value,
                ]
            ],
            dtype=torch.int64,
        ),
        content=torch.tensor([[2, 0, 0, 0]], dtype=torch.int64),
        present=torch.ones((1, 4), dtype=torch.bool),
        trainable=torch.zeros((1, 4), dtype=torch.bool),
        inferred=torch.zeros((1, 4), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )

    output = embedder.embed(field).payload
    expected = embedder.embeddings[TensorKey.state.name](field.state)
    assign_logits = embedder.embeddings[TensorKey.group.name](torch.tensor([2]))
    hard = torch.zeros_like(assign_logits)
    hard.scatter_(-1, assign_logits.argmax(dim=-1, keepdim=True), 1.0)
    expected[:, 0] += embedder.embeddings[TensorKey.content.name](hard).reshape(-1)
    assert torch.allclose(output, expected)


def test_group_embedder_rejects_indices_beyond_capacity():
    schema = build_schema(n_groups=4, p_unavailable=0.0)
    embedder = build_embedder(schema)

    field = TensorField(
        state=torch.tensor([[Tokens.valued.value]], dtype=torch.int64),
        content=torch.tensor([[9]], dtype=torch.int64),
        present=torch.ones((1, 1), dtype=torch.bool),
        trainable=torch.zeros((1, 1), dtype=torch.bool),
        inferred=torch.zeros((1, 1), dtype=torch.bool),
        targets=TensorDict({}),
        batch_size=1,
    )
    with pytest.raises(ValueError):
        embedder.embed(field)


def test_group_embedder_init_commits_only_lower_columns():
    # Init at ``lower`` keeps Sinkhorn from locking n_committed at ``upper`` from step 0.
    schema = build_schema(n_groups=(2, 6))
    embedder = build_embedder(schema)

    assert embedder.committed.tolist() == [True, True, False, False, False, False]
    assert torch.allclose(
        embedder.usage_ema,
        torch.tensor([0.5, 0.5, 0.0, 0.0, 0.0, 0.0]),
    )


def test_group_embedder_init_perplexity_matches_lower_bound():
    # perplexity(usage_ema) at init must equal ``lower`` so the loss's n_committed selection
    # starts small and only grows through revive.
    import math

    schema = build_schema(n_groups=(3, 8))
    embedder = build_embedder(schema)

    normalized = embedder.usage_ema / embedder.usage_ema.sum().clamp_min(1e-12)
    safe = normalized.clamp_min(1e-12)
    entropy = -(safe * safe.log()).sum().item()
    perplexity = math.exp(entropy)
    assert round(perplexity) == 3


def test_group_embedder_init_matches_upper_when_n_groups_fixed():
    # lower == upper is the "fixed K" case; init should commit all columns uniformly.
    schema = build_schema(n_groups=(4, 4))
    embedder = build_embedder(schema)

    assert embedder.committed.tolist() == [True] * 4
    assert torch.allclose(embedder.usage_ema, torch.full((4,), 0.25))


# ---------- write ----------


def write_model(*, vocab_tokens: list[str], capacity: int, K: int, assign_weight: torch.Tensor | None = None):
    configured = rf.Model.xs(
        d_model=8,
        items=rf.Branch(length=2, group=rf.Group(n_groups=K, mask=True)),
    )
    embedder = configured.nodes[ADDRESS].embedder
    resize = Resize()
    embedder.resize(capacity, resize)
    resize.commit()
    embedder.vocab.load_snapshot(vocab_tokens)
    with torch.no_grad():
        weight = embedder.embeddings["group"].weight
        if assign_weight is None:
            weight.zero_()
            rows = torch.arange(capacity + 1)
            weight[rows, rows.remainder(K)] = 1
        else:
            weight.copy_(assign_weight)
    return configured


def test_group_write_emits_arrow_group_and_content_payloads():
    K = 4
    capacity = 5
    module = write_model(
        vocab_tokens=["ALPHA", "BETA", "GAMMA", "DELTA", "EPS"],
        capacity=capacity,
        K=K,
    )

    state_logits = torch.zeros(2, 1, len(Tokens))
    state_logits[0, 0, Tokens.valued.value] = 10.0
    state_logits[1, 0, Tokens.padded.value] = 10.0

    group_logits = torch.zeros(2, 1, K)
    group_logits[0, 0, 1] = 10.0
    group_logits[1, 0, 2] = 10.0

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.group: group_logits},
            batch_size=[2],
        ),
    )

    datatype = output_type(module, ADDRESS)
    output = write(module=module, prediction=prediction, datatype=datatype)
    assert output.type == datatype
    assert output.type.names == [TensorKey.group.name, TensorKey.content.name]

    group_payload = output.field(TensorKey.group.name)
    assert group_payload.field(TensorKey.value.name).to_pylist() == [1, 2]
    assert all(value > 0.99 for value in group_payload.field(TensorKey.probability.name).to_pylist())

    # Assignment table row i is one-hot on group (i % K), so vocab argmax on group 1 -> BETA,
    # group 2 -> GAMMA.
    content_payload = output.field(TensorKey.content.name)
    assert content_payload.field(TensorKey.value.name).to_pylist() == ["BETA", "GAMMA"]


def test_group_write_drops_sentinel_column_from_content_prediction():
    K = 3
    capacity = 3
    assign_weight = torch.zeros(capacity + 1, K)
    assign_weight[0, 1] = 1.0
    assign_weight[1, 1] = 1.0
    assign_weight[2, 2] = 1.0
    # Sentinel row (index=capacity) is one-hot on group 0.
    assign_weight[3, 0] = 1.0
    module = write_model(
        vocab_tokens=["A", "B", "C"],
        capacity=capacity,
        K=K,
        assign_weight=assign_weight,
    )

    group_logits = torch.zeros(1, 1, K)
    group_logits[0, 0, 0] = 10.0
    state_logits = torch.zeros(1, 1, len(Tokens))

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.group: group_logits},
            batch_size=[1],
        ),
    )
    output = write(module=module, prediction=prediction, datatype=output_type(module, ADDRESS))

    # The sentinel is excluded from the normalized identity likelihood. Tiny
    # probability on group 2 favors its sole preferred real identity, C.
    content_payload = output.field(TensorKey.content.name)
    assert content_payload.field(TensorKey.value.name).to_pylist() == ["C"]


def test_group_write_returns_none_labels_when_vocab_is_empty():
    K = 4
    capacity = 5
    module = write_model(vocab_tokens=[], capacity=capacity, K=K)

    state_logits = torch.zeros(1, 1, len(Tokens))
    group_logits = torch.zeros(1, 1, K)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state_logits, TensorKey.group: group_logits},
            batch_size=[1],
        ),
    )
    output = write(module=module, prediction=prediction, datatype=output_type(module, ADDRESS))
    content_payload = output.field(TensorKey.content.name)
    assert content_payload.field(TensorKey.value.name).to_pylist() == [None]
    assert content_payload.field(TensorKey.probability.name).to_pylist() == [0.0]


# ---------- loss ----------


class TrackingModule:
    def __init__(self, schema: Schema, embedder: Embedder, decoder: Decoder):
        self.schema = schema
        self.nodes = {ADDRESS: SimpleNamespace(embedder=embedder, decoder=decoder)}
        self.tracked: dict[tuple, torch.Tensor] = {}

    def track(self, names: tuple, value: torch.Tensor) -> torch.Tensor:
        self.tracked[names] = value
        return value


def _prepared_train_batch(
    schema: Schema,
    *,
    rows: list[list[Any]],
) -> tuple[TensorField, Embedder]:
    embedder = build_embedder(schema)
    field = tensorfield(
        rows=rows,
        schema=schema,
        strata=Strata.train,
        state=embedder.vocab.state,
    )
    return field, embedder


def test_group_loss_sinkhorn_pushes_gradient_toward_spread():
    # Logits must be commensurate with ε=0.05 so Sinkhorn's Boltzmann map is not saturated.
    schema = build_schema(
        n_groups=(4, 4),
        p_unavailable=1.0,
        mask=Mask(reconstruct=True),
    )
    field, embedder = _prepared_train_batch(schema, rows=[["ALPHA", "BETA"]] * 4)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema, embedder, decoder)
    K = schema.requests[ADDRESS].n_groups[-1]

    group_logits = torch.zeros(*field.state.shape, K)
    group_logits[..., 0] = 0.025
    group_logits = group_logits.detach().requires_grad_(True)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.group: group_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    balance = module.tracked[(ADDRESS, Strata.train, "loss", TensorKey.group)]
    assert torch.isfinite(balance) and balance.item() > 0.0

    balance.backward()
    grad = group_logits.grad
    assert grad is not None
    valued_mask = field.targets[TensorKey.state].eq(Tokens.valued.value)
    valued_grad = grad[valued_mask]
    # Collapsed column receives positive gradient (push probability down); empty columns receive
    # negative gradient (pull them up).
    assert valued_grad[:, 0].mean().item() > 0.0
    assert valued_grad[:, 1:].mean().item() < 0.0


def test_group_loss_balance_penalizes_uncommitted_mass():
    # Balance loss uses a full-K log-softmax so uncommitted decoder rows receive gradient
    # through the normalizer. Raising uncommitted logits (as revive does when it copies a
    # donor row into a dead column) must INCREASE balance loss, which is exactly the signal
    # the model needs to eventually push those uncommitted rows back down.
    schema = build_schema(
        n_groups=(2, 4),
        p_unavailable=1.0,
        mask=Mask(reconstruct=True),
    )
    field, embedder = _prepared_train_batch(schema, rows=[["ALPHA", "BETA"]] * 4)
    decoder = Decoder(schema=schema, address=ADDRESS)
    K = schema.requests[ADDRESS].n_groups[-1]
    lower = schema.requests[ADDRESS].n_groups[0]

    base_logits = torch.zeros(*field.state.shape, K)
    base_logits[..., 0] = 0.075
    base_logits[..., 1] = 0.025

    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
        embedder.usage_ema[1] = 0.9

    def _balance(group_logits: torch.Tensor) -> float:
        module = TrackingModule(schema, embedder, decoder)
        prediction = Prediction(
            address=ADDRESS,
            payload=TensorDict(
                {
                    TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                    TensorKey.group: group_logits,
                },
                batch_size=field.batch_size,
            ),
        )
        loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
        return module.tracked[(ADDRESS, Strata.train, "loss", TensorKey.group)].item()

    baseline = _balance(base_logits.clone())

    revived_logits = base_logits.clone()
    revived_logits[..., 2] = base_logits[..., 0]
    revived_logits[..., 3] = base_logits[..., 0]

    _, expected_committed = torch.tensor([1.0, 0.9, 0.0, 0.0]).topk(lower)
    assert set(expected_committed.tolist()) == {0, 1}

    revived = _balance(revived_logits)
    assert revived > baseline


def test_group_loss_balance_gradient_reaches_uncommitted_columns():
    # Under the full-K normalizer, backprop from balance loss must produce a *negative*-going
    # gradient on uncommitted logits (push them down) while committed logits still get pulled
    # up. This is the mechanism that lets ``adherence_ema`` decay once the true K is reached.
    schema = build_schema(
        n_groups=(2, 4),
        p_unavailable=1.0,
        mask=Mask(reconstruct=True),
    )
    field, embedder = _prepared_train_batch(schema, rows=[["ALPHA", "BETA"]] * 4)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema, embedder, decoder)
    K = schema.requests[ADDRESS].n_groups[-1]

    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
        embedder.usage_ema[1] = 0.9

    group_logits = torch.zeros(*field.state.shape, K)
    group_logits[..., 0] = 0.05
    group_logits[..., 1] = 0.02
    # Uncommitted logits raised above baseline; a healthy loss should pull them back down.
    group_logits[..., 2] = 0.10
    group_logits[..., 3] = 0.10
    group_logits = group_logits.detach().requires_grad_(True)

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.group: group_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    module.tracked[(ADDRESS, Strata.train, "loss", TensorKey.group)].backward()

    grad = group_logits.grad
    assert grad is not None
    valued_mask = field.targets[TensorKey.state].eq(Tokens.valued.value)
    valued_grad = grad[valued_mask]
    # Uncommitted (columns 2, 3): positive gradient (loss increases with these logits -> SGD
    # will push them down).
    assert valued_grad[:, 2:].mean().item() > 0.0
    # Committed (columns 0, 1): non-positive net gradient (loss decreases as these logits rise).
    assert valued_grad[:, :2].mean().item() <= 0.0


def test_group_coverage_ignores_simulated_input_unavailability():
    schema = build_schema(
        n_groups=(2, 4),
        p_unavailable=1.0,
        mask=Mask(reconstruct=True),
    )
    field, embedder = _prepared_train_batch(
        schema,
        rows=[["ALPHA", "BETA"], ["BETA", "ALPHA"]],
    )
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema, embedder, decoder)
    K = schema.requests[ADDRESS].n_groups[-1]

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.group: torch.randn(*field.state.shape, K),
            },
            batch_size=field.batch_size,
        ),
    )
    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)

    scores = decoder.metrics["train_metrics"].compute()
    assert scores["coverage.content"] == 1.0
    assert scores["targets.unavailable"] == 0


def test_group_content_counter_does_not_reweight_probability_objective():
    # Exposure counts are not a request to replace the observed class prior.
    schema = build_schema(
        n_groups=(4, 4),
        p_unavailable=0.0,
        mask=Mask(reconstruct=True),
    )
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    K = schema.requests[ADDRESS].n_groups[-1]
    capacity = embedder.capacity

    content_counter = embedder.counters[TensorKey.content.name]
    assert isinstance(content_counter, Counter)
    assert content_counter.size == capacity + 1

    # Encode the batch through the embedder's own vocab so the vocab-size gated accuracy
    # metric can evaluate under known content targets.
    field = tensorfield(
        rows=[["ALPHA", "BETA"]] * 4,
        schema=schema,
        strata=Strata.train,
        state=embedder.vocab.state,
    )
    torch.manual_seed(0)
    group_logits = torch.randn(*field.state.shape, K)
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.group: group_logits,
            },
            batch_size=field.batch_size,
        ),
    )

    module_uniform = TrackingModule(schema, embedder, decoder)
    loss(module=module_uniform, prediction=prediction, batch=field, strata=Strata.train)
    loss_uniform = decoder.metrics["train_metrics"].compute()["loss.content"].item()
    decoder.metrics["train_metrics"].reset()

    with torch.no_grad():
        content_counter.counts.fill_(1)
        content_counter.counts[0] = 1000

    module_skewed = TrackingModule(schema, embedder, decoder)
    loss(module=module_skewed, prediction=prediction, batch=field, strata=Strata.train)
    loss_skewed = decoder.metrics["train_metrics"].compute()["loss.content"].item()

    assert loss_uniform == pytest.approx(loss_skewed, abs=1e-6)


def test_group_loss_usage_ema_updates_toward_batch_distribution():
    schema = build_schema(
        n_groups=(4, 4),
        p_unavailable=1.0,
        mask=Mask(reconstruct=True),
    )
    field, embedder = _prepared_train_batch(schema, rows=[["ALPHA", "BETA"]] * 4)
    decoder = Decoder(schema=schema, address=ADDRESS)
    module = TrackingModule(schema, embedder, decoder)
    K = schema.requests[ADDRESS].n_groups[-1]

    prior = embedder.usage_ema.clone()
    group_logits = torch.zeros(*field.state.shape, K)
    group_logits[..., 0] = 10.0

    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.group: group_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    loss(module=module, prediction=prediction, batch=field, strata=Strata.train)
    GroupUsageCallback().on_train_batch_end(FakeTrainer(), module, None, field, 0)

    # Column 0 usage should increase, others should decrease (weighted by ema_decay).
    assert embedder.usage_ema[0].item() > prior[0].item()
    for k in range(1, K):
        assert embedder.usage_ema[k].item() < prior[k].item()
    # Distribution stays a probability vector.
    assert torch.isclose(embedder.usage_ema.sum(), torch.tensor(1.0), atol=1e-5)


# ---------- GroupReviveCallback ----------


def _seed_uncommitted_batch(*, revive_temperature: float | None = None):
    """Prepare a train batch + real embedder/decoder with a range-bounded group leaf.

    Seeds ``usage_ema`` to concentrate on the lower half so dynamic committed selection lands
    on ``[0, lower)``, marks the upper half of ``committed`` as dead (for callback consumers),
    and biases the group logits toward those dead columns so ``adherence_ema`` accumulates.
    """
    schema = build_schema(
        n_groups=(2, 4),
        p_unavailable=1.0,
        revive_temperature=revive_temperature,
        mask=Mask(reconstruct=True),
    )
    field, embedder = _prepared_train_batch(schema, rows=[["ALPHA", "BETA"]] * 4)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    K = schema.requests[ADDRESS].n_groups[-1]
    lower = schema.requests[ADDRESS].n_groups[0]
    with torch.no_grad():
        # Force perplexity ~ lower so dynamic n_committed lands at `lower` and topk picks [0, lower).
        embedder.usage_ema.zero_()
        embedder.usage_ema[:lower] = 1.0 / lower
    embedder.committed[lower:] = False

    group_logits = torch.zeros(*field.state.shape, K)
    group_logits[..., lower:] = 5.0
    prediction = Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {
                TensorKey.state: torch.zeros(*field.state.shape, len(Tokens)),
                TensorKey.group: group_logits,
            },
            batch_size=field.batch_size,
        ),
    )
    module = TrackingModule(schema, embedder, decoder)
    return schema, field, module, embedder, decoder, prediction


class FakeTrainer:
    def __init__(self, is_global_zero: bool = True, current_epoch: int = 0):
        self.is_global_zero = is_global_zero
        self.current_epoch = current_epoch


def test_revive_callback_noop_when_revive_temperature_zero():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=0.0)
    embedder.adherence_ema.fill_(0.9)
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()
    prior_usage = embedder.usage_ema.detach().clone()

    GroupReviveCallback().on_train_epoch_end(FakeTrainer(), module)

    # revive_temperature == 0 -> callback is a no-op, no weights touched, ema preserved.
    assert torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)
    assert torch.equal(embedder.usage_ema, prior_usage)
    assert embedder.adherence_ema.item() == pytest.approx(0.9)


def test_revive_callback_noop_when_n_groups_are_fixed():
    schema = build_schema(n_groups=(4, 4), p_unavailable=1.0, revive_temperature=10.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    embedder.adherence_ema.fill_(0.9)
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()

    module = TrackingModule(schema, embedder, decoder)
    GroupReviveCallback().on_train_epoch_end(FakeTrainer(), module)

    # lower == upper -> no dead columns are eligible; weights untouched, adherence preserved.
    assert torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == pytest.approx(0.9)


def test_revive_callback_noop_when_all_signals_are_off():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    # Past warmup, adherence == 0, committed usage concentrated on a single column so it is
    # not saturated. With every signal off the gate stays closed even under a live temperature.
    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()

    GroupReviveCallback().on_train_epoch_end(FakeTrainer(current_epoch=100), module)

    assert torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == 0.0


def test_revive_callback_warmup_fires_without_adherence():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    # During warmup the callback fires even with adherence == 0, seeding dead columns from the
    # top-usage committed donor(s) so the model can reach the upper bound without needing to
    # first accumulate mass on columns whose weights are still random.
    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 1.0
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()

    torch.manual_seed(0)
    GroupReviveCallback().on_train_epoch_end(FakeTrainer(current_epoch=0), module)

    assert not torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)


def test_revive_callback_does_not_fire_on_saturated_committed_alone():
    # Saturation of the committed set (perfectly uniform usage on committed columns) is the
    # natural end state of Sinkhorn balance, not a signal that more groups are needed. Past
    # warmup, with adherence == 0, saturation alone must NOT fire revive.
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    lower = module.schema.requests[ADDRESS].n_groups[0]
    with torch.no_grad():
        embedder.usage_ema.zero_()
        embedder.usage_ema[:lower] = 1.0 / lower  # perfectly balanced -> "saturated"
        embedder.adherence_ema.zero_()
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()

    GroupReviveCallback().on_train_epoch_end(FakeTrainer(current_epoch=100), module)

    assert torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)


def test_revive_callback_splits_donor_into_dead_column_and_resets_state():
    torch.manual_seed(0)
    # n_groups=(3, 4) yields exactly 1 uncommitted column so the donor is halved exactly once.
    schema = build_schema(n_groups=(3, 4), p_unavailable=1.0, revive_temperature=10.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    lower = schema.requests[ADDRESS].n_groups[0]
    embedder.committed[lower:] = False
    module = TrackingModule(schema, embedder, decoder)

    # Force adherence high enough that the per-column trial passes deterministically.
    embedder.adherence_ema.fill_(1.0)

    # Concentrate all usage on column 0 so the donor is deterministic.
    embedder.usage_ema.zero_()
    embedder.usage_ema[0] = 1.0

    # Give distinct values to donor + the dead column to detect the copy.
    assign_w = embedder.embeddings[TensorKey.group.name].weight
    content_w = embedder.embeddings[TensorKey.content.name].weight
    group_w = decoder.linears[TensorKey.group.name].weight
    dead = int((~embedder.committed).nonzero(as_tuple=True)[0][0].item())

    donor_assign = torch.arange(assign_w.shape[0], dtype=assign_w.dtype)
    donor_content = torch.arange(content_w.shape[0], dtype=content_w.dtype)
    donor_group = torch.arange(group_w.shape[1], dtype=group_w.dtype)
    assign_w.data[:, 0] = donor_assign
    content_w.data[:, 0] = donor_content
    group_w.data[0, :] = donor_group
    assign_w.data[:, dead] = 0.0
    content_w.data[:, dead] = 0.0
    group_w.data[dead, :] = 0.0

    GroupReviveCallback().on_train_epoch_end(FakeTrainer(current_epoch=0), module)

    # Symmetry-breaking noise is small, so the dead column is close to the donor.
    assert torch.allclose(assign_w.data[:, dead], donor_assign, atol=0.2)
    assert torch.allclose(content_w.data[:, dead], donor_content, atol=0.2)
    assert torch.allclose(group_w.data[dead, :], donor_group, atol=0.2)
    # Donor usage halved; revived column inherits half of donor's usage.
    assert embedder.usage_ema[0].item() == pytest.approx(0.5)
    assert embedder.usage_ema[dead].item() == pytest.approx(0.5)
    assert embedder.committed[dead]
    assert not embedder.retired[dead]
    # adherence_ema reset after revive.
    assert embedder.adherence_ema.item() == 0.0


def test_revive_callback_respects_non_global_zero_rank():
    _, _, module, embedder, decoder, _ = _seed_uncommitted_batch(revive_temperature=10.0)
    embedder.adherence_ema.fill_(1.0)
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()

    # No distributed backend is up -> broadcast_object is a passthrough, so a rank-non-zero
    # trainer produces an empty plans dict; no weights change.
    GroupReviveCallback().on_train_epoch_end(FakeTrainer(is_global_zero=False), module)

    assert torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)
    assert embedder.adherence_ema.item() == pytest.approx(1.0)


def test_revive_callback_base_probability_decays_with_epoch():
    # At epoch >> revive_temperature the base probability decays to ~0 and revival never fires
    # even with saturated adherence.
    torch.manual_seed(0)
    schema = build_schema(n_groups=(3, 4), p_unavailable=1.0, revive_temperature=1.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    lower = schema.requests[ADDRESS].n_groups[0]
    embedder.committed[lower:] = False
    embedder.adherence_ema.fill_(1.0)
    module = TrackingModule(schema, embedder, decoder)
    prior_assign = embedder.embeddings[TensorKey.group.name].weight.detach().clone()

    # exp(-100/1) ≈ 3.7e-44, so the per-column Bernoulli trial cannot pass.
    GroupReviveCallback().on_train_epoch_end(FakeTrainer(current_epoch=100), module)

    assert torch.equal(embedder.embeddings[TensorKey.group.name].weight, prior_assign)
    # Adherence stays intact because no revive occurred.
    assert embedder.adherence_ema.item() == pytest.approx(1.0)


def test_revive_callback_warmup_caps_expected_revivals_at_one_per_epoch():
    # During warmup the base probability is 1.0 and adherence is boosted to 1.0, but the
    # per-column cap is still ``1 / n_dead`` so the expected number of revivals in a single
    # epoch is ~1 (not "all dead columns"). This is what prevents ``n_committed`` from jumping
    # straight to the upper bound on the first epoch.
    schema = build_schema(n_groups=(1, 12), p_unavailable=1.0, revive_temperature=10.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    lower = schema.requests[ADDRESS].n_groups[0]
    embedder.committed[:] = False
    embedder.committed[:lower] = True
    embedder.adherence_ema.zero_()  # warmup should drive the plan even without adherence

    n_dead = int((~embedder.committed).sum().item())
    assert n_dead == 11

    # ``apply`` sets ``usage_ema[k] = usage_ema[donor] * 0.5`` for each revived dead column,
    # so counting non-zero entries after the callback is an exact proxy for revive count.
    revive_counts: list[int] = []
    module = TrackingModule(schema, embedder, decoder)
    for seed in range(50):
        with torch.no_grad():
            embedder.usage_ema.zero_()
            embedder.usage_ema[0] = 1.0
            embedder.committed.zero_()
            embedder.committed[0] = True
            embedder.retired.zero_()
        torch.manual_seed(seed)
        GroupReviveCallback().on_train_epoch_end(FakeTrainer(current_epoch=0), module)
        revived = int((embedder.usage_ema > 0).sum().item()) - 1  # minus the donor column
        revive_counts.append(revived)

    mean = sum(revive_counts) / len(revive_counts)
    # cap = 1/11, signal = 1 -> expected revivals per epoch = 1.
    assert 0.5 <= mean <= 2.0, f"expected mean ≈ 1, got {mean} (samples: {revive_counts[:10]})"
    # No single epoch should revive more than a small handful of columns.
    assert max(revive_counts) < 6


# ---------- GroupMergeCallback ----------


def _seed_committed_pair(*, similarity: float):
    """Build a real embedder/decoder with two committed columns whose content_w columns and
    group_w rows share ``similarity`` cosine similarity (both weights, since the merge plan
    requires joint redundancy)."""
    schema = build_schema(n_groups=(2, 4), p_unavailable=1.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    lower = schema.requests[ADDRESS].n_groups[0]
    K = schema.requests[ADDRESS].n_groups[-1]

    # Commit only [0, 1, 2] so K==4 with one dead column; merge target is between 0 and 1.
    embedder.committed[:] = False
    embedder.committed[: lower + 1] = True
    assert int(embedder.committed.sum().item()) > lower

    torch.manual_seed(0)
    content_w = embedder.embeddings[TensorKey.content.name].weight
    group_w = decoder.linears[TensorKey.group.name].weight
    with torch.no_grad():
        content_w.data.normal_()
        group_w.data.normal_()
        # Blend column 1 (content) and row 1 (group) toward column/row 0.
        content_w.data[:, 1] = similarity * content_w.data[:, 0] + (1.0 - similarity) * content_w.data[:, 1]
        group_w.data[1, :] = similarity * group_w.data[0, :] + (1.0 - similarity) * group_w.data[1, :]
        # Winner is column 0 (higher usage) so the merge should decommit column 1.
        embedder.usage_ema.zero_()
        embedder.usage_ema[0] = 0.6
        embedder.usage_ema[1] = 0.3
        embedder.usage_ema[2] = 0.1

    module = TrackingModule(schema, embedder, decoder)
    return schema, module, embedder, decoder, K


def test_merge_callback_decommits_duplicate_lower_usage_column():
    schema, module, embedder, decoder, _ = _seed_committed_pair(similarity=1.0)
    prior_usage_winner = float(embedder.usage_ema[0].item())
    prior_usage_loser = float(embedder.usage_ema[1].item())

    GroupMergeCallback().on_train_epoch_end(FakeTrainer(), module)

    # Column 1 (lower usage of the duplicated pair) is demoted; column 0 keeps its commit.
    assert not bool(embedder.committed[1].item())
    assert bool(embedder.committed[0].item())
    # Loser mass rolled into winner so perplexity can actually drop.
    assert embedder.usage_ema[1].item() == pytest.approx(0.0)
    assert embedder.usage_ema[0].item() == pytest.approx(prior_usage_winner + prior_usage_loser)


def test_merge_callback_zeros_loser_decoder_row():
    # Retired coordinates keep no stale decoder weights; support masking excludes their predictions.
    _, module, embedder, decoder, _ = _seed_committed_pair(similarity=1.0)
    GroupMergeCallback().on_train_epoch_end(FakeTrainer(), module)

    loser_row = decoder.linears[TensorKey.group.name].weight.data[1, :]
    assert torch.equal(loser_row, torch.zeros_like(loser_row))
    assert embedder.retired[1]


def test_merge_callback_noop_when_columns_are_distinct():
    _, module, embedder, decoder, _ = _seed_committed_pair(similarity=0.0)
    prior_committed = embedder.committed.detach().clone()
    prior_group_w = decoder.linears[TensorKey.group.name].weight.detach().clone()

    GroupMergeCallback().on_train_epoch_end(FakeTrainer(), module)

    # No pair exceeds the similarity threshold -> nothing changes.
    assert torch.equal(embedder.committed, prior_committed)
    assert torch.equal(decoder.linears[TensorKey.group.name].weight, prior_group_w)


def test_merge_callback_noop_when_n_groups_are_fixed():
    # With a fixed range there is no "shrink room"; the merge plan must not fire even under
    # perfect redundancy so the model can rely on the full committed set.
    schema = build_schema(n_groups=(3, 3), p_unavailable=1.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    with torch.no_grad():
        content_w = embedder.embeddings[TensorKey.content.name].weight
        group_w = decoder.linears[TensorKey.group.name].weight
        content_w.data.normal_()
        group_w.data.normal_()
        content_w.data[:, 1] = content_w.data[:, 0]
        group_w.data[1, :] = group_w.data[0, :]
    prior_committed = embedder.committed.detach().clone()

    module = TrackingModule(schema, embedder, decoder)
    GroupMergeCallback().on_train_epoch_end(FakeTrainer(), module)

    assert torch.equal(embedder.committed, prior_committed)


def test_merge_callback_noop_at_lower_bound():
    # ``n_committed == lower`` means the model already believes it's at the minimum plausible K
    # for this batch; refusing to shrink further prevents cannibalizing the last group.
    schema = build_schema(n_groups=(2, 4), p_unavailable=1.0)
    embedder = build_embedder(schema)
    decoder = Decoder(schema=schema, address=ADDRESS)
    # These callback fixtures represent an epoch with known reconstruction evidence.
    decoder.metrics["train_metrics"].update(torch.tensor([2, 2, 0]), torch.zeros(2))
    lower = schema.requests[ADDRESS].n_groups[0]
    # Only ``lower`` columns committed, and make them duplicates to prove the guard fires
    # before the similarity check.
    embedder.committed[:] = False
    embedder.committed[:lower] = True
    with torch.no_grad():
        content_w = embedder.embeddings[TensorKey.content.name].weight
        group_w = decoder.linears[TensorKey.group.name].weight
        content_w.data.normal_()
        group_w.data.normal_()
        content_w.data[:, 1] = content_w.data[:, 0]
        group_w.data[1, :] = group_w.data[0, :]
    prior_committed = embedder.committed.detach().clone()

    module = TrackingModule(schema, embedder, decoder)
    GroupMergeCallback().on_train_epoch_end(FakeTrainer(), module)

    assert torch.equal(embedder.committed, prior_committed)


def test_merge_callback_respects_non_global_zero_rank():
    _, module, embedder, decoder, _ = _seed_committed_pair(similarity=1.0)
    prior_committed = embedder.committed.detach().clone()

    # Rank != 0 -> planning is skipped, no distributed backend means broadcast_object returns
    # the empty dict, and nothing is applied.
    GroupMergeCallback().on_train_epoch_end(FakeTrainer(is_global_zero=False), module)

    assert torch.equal(embedder.committed, prior_committed)
