"""Ordered labels and open integer ranks share one numeric modeling path."""

import json

import lightning.pytorch as lit
import polars as pl
import pyarrow as pa
import pytest
import torch
import yaml
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.structs.packages import Prediction
from relflow.tensorfields.extensions import number, ordinal

ADDRESS = rf.Address("level")
VALUES = ("low", "medium", "high")


def model(values=None, **options):
    return rf.Model(preset="small", d_model=8, level=rf.Ordinal(values=values, **options))


def prediction(values):
    ranks = torch.tensor(values, dtype=torch.float64)
    return Prediction(address=ADDRESS, payload=TensorDict({rf.TensorKey.content: ranks}, batch_size=[len(ranks)]))


def test_ordinal_reuses_numeric_learning_without_a_class_vocabulary():
    assert rf.TENSORFIELDS["ordinal"].Request is rf.Ordinal
    assert issubclass(ordinal.Embedder, number.Embedder)
    assert issubclass(ordinal.Decoder, number.Decoder)
    for role in (rf.Component.TensorField, rf.Component.observe, rf.Component.learn, rf.Component.loss):
        assert ordinal.ordinal.component(role) is number.number.component(role)
    configured = model(VALUES, mask=True)
    assert configured.nodes[ADDRESS].decoder.regression.out_features == 1
    assert ordinal.ordinal.callback_factories == number.number.callback_factories


@pytest.mark.parametrize("values", [(), ("a", "a"), (True, 1), (1, 2.0), (None,), (float("nan"),), (float("inf"),)])
def test_ordinal_rejects_ambiguous_or_invalid_label_declarations(values):
    with pytest.raises(ValueError, match="Ordinal"):
        rf.Ordinal(values=values)


@pytest.mark.parametrize("query", [("one", "two"), (), "broken[", "a..b"])
def test_ordinal_keeps_structural_query_validation(query):
    with pytest.raises(ValueError, match="query|Ordinal"):
        rf.Ordinal(query=query)


@pytest.mark.parametrize("mask", [False, True])
def test_declared_order_becomes_numeric_rank_before_statistics_and_masking(mask):
    configured = model(VALUES, mask=mask)
    field = configured.encode(pa.table({"level": ["high", "low", None, "medium"]}), strata="train")[ADDRESS]
    content = field.targets[rf.TensorKey.content] if mask else field.content
    assert content.flatten().tolist() == [2.0, 0.0, 0.0, 1.0]
    assert rf.Ordinal.normalization(configured, ADDRESS)["mean"] == 1.0
    assert rf.Ordinal.normalization(configured, ADDRESS)["count"] == 3
    if not mask:
        assert field.state.flatten().tolist() == [rf.Tokens.valued, rf.Tokens.valued, rf.Tokens.null, rf.Tokens.valued]


@pytest.mark.parametrize("strata", ["train", "validate", "test", "predict"])
def test_implicit_integer_ranks_accept_unseen_values_without_admission(strata):
    configured = model()
    configured.encode(pa.table({"level": [-10, 0, 10]}), strata="train")
    previous = rf.Ordinal.normalization(configured, ADDRESS)
    field = configured.encode(pa.table({"level": [-9, 7, 100]}), strata=strata)[ADDRESS]
    assert field.content.flatten().tolist() == [-9.0, 7.0, 100.0]
    if strata != "train":
        assert rf.Ordinal.normalization(configured, ADDRESS) == previous


@pytest.mark.parametrize("values", [[1.0], [True], ["1"]])
def test_implicit_ranks_require_integer_arrow_atoms(values):
    with pytest.raises(ValueError, match="level.*integer ranks"):
        model().encode(pa.table({"level": values}))


@pytest.mark.parametrize(
    "values", [pa.array([2**53 + 1]), pa.array([-(2**53) - 1]), pa.array([2**64 - 1], type=pa.uint64())]
)
def test_implicit_ranks_reject_inexact_or_unsigned_overflow(values):
    with pytest.raises(ValueError, match="level.*2\\*\\*53"):
        model().encode(pa.table({"level": values}))


def test_unsigned_supported_boundary_and_all_null_sources_preserve_states():
    configured = model()
    field = configured.encode(pa.table({"level": pa.array([0, 2**53, None], type=pa.uint64())}))[ADDRESS]
    assert field.content.flatten().tolist() == [0.0, float(2**53), 0.0]
    assert field.state.flatten().tolist() == [rf.Tokens.valued, rf.Tokens.valued, rf.Tokens.null]
    for values in (pa.nulls(2), pa.array([None, None], type=pa.int64())):
        field = configured.encode(pa.table({"level": values}))[ADDRESS]
        assert field.state.eq(rf.Tokens.null).all()


def test_declared_large_integer_labels_keep_exact_identity_through_rank_conversion():
    labels = (2**53 + 1, 2**53 + 2)
    configured = model(labels)
    field = configured.encode(pa.table({"level": list(reversed(labels))}))[ADDRESS]
    assert field.content.flatten().tolist() == [1.0, 0.0]
    written = ordinal.write(configured, prediction([1, 0]), ordinal.output(configured, ADDRESS))
    assert written.field("content").to_pylist() == list(reversed(labels))


def test_declared_dictionary_values_and_nulls_have_the_same_rank_encoding():
    labels = pa.array(["high", "low", None]).dictionary_encode()
    field = model(VALUES).encode(pa.table({"level": labels}))[ADDRESS]
    assert field.content.flatten().tolist() == [2.0, 0.0, 0.0]
    assert field.state.flatten().tolist() == [rf.Tokens.valued, rf.Tokens.valued, rf.Tokens.null]


@pytest.mark.parametrize("labels", [["unknown"], [1]])
@pytest.mark.parametrize("mask", [False, True])
def test_declared_labels_are_checked_before_masking(labels, mask):
    with pytest.raises(ValueError, match="Ordinal.*level"):
        model(VALUES, mask=mask).encode(pa.table({"level": labels}), strata="train")


def test_query_and_branch_cropping_preserve_record_holes_and_rank_values():
    records = pa.Table.from_pylist(
        [
            {"events": [{"rating": "low"}, None, {"rating": "high"}]},
            {"events": []},
            {"events": None},
            {"events": [{"rating": None}]},
        ]
    )
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=2, overflow="tail", level=rf.Ordinal(values=VALUES, query="rating", embed=True)),
    )
    field = configured.encode(records)["/events/level"]
    assert field.content.reshape(4, 2).tolist() == [[0, 2], [0, 0], [0, 0], [0, 0]]
    assert field.state.reshape(4, 2).tolist() == [
        [rf.Tokens.padded, rf.Tokens.valued],
        [rf.Tokens.padded, rf.Tokens.padded],
        [rf.Tokens.padded, rf.Tokens.padded],
        [rf.Tokens.null, rf.Tokens.padded],
    ]
    assert len(configured.predict(records)) == 4


@pytest.mark.parametrize("labels", [(False, True), (100, 20, -1), (1.25, -3.5), (b"low", b"high"), VALUES])
def test_native_declared_labels_survive_json_yaml_and_checkpoint_roundtrips(labels, tmp_path):
    configured = model(labels, mask=True)
    declaration = configured.schema.requests[ADDRESS].model_dump(mode="json", exclude={"name"})
    assert rf.Ordinal.model_validate({"values": json.loads(json.dumps(declaration))["values"]}).values == labels
    assert rf.Schema.model_validate_json(configured.schema.model_dump_json()).requests[ADDRESS].values == labels
    declaration["mask"] = True
    path = tmp_path / "ordinal.yaml"
    path.write_text(yaml.safe_dump({"preset": "small", "d_model": 8, "fields": {"level": declaration}}))
    restored = rf.Model.from_yaml(path)
    assert restored.schema.requests[ADDRESS].values == labels
    configured.encode(pa.table({"level": list(labels)}), strata="train")
    path = tmp_path / "ordinal.pt"
    configured.save(path)
    restored = rf.Model.load(path)
    assert restored.schema.requests[ADDRESS].values == labels
    assert rf.Ordinal.normalization(restored, ADDRESS) == rf.Ordinal.normalization(configured, ADDRESS)
    written = ordinal.write(restored, prediction([0, len(labels) - 1]), ordinal.output(restored, ADDRESS))
    assert written.field("content").type == pa.array(labels).type
    assert written.field("content").to_pylist() == [labels[0], labels[-1]]


@pytest.mark.parametrize("labels", [VALUES[::-1], (0, 1, 2), None])
def test_checkpoint_rejects_changed_rank_meanings(labels):
    with pytest.raises(ValueError, match="Ordinal.*level.*declaration"):
        model(labels, mask=True).load_state_dict(model(VALUES, mask=True).state_dict())


def test_stale_prefetched_batch_is_rejected_after_reordering():
    configured = model(VALUES, mask=True)
    pending = encode(
        pa.table({"level": ["low", "high"]}),
        configured.schema,
        rf.Strata.train,
        configured.interprocess_encoding_context,
    )
    configured.update(rf.where("address") == ADDRESS, values=VALUES[::-1])
    with pytest.raises(ValueError, match="Ordinal batch.*level.*declaration"):
        bind(configured, pending, rf.Strata.train)
    assert rf.Ordinal.normalization(configured, ADDRESS)["count"] == 0


def test_updating_a_declaration_resets_rank_model_weights_and_statistics():
    configured = model(VALUES, mask=True)
    configured.encode(pa.table({"level": ["high"]}), strata="train")
    with torch.no_grad():
        configured.nodes[ADDRESS].decoder.regression.weight.fill_(100)
    configured.update(rf.where("address") == ADDRESS, values=VALUES[::-1])
    assert not configured.nodes[ADDRESS].decoder.regression.weight.eq(100).all()
    assert rf.Ordinal.normalization(configured, ADDRESS)["count"] == 0


def test_unrelated_extension_preserves_pending_counter_state_and_rank_statistics():
    configured = model(VALUES, mask=True)
    configured.encode(pa.table({"level": ["high", "low"]}), strata="train")
    counter = configured.nodes[ADDRESS].embedder.counter
    counter.is_full = True
    pending = counter._pending_counts.clone()
    counts = counter.counts.clone()
    statistics = rf.Ordinal.normalization(configured, ADDRESS)
    configured.extend(rf.where("address") == "/", other=rf.Number)
    retained = configured.nodes[ADDRESS].embedder.counter
    assert retained.is_full
    torch.testing.assert_close(retained._pending_counts, pending)
    torch.testing.assert_close(retained.counts, counts)
    assert rf.Ordinal.normalization(configured, ADDRESS) == statistics


@pytest.mark.parametrize("source", [None, (True, False), (1.0, 0.0)])
def test_checkpoint_distinguishes_implicit_mode_and_label_scalar_types(source):
    with pytest.raises(ValueError, match="Ordinal.*declaration"):
        model((1, 0), mask=True).load_state_dict(model(source, mask=True).state_dict())


def test_writer_rounds_ties_to_even_and_clamps_declared_labels():
    configured = model(VALUES, mask=True)
    written = ordinal.write(configured, prediction([-10, 0.49, 0.5, 1.5, 20]), ordinal.output(configured, ADDRESS))
    assert written.field("content").to_pylist() == ["low", "low", "low", "high", "high"]


def test_writer_emits_open_integer_ranks_and_saturates_before_integer_cast():
    configured = model(mask=True)
    written = ordinal.write(configured, prediction([-1e30, -2.5, -1.5, 6.8, 1e30]), ordinal.output(configured, ADDRESS))
    assert written.field("content").type == pa.int64()
    assert written.field("content").to_pylist() == [-(2**53), -2, -2, 7, 2**53]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_writer_rejects_nonfinite_ranks_before_rounding(value):
    configured = model(mask=True)
    with pytest.raises(ValueError, match="Ordinal prediction.*level.*nonfinite"):
        ordinal.write(configured, prediction([value]), ordinal.output(configured, ADDRESS))


def test_numeric_objective_penalizes_larger_rank_errors_and_propagates_gradient(monkeypatch):
    configured = model(VALUES, mask=True, objective="mse")
    field = configured.encode(pa.table({"level": ["low", "medium", "high"]}), strata="train")[ADDRESS]
    monkeypatch.setattr(configured, "track", lambda *args, value, **kwargs: value)
    losses = []
    for offset in (0.0, 1.0, 2.0):
        predicted = field.targets[rf.TensorKey.content].detach().clone().add(offset).requires_grad_()
        payload = TensorDict(
            {
                rf.TensorKey.state: torch.zeros((*field.state.shape, len(rf.Tokens)), requires_grad=True),
                rf.TensorKey.content: predicted,
                "scaled": predicted,
            },
            batch_size=field.batch_size,
        )
        losses.append(number.loss(configured, Prediction(address=ADDRESS, payload=payload), field, rf.Strata.train))
    assert losses[0] < losses[1] < losses[2]
    losses[-1].backward()
    assert predicted.grad is not None and predicted.grad.gt(0).all()


@pytest.mark.integration
@pytest.mark.parametrize("values", [None, VALUES], ids=["open-integer-ranks", "declared-labels"])
def test_one_batch_training_and_validation_keep_native_rank_or_label_output(values):
    torch.manual_seed(91)
    configured = rf.Model(preset="small", d_model=8, amount=rf.Number, level=rf.Ordinal(values=values, mask=True))
    train = pa.table(
        {"amount": [-3, -1, 1, 3], "level": [-3, -1, 1, 3] if values is None else ["low", "low", "high", "high"]}
    )
    validate = pa.table(
        {"amount": [-2, 0, 2, 4], "level": [-2, 0, 2, 4] if values is None else ["low", "medium", "medium", "high"]}
    )
    gradients = []

    class Audit(lit.Callback):
        def on_after_backward(self, trainer, pl_module):
            gradient = pl_module.nodes[ADDRESS].decoder.regression.weight.grad
            assert gradient is not None
            gradients.append(gradient.detach().clone())

    data = rf.ArrowDataModule(
        model=configured, train=train, validate=validate, num_workers=0, persistent_workers=False, pin_memory=False
    )
    configured.fit(
        data,
        batch_size=4,
        learning_rate=0.01,
        early_stopping=False,
        rollback=False,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        precision="32-true",
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[Audit()],
    )
    assert len(gradients) == 1 and torch.isfinite(gradients[0]).all() and gradients[0].abs().sum() > 0
    predicted = (
        configured.predict(validate.drop(["level"]))["predictions"].struct.field("level").struct.field("content")
    )
    assert predicted.dtype == (pl.Int64 if values is None else pl.String)
    assert len(predicted) == len(validate)
    if values is not None:
        assert set(predicted.to_list()) <= set(values)
