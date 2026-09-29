"""Closed categorical declarations retain class identity across the model lifecycle."""

from pathlib import Path

import lightning.pytorch as lit
import pyarrow as pa
import pydantic
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.binding import bind
from relflow.architecture.runtime import ModelRuntime
from relflow.data.iterables import encode
from relflow.structs.enums import TensorKey, Tokens
from relflow.structs.packages import Prediction

ADDRESS = rf.Address("status")
VALUES = ("pending", "complete", "rejected")


def model(labels=VALUES, **options) -> rf.Model:
    return rf.Model.xs(
        d_model=8,
        n_heads=2,
        n_layers=1,
        batch_size=2,
        status=rf.Enum(values=labels, **{"p_unavailable": 0.0, **options}),
    )


def records(*values) -> pa.Table:
    return pa.table({"status": pa.array(values, type=pa.string())})


def prediction(field, logits: torch.Tensor) -> Prediction:
    state = torch.zeros((*field.state.shape, len(Tokens)), requires_grad=True)
    return Prediction(
        address=ADDRESS,
        payload=TensorDict(
            {TensorKey.state: state, TensorKey.content: logits},
            batch_size=field.batch_size,
        ),
    )


def test_enum_requires_values_and_preserves_declared_order():
    with pytest.raises((TypeError, pydantic.ValidationError)):
        rf.Enum()
    with pytest.raises((TypeError, pydantic.ValidationError)):
        rf.Enum(VALUES)
    with pytest.raises((TypeError, pydantic.ValidationError)):
        rf.Model.xs(status=rf.Enum)

    request = rf.Enum(values=VALUES)
    assert request.values == VALUES
    assert request.p_unavailable == 0.01
    assert request.topk == []
    assert request.type == "enum"


@pytest.mark.parametrize(
    "values",
    [[], list(VALUES), set(VALUES), "pending", (), ("a", "a"), (None,), (1, "a"), (True, 1), (1, 2.0), ([1],)],
)
def test_enum_rejects_invalid_declarations(values):
    with pytest.raises((TypeError, ValueError)):
        rf.Enum(values=values)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_enum_rejects_nonfinite_declared_numbers(value):
    with pytest.raises((TypeError, ValueError)):
        rf.Enum(values=(value,))


@pytest.mark.parametrize(("declared", "observed"), [(0.0, -0.0), (-0.0, 0.0)])
def test_enum_signed_zero_uses_one_numeric_class_identity(declared, observed):
    configured = model((declared, 1.0))
    field = configured.encode(pa.table({"status": [observed]}), strata=rf.Strata.train)[ADDRESS]
    assert field.content.item() == 0
    assert rf.Enum.counts(configured, ADDRESS) == {declared: 1, 1.0: 0}
    with pytest.raises(ValueError):
        rf.Enum(values=(declared, observed))


@pytest.mark.parametrize("options", [{"p_unavailable": -0.1}, {"p_unavailable": 1.1}, {"topk": [0]}, {"topk": [1]}])
def test_enum_rejects_invalid_category_options(options):
    with pytest.raises(pydantic.ValidationError):
        rf.Enum(values=VALUES, **options)


@pytest.mark.parametrize("labels", [(False, True), (7, -4), (2.5, -7.25), ("z", "a"), (b"\xff", b"\x00")])
def test_enum_scalar_families_keep_stable_ids_and_schema_round_trips(labels):
    configured = model(labels)
    source = pa.table({"status": pa.chunked_array([[labels[1]], [labels[0], labels[1]]])})
    field = configured.encode(source, strata=rf.Strata.train)[ADDRESS]
    assert field.content.flatten().tolist() == [1, 0, 1]
    assert rf.Enum.vocabulary(configured, ADDRESS) == labels
    assert rf.Enum.vocabulary(configured.interprocess_encoding_context, ADDRESS) == labels
    assert rf.Enum.counts(configured, ADDRESS) == {labels[0]: 1, labels[1]: 2}

    for restored in (
        rf.Schema.model_validate(configured.schema.model_dump(round_trip=True)),
        rf.Schema.model_validate_json(configured.schema.model_dump_json(round_trip=True)),
    ):
        values = restored.requests[ADDRESS].values
        assert values == labels
        assert isinstance(values, tuple)
        assert [type(value) for value in values] == [type(value) for value in labels]


@pytest.mark.parametrize(
    ("labels", "column"),
    [
        ((1, 2), pa.array([1.0])),
        ((1.0, 2.0), pa.array([1])),
        ((True, False), pa.array([1])),
        (("a", "b"), pa.array([b"a"])),
        ((b"a", b"b"), pa.array(["a"])),
        (("a", "b"), pa.array([["a"]])),
    ],
)
def test_enum_rejects_values_from_a_different_scalar_family(labels, column):
    configured = model(labels)
    with pytest.raises((TypeError, ValueError), match="status"):
        configured.encode(pa.table({"status": column}), strata=rf.Strata.train)
    assert rf.Enum.counts(configured, ADDRESS) == dict.fromkeys(labels, 0)


@pytest.mark.parametrize("strata", [rf.Strata.train, rf.Strata.validate, rf.Strata.test, rf.Strata.predict])
@pytest.mark.parametrize("masked", [False, True])
def test_enum_rejects_undeclared_pristine_values_even_when_masked(strata, masked):
    configured = model(mask=masked)
    with pytest.raises(ValueError, match="status"):
        configured.encode(records("pending", "unknown"), strata=strata)
    assert rf.Enum.vocabulary(configured, ADDRESS) == VALUES
    assert rf.Enum.counts(configured, ADDRESS) == dict.fromkeys(VALUES, 0)


@pytest.mark.parametrize("owner", ["leaf", "branch", "root"])
def test_enum_repeated_masked_targets_preserve_nulls_padding_and_counts(owner):
    address = rf.Address("items", "status")
    configured = rf.Model.xs(
        d_model=8,
        n_heads=2,
        n_layers=1,
        mask=owner == "root",
        items=rf.Branch(
            length=3,
            mask=owner == "branch",
            status=rf.Enum(values=VALUES, mask=owner == "leaf", p_unavailable=0.0),
        ),
    )
    source = pa.Table.from_pylist(
        [
            {"items": []},
            {"items": [{"status": "rejected"}, {"status": None}, {"status": "complete"}]},
        ]
    )
    field = configured.encode(source, strata=rf.Strata.train)[address]
    assert field.state.shape == (2, 1, 3)
    assert not field.present.any()
    assert field.targets.batch_size == field.state.shape
    assert field.targets[TensorKey.state][0].eq(Tokens.padded).all()
    assert field.targets[TensorKey.state][1].flatten().tolist() == [Tokens.valued, Tokens.null, Tokens.valued]
    assert field.targets[TensorKey.content][1].flatten()[[0, 2]].tolist() == [2, 1]
    assert rf.Enum.counts(configured, address) == {"pending": 0, "complete": 1, "rejected": 1}


def test_enum_unavailable_input_augmentation_keeps_targets_and_counts():
    configured = model(p_unavailable=1.0, mask=rf.Mask(query="selected", reconstruct=True))
    source = records("pending", "rejected", None).append_column("selected", pa.array([False, True, False]))
    field = configured.encode(source, strata=rf.Strata.train)[ADDRESS]
    assert field.content.flatten()[0].item() == -1
    assert field.state.flatten().tolist() == [Tokens.valued, Tokens.masked, Tokens.null]
    assert field.targets[TensorKey.content].flatten()[1].item() == 2
    assert field.targets[TensorKey.state].flatten()[1].item() == Tokens.valued
    assert rf.Enum.counts(configured, ADDRESS) == {"pending": 1, "complete": 0, "rejected": 1}

    for strata in (rf.Strata.validate, rf.Strata.test, rf.Strata.predict):
        source = records("pending", "complete").append_column("selected", pa.array([False, False]))
        field = configured.encode(source, strata=strata)[ADDRESS]
        assert field.content.flatten().tolist() == [0, 1]
    assert rf.Enum.counts(configured, ADDRESS) == {"pending": 1, "complete": 0, "rejected": 1}


@pytest.mark.parametrize("workers", [0, 1])
def test_enum_prefetch_keeps_ids_fixed_and_counts_only_consumed_batches(workers):
    configured = model(mask=True)
    data = rf.ArrowDataModule(
        model=configured,
        train=records("rejected", "complete", "pending", "pending"),
        shuffle=False,
        num_workers=workers,
        persistent_workers=False,
        pin_memory=False,
    )
    batches = list(data.train_dataloader())
    assert len(batches) == 2
    assert rf.Enum.counts(configured, ADDRESS) == dict.fromkeys(VALUES, 0)
    assert batches[0].tensors[ADDRESS].targets[TensorKey.content].flatten().tolist() == [2, 1]
    assert batches[1].tensors[ADDRESS].targets[TensorKey.content].flatten().tolist() == [0, 0]

    consumed = bind(configured, batches[0], rf.Strata.train)
    ModelRuntime.learn(configured, consumed.observations, strata=rf.Strata.train)
    assert not consumed.bindings
    assert rf.Enum.counts(configured, ADDRESS) == {"pending": 0, "complete": 1, "rejected": 1}
    assert rf.Enum.vocabulary(configured, ADDRESS) == VALUES


def test_enum_output_includes_declared_unobserved_classes_and_caps_topk():
    configured = model(mask=True, topk=[8, 2, 8])
    field = configured.encode(records("pending", "pending"), strata=rf.Strata.train)[ADDRESS]
    logits = torch.tensor([[[0.0, 1.0, 3.0]], [[0.0, 1.0, 3.0]]])
    predicted = prediction(field, logits)
    extension = rf.TENSORFIELDS["enum"]
    result = extension.write(configured, predicted, extension.output(configured, ADDRESS)).field("content")
    assert configured.schema.requests[ADDRESS].topk == [2, 8]
    for row in result.to_pylist():
        assert row["value"] == "rejected"
        assert row["probability"] == pytest.approx(logits[0, 0].softmax(-1)[2].item())
        assert [candidate["value"] for candidate in row["topk"]] == ["rejected", "complete", "pending"]
        assert sum(candidate["probability"] for candidate in row["topk"]) == pytest.approx(1.0)
    assert rf.Enum.counts(configured, ADDRESS)["rejected"] == 0


def test_enum_loss_supervises_declared_labels_and_excludes_null_content(monkeypatch):
    configured = model(mask=True, topk=[2])
    field = configured.encode(records("complete", None), strata=rf.Strata.train)[ADDRESS]
    logits = torch.tensor([[[1.0, 2.0, 0.0]], [[20.0, -40.0, 10.0]]], requires_grad=True)
    predicted = prediction(field, logits)
    monkeypatch.setattr(configured, "track", lambda names, value: value)
    objective = rf.TENSORFIELDS["enum"].loss(configured, predicted, field, rf.Strata.train)
    expected = torch.log(torch.tensor(float(len(Tokens)))) + torch.nn.functional.cross_entropy(
        logits[0], torch.tensor([1])
    )
    torch.testing.assert_close(objective, expected)
    objective.backward()
    assert logits.grad[0].abs().sum() > 0
    assert logits.grad[1].eq(0).all()
    metrics = configured.nodes[ADDRESS].decoder.metrics["train_metrics"].compute()
    assert metrics["targets.known"].item() == 1
    assert metrics["targets.unavailable"].item() == 0
    assert metrics["coverage.content"].item() == 1.0
    assert metrics["accuracy.content"].item() == 1.0


def test_enum_all_null_targets_have_finite_state_loss_and_zero_content_gradient(monkeypatch):
    configured = model(mask=True)
    field = configured.encode(records(None, None), strata=rf.Strata.train)[ADDRESS]
    logits = torch.zeros((*field.state.shape, len(VALUES)), requires_grad=True)
    predicted = prediction(field, logits)
    monkeypatch.setattr(configured, "track", lambda names, value: value)
    objective = rf.TENSORFIELDS["enum"].loss(configured, predicted, field, rf.Strata.train)
    assert torch.isfinite(objective)
    objective.backward()
    assert logits.grad is not None
    assert logits.grad.eq(0).all()
    assert rf.Enum.counts(configured, ADDRESS) == dict.fromkeys(VALUES, 0)


def test_enum_empty_prediction_and_single_class_are_well_defined():
    configured = model(("only",), mask=True, topk=[2])
    empty = configured.predict(records())
    assert empty.num_rows == 0
    assert (
        empty.schema.field("predictions").type.field(str(ADDRESS)).type.field("content").type.field("value").type
        == pa.large_string()
    )
    field = configured.encode(records("only"), strata=rf.Strata.train)[ADDRESS]
    extension = rf.TENSORFIELDS["enum"]
    result = (
        extension.write(
            configured,
            prediction(field, torch.zeros((*field.state.shape, 1))),
            extension.output(configured, ADDRESS),
        )
        .field("content")[0]
        .as_py()
    )
    assert result == {"value": "only", "probability": 1.0, "topk": [{"value": "only", "probability": 1.0}]}


def test_enum_checkpoint_and_unchanged_rebuild_preserve_weights_and_counts(tmp_path: Path):
    configured = model(mask=True)
    configured.encode(records("pending", "rejected"), strata=rf.Strata.train)
    node = configured.nodes[ADDRESS]
    assert node.embedder.embeddings[TensorKey.content.name].num_embeddings == len(VALUES)
    assert node.decoder.linears[TensorKey.content.name].out_features == len(VALUES)
    with torch.no_grad():
        node.embedder.embeddings[TensorKey.content.name].weight.fill_(12.5)
        node.decoder.linears[TensorKey.content.name].weight.fill_(7.25)
    pathname = tmp_path / "enum.ckpt"
    configured.save(pathname)
    restored = rf.Model.load(pathname)
    restored.extend(other=rf.Number)
    restored.update(rf.where("address") == ADDRESS, topk=[2])

    assert rf.Enum.vocabulary(restored, ADDRESS) == VALUES
    assert rf.Enum.counts(restored, ADDRESS) == {"pending": 1, "complete": 0, "rejected": 1}
    node = restored.nodes[ADDRESS]
    assert node.embedder.embeddings[TensorKey.content.name].weight.eq(12.5).all()
    assert node.decoder.linears[TensorKey.content.name].weight.eq(7.25).all()
    field = restored.encode(pa.table({"status": ["rejected"], "other": [0.0]}), strata=rf.Strata.train)[ADDRESS]
    assert field.targets[TensorKey.content].item() == 2


@pytest.mark.parametrize("labels", [VALUES[::-1], ("pending", "complete", "other"), ("complete",)])
def test_enum_changing_declaration_resets_class_rows_and_exposure(labels):
    configured = model(mask=True)
    configured.encode(records("pending", "rejected"), strata=rf.Strata.train)
    with torch.no_grad():
        configured.nodes[ADDRESS].embedder.embeddings[TensorKey.content.name].weight.fill_(100.0)
        configured.nodes[ADDRESS].decoder.linears[TensorKey.content.name].weight.fill_(200.0)

    configured.update(rf.where("address") == ADDRESS, values=labels)
    node = configured.nodes[ADDRESS]
    assert rf.Enum.vocabulary(configured, ADDRESS) == labels
    assert rf.Enum.counts(configured, ADDRESS) == dict.fromkeys(labels, 0)
    assert node.embedder.embeddings[TensorKey.content.name].weight.shape[0] == len(labels)
    assert node.decoder.linears[TensorKey.content.name].weight.shape[0] == len(labels)
    assert not node.embedder.embeddings[TensorKey.content.name].weight.eq(100.0).all()
    assert not node.decoder.linears[TensorKey.content.name].weight.eq(200.0).all()


def test_enum_rejects_stale_prefetched_class_ids_after_reordering():
    configured = model(mask=True)
    pending = encode(
        records("pending", "rejected"), configured.schema, rf.Strata.train, configured.interprocess_encoding_context
    )
    configured.update(rf.where("address") == ADDRESS, values=VALUES[::-1])
    with pytest.raises(ValueError, match="status"):
        bind(configured, pending, rf.Strata.train)
    assert rf.Enum.counts(configured, ADDRESS) == dict.fromkeys(VALUES, 0)


def test_enum_rejects_checkpoint_weights_for_different_class_meanings():
    source = model(mask=True)
    destination = model(VALUES[::-1], mask=True)
    with pytest.raises((RuntimeError, ValueError), match="[Ee]num|values|vocabulary|declar"):
        destination.load_state_dict(source.state_dict())


def test_enum_introspection_rejects_wrong_fields_and_sources():
    configured = rf.Model.xs(status=rf.Enum(values=VALUES), amount=rf.Number)
    for inspect in (rf.Enum.vocabulary, rf.Enum.counts):
        with pytest.raises(KeyError, match="missing"):
            inspect(configured, rf.Address("missing"))
        with pytest.raises(TypeError, match="amount"):
            inspect(configured, rf.Address("amount"))
        with pytest.raises(TypeError):
            inspect(object(), ADDRESS)


def test_enum_lightning_training_preserves_fixed_classes_and_training_counts():
    torch.manual_seed(923)
    configured = rf.Model.xs(
        d_model=8,
        n_heads=2,
        n_layers=1,
        batch_size=64,
        amount=rf.Number,
        status=rf.Enum(values=VALUES, mask=True, topk=[2]),
    )
    configured.optimizer = rf.adamw(learning_rate=0.01, weight_decay=0.0, fused=False)
    amounts = torch.linspace(0.0, 1.0, 64).numpy()
    train = pa.table({"amount": amounts, "status": ["pending"] * 32 + ["complete"] * 32})
    validate = pa.table({"amount": amounts, "status": ["rejected"] * 64})
    before = configured.nodes[ADDRESS].decoder.linears[TensorKey.content.name].weight.detach().clone()
    losses = []
    gradients = []

    class Audit(lit.Callback):
        def on_after_backward(self, trainer, pl_module):
            gradient = pl_module.nodes[ADDRESS].decoder.linears[TensorKey.content.name].weight.grad
            assert gradient is not None
            gradients.append(gradient.detach().clone())

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            losses.append(outputs["loss"].detach().clone())
            assert not batch.bindings

    data = rf.ArrowDataModule(
        model=configured,
        train=train,
        validate=validate,
        shuffle=False,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    trainer = lit.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        callbacks=[Audit()],
    )
    trainer.fit(configured, datamodule=data)

    assert len(losses) == len(gradients) == 1
    assert torch.isfinite(losses[0])
    assert torch.isfinite(gradients[0]).all()
    assert gradients[0].abs().sum() > 0
    node = configured.nodes[ADDRESS]
    assert not torch.equal(node.decoder.linears[TensorKey.content.name].weight, before)
    assert node.embedder.embeddings[TensorKey.content.name].num_embeddings == len(VALUES)
    assert node.decoder.linears[TensorKey.content.name].out_features == len(VALUES)
    assert rf.Enum.vocabulary(configured, ADDRESS) == VALUES
    expected_counts = {"pending": 32, "complete": 32, "rejected": 0}
    assert rf.Enum.counts(configured, ADDRESS) == expected_counts
    output = configured.predict(validate)["predictions"].combine_chunks().field(str(ADDRESS)).field("content")
    assert len(output) == 64
    assert set(output.field("value").to_pylist()) <= set(VALUES)
    probabilities = torch.from_numpy(output.field("probability").to_numpy().copy())
    assert torch.isfinite(probabilities).all()
    assert probabilities.gt(0).all() and probabilities.le(1).all()
    assert rf.Enum.counts(configured, ADDRESS) == expected_counts
