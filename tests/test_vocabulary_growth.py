"""Vocabulary admission belongs to batch consumption, including after storage grows."""

from collections import Counter
from copy import deepcopy
from typing import Literal

import pyarrow as pa
import pytest
import torch
from lightning.pytorch import Callback, Trainer

import relflow as rf
from relflow.architecture.binding import bind
from relflow.data.iterables import encode
from relflow.tensorfields.shared.vocabulary import VocabularyBatch

ADDRESS = rf.Address("/value")
KINDS = ("category", "set", "cluster")
REQUESTS = {"category": rf.Category, "set": rf.Set, "cluster": rf.Cluster}


def build(kind):
    options = {"p_unavailable": 0.0, "mask": rf.Mask(query="supervise", reconstruct=True)}
    request = (
        rf.Cluster(bounds=3, revive_temperature=0.0, **options) if kind == "cluster" else REQUESTS[kind](**options)
    )
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        reduction=rf.Mean(),
        batch_size=2,
        embed=True,
        signal=rf.Number,
        value=request,
    )


def table(kind, labels, *, supervise=False):
    values = [[label] if label is not None else None for label in labels] if kind == "set" else labels
    return pa.table({"value": values, "signal": [1.0] * len(labels), "supervise": [supervise] * len(labels)})


def assert_counts(model, kind, expected):
    vocabulary = REQUESTS[kind].vocabulary(model, ADDRESS)
    assert set(vocabulary) == set(expected)
    counter = model.nodes[ADDRESS].embedder.counters["content"]
    # Counters contain one prior per row; unused storage and the Cluster sentinel
    # must not receive observations when a local dictionary is remapped.
    counts = torch.ones_like(counter.counts)
    for index, label in enumerate(vocabulary):
        counts[index] += expected[label]
    torch.testing.assert_close(counter.counts, counts, rtol=0, atol=0)
    if kind == "category":
        assert rf.Category.counts(model, ADDRESS) == expected


def assert_shapes(model, kind, capacity):
    node = model.nodes[ADDRESS]
    assert node.embedder.vocab.size == capacity
    counter = node.embedder.counters["content"]
    rows = capacity + (kind == "cluster")
    assert counter.size == rows
    assert counter.counts.shape == (rows,)
    if kind == "cluster":
        embedding = node.embedder.embeddings["cluster"]
        assert embedding.weight.shape == (rows, 3)
        assert node.embedder.size == 3
        assert node.embedder.embeddings["content"].weight.shape == (8, 3)
        assert node.decoder.linears["cluster"].weight.shape == (3, 8)
        assert node.embedder.committed.shape == node.embedder.usage_ema.shape == (3,)
    else:
        embedding = node.embedder.embeddings["content"]
        assert embedding.weight.shape == (rows, 8)
        linear = node.decoder.linears["content"]
        assert linear.out_features == rows
        assert linear.weight.shape == (rows, 8)
        assert linear.bias.shape == (rows,)
    assert embedding.num_embeddings == rows


def assert_encoding(model, kind, field, labels, *, inputs=True):
    vocabulary = REQUESTS[kind].vocabulary(model, ADDRESS)
    capacity = model.nodes[ADDRESS].embedder.vocab.size
    contents = [
        (field.targets[rf.TensorKey.content], field.targets[rf.TensorKey.state].reshape(-1).eq(rf.Tokens.valued))
    ]
    if inputs:
        contents.append((field.content, field.state.reshape(-1).eq(rf.Tokens.valued)))
        assert field.state.reshape(-1).eq(rf.Tokens.valued).all()
    if kind == "set":
        membership = torch.zeros(len(labels), capacity)
        unavailable = torch.zeros(len(labels), dtype=torch.int64)
        for row, label in enumerate(labels):
            if label in vocabulary:
                membership[row, vocabulary.index(label)] = 1
            else:
                unavailable[row] = 1
        for content, valued in contents:
            torch.testing.assert_close(content["membership"].reshape(len(labels), capacity)[valued], membership[valued])
            torch.testing.assert_close(content["unavailable"].reshape(-1)[valued], unavailable[valued])
    else:
        unavailable = model.interprocess_encoding_context[ADDRESS].unavailable_index
        expected = [vocabulary.index(label) if label in vocabulary else unavailable for label in labels]
        for content, valued in contents:
            assert content.reshape(-1)[valued].tolist() == torch.tensor(expected)[valued].tolist()


def assert_state(actual, expected):
    assert actual.keys() == expected.keys()
    for name, value in actual.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, expected[name], rtol=0, atol=0)
        else:
            assert value == expected[name]


@pytest.mark.parametrize("kind", KINDS)
def test_train_encode_grows_geometrically_preserving_ids_and_rows(kind):
    model = build(kind)
    schema = model.schema.model_dump(mode="json")
    node = model.nodes[ADDRESS]
    initial_status = rf.Cluster.status(model, ADDRESS) if kind == "cluster" else None
    seen = Counter()
    vocabulary = ()

    for labels, capacity in ((["a", "b", "a"], 2), (["c", "a"], 4), (["d", "e", "b"], 8)):
        previous = {name: value.detach().clone() for name, value in node.named_parameters()}
        old_capacity = node.embedder.vocab.size
        encoded = model.encode(table(kind, labels), strata="train")
        current = REQUESTS[kind].vocabulary(model, ADDRESS)
        assert current[: len(vocabulary)] == vocabulary
        vocabulary = current
        seen.update(labels)

        assert_shapes(model, kind, capacity)
        assert_encoding(model, kind, encoded[ADDRESS], labels)
        assert_counts(model, kind, seen)
        assert model.schema.model_dump(mode="json") == schema
        for name, value in node.named_parameters():
            old = previous[name]
            if kind == "cluster" and name == "embedder.embeddings.cluster.weight":
                torch.testing.assert_close(value[:old_capacity], old[:old_capacity], rtol=0, atol=0)
                torch.testing.assert_close(value[-1], old[-1], rtol=0, atol=0)
            else:
                torch.testing.assert_close(value[: old.shape[0]], old, rtol=0, atol=0)
        if kind == "cluster":
            assert rf.Cluster.status(model, ADDRESS) == initial_status

    assert vocabulary == ("a", "b", "c", "d", "e")
    assert_encoding(model, kind, model.encode(table(kind, ["b", "a"]))[ADDRESS], ["b", "a"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("order", [(0, 1, 2, 0), (1, 0, 2, 1)], ids=["forward", "reordered"])
@pytest.mark.parametrize("supervise", [False, True], ids=["inputs", "targets"])
def test_independent_prefetched_batches_can_be_rebound_without_mutation(kind, order, supervise):
    model = build(kind)
    labels = (["b", "a", "b"], ["c", "b", "d"], ["e", "c", "a"])
    prefetched = []
    for values in labels:
        worker = build(kind)
        batch = encode(
            table(kind, values, supervise=supervise),
            worker.schema,
            rf.Strata.train,
            worker.interprocess_encoding_context,
        )
        assert isinstance(batch.bindings[ADDRESS], VocabularyBatch)
        assert batch.bindings[ADDRESS].labels.to_pylist() == list(dict.fromkeys(values))
        assert REQUESTS[kind].vocabulary(worker, ADDRESS) == ()
        prefetched.append(batch)
    snapshots = deepcopy(prefetched)
    expected = Counter()
    vocabulary = []

    for step, index in enumerate(order):
        batch = prefetched[index]
        bound = bind(model, batch, rf.Strata.train)
        assert not bound.bindings
        assert bound.source is batch.source
        assert bind(model, bound, rf.Strata.train) is bound
        for label in labels[index]:
            if label not in vocabulary:
                vocabulary.append(label)
        assert REQUESTS[kind].vocabulary(model, ADDRESS) == tuple(vocabulary)
        assert_encoding(model, kind, bound.tensors[ADDRESS], labels[index], inputs=not supervise)
        # Binding resolves IDs and observations; only consuming the batch learns.
        assert_counts(model, kind, {label: expected[label] for label in vocabulary})
        model.training_step(bound, step)
        expected.update(labels[index])
        assert_counts(model, kind, expected)

        pristine = snapshots[index]
        assert batch.bindings[ADDRESS].labels.equals(pristine.bindings[ADDRESS].labels)
        assert batch.bindings[ADDRESS].size == pristine.bindings[ADDRESS].size
        assert (batch.tensors == pristine.tensors).all()
        for address, observation in batch.observations.items():
            assert (observation == pristine.observations[address]).all()

    assert_shapes(model, kind, 8)


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("strata", [rf.Strata.validate, rf.Strata.test, rf.Strata.predict])
def test_prefetched_evaluation_unknowns_remain_unknown_without_admission(kind, strata):
    model = build(kind)
    model.encode(table(kind, ["a", "b"]), strata="train")
    source = table(kind, ["never-trained", "a"])
    prefetched = encode(source, model.schema, strata, model.interprocess_encoding_context)
    assert not prefetched.observations
    original = deepcopy(prefetched)

    for capacity, new in ((2, []), (4, ["c"]), (8, ["d", "e"])):
        if new:
            model.encode(table(kind, new), strata="train")
        state = deepcopy(model.state_dict())
        bound = bind(model, prefetched, strata)
        assert_encoding(model, kind, bound.tensors[ADDRESS], ["never-trained", "a"])
        assert_encoding(model, kind, model.encode(source, strata=strata)[ADDRESS], ["never-trained", "a"])
        assert_shapes(model, kind, capacity)
        prediction = model.predict(table(kind, ["never-trained"], supervise=True))
        assert prediction.num_rows == 1
        assert "predictions" in prediction.column_names
        assert "never-trained" not in REQUESTS[kind].vocabulary(model, ADDRESS)
        assert_state(model.state_dict(), state)
        assert (prefetched.tensors == original.tensors).all()


def test_set_rebinding_preserves_distinct_unknown_members_and_pristine_exposure():
    model = build("set")
    model.encode(table("set", ["a", "b"]), strata="train")
    source = pa.table(
        {
            "value": [["a", "novel", "novel", "other"], [], None],
            "signal": [1.0] * 3,
            "supervise": [False] * 3,
        }
    )
    prefetched = encode(source, model.schema, rf.Strata.validate, model.interprocess_encoding_context)
    model.encode(table("set", ["c", "d", "e"]), strata="train")
    bound = bind(model, prefetched, rf.Strata.validate).tensors[ADDRESS]
    assert bound.content["membership"].reshape(3, 8).sum(-1).tolist() == [1, 0, 0]
    assert bound.content["membership"].reshape(3, 8)[0, 0] == 1
    assert bound.content["unavailable"].reshape(-1).tolist() == [2, 0, 0]
    assert bound.state.reshape(-1).tolist() == [rf.Tokens.valued, rf.Tokens.valued, rf.Tokens.null]
    assert_counts(model, "set", dict.fromkeys("abcde", 1))

    model.encode(source, strata="train")
    assert_counts(model, "set", {"a": 2, "b": 1, "c": 1, "d": 1, "e": 1, "novel": 2, "other": 1})


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("format", ["checkpoint", "state_dict"])
def test_grown_checkpoint_restores_into_initial_schema_and_can_grow_again(kind, format, tmp_path):
    model = build(kind)
    schema = model.schema.model_dump(mode="json")
    labels = ["a", "b", "c", "d", "e", "a"]
    model.encode(table(kind, labels), strata="train")
    assert_shapes(model, kind, 8)
    prediction_source = table(kind, ["a", "e", "unknown"], supervise=True)
    prediction = model.predict(prediction_source)
    path = tmp_path / "grown.pt"

    if format == "checkpoint":
        model.save(path)
        saved = torch.load(path, weights_only=False, map_location="cpu")
        assert saved["schema"] == schema
        restored = rf.Model.load(path)
    else:
        torch.save(model.state_dict(), path)
        restored = build(kind)
        assert_shapes(restored, kind, 1)
        result = restored.load_state_dict(torch.load(path, weights_only=True, map_location="cpu"), strict=True)
        assert not result.missing_keys and not result.unexpected_keys

    assert restored.schema.model_dump(mode="json") == schema
    assert_shapes(restored, kind, 8)
    assert_counts(restored, kind, Counter(labels))
    assert_state(restored.state_dict(), model.state_dict())
    assert restored.predict(prediction_source).equals(prediction)
    probe = ["e", "a", "unknown"]
    assert_encoding(restored, kind, restored.encode(table(kind, probe))[ADDRESS], probe)

    more = ["f", "g", "h", "i", "a"]
    restored.encode(table(kind, more), strata="train")
    assert_shapes(restored, kind, 16)
    assert REQUESTS[kind].vocabulary(restored, ADDRESS) == tuple("abcdefghi")
    assert_counts(restored, kind, Counter(labels + more))
    assert restored.schema.model_dump(mode="json") == schema
    assert_shapes(model, kind, 8)
    assert_counts(model, kind, Counter(labels))


@pytest.mark.parametrize("kind", KINDS)
def test_retained_encoded_unknown_still_forwards_as_unknown_after_growth(kind):
    model = build(kind).eval()
    model.encode(table(kind, ["a", "b"]), strata="train")
    source = table(kind, ["never-trained", "a"])
    retained = model.encode(source)
    pristine = retained.clone()
    if kind != "set":
        assert retained[ADDRESS].content.reshape(-1)[0].item() == -1

    with torch.no_grad():
        before = model(retained, strata="predict")
    model.encode(table(kind, ["c", "d", "e"]), strata="train")
    fresh = model.encode(source)
    if kind == "set":
        assert retained[ADDRESS].content["membership"].shape[-1] == 2
        assert fresh[ADDRESS].content["membership"].shape[-1] == 8
    with torch.no_grad():
        after = model(retained, strata="predict")
        current = model(fresh, strata="predict")

    assert (retained == pristine).all()
    actual = next(prediction.payload[rf.TensorKey.embedding] for prediction in after if prediction.address == "/")
    for predictions in (before, current):
        expected = next(
            prediction.payload[rf.TensorKey.embedding] for prediction in predictions if prediction.address == "/"
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert "never-trained" not in REQUESTS[kind].vocabulary(model, ADDRESS)


def test_retained_set_targets_still_train_after_membership_columns_grow():
    model = build("set")
    source = table("set", ["a", "b"], supervise=True)
    retained = model.encode(source, strata="train")
    pristine = retained.clone()
    model.encode(table("set", ["c", "d", "e"]), strata="train")
    assert retained[ADDRESS].targets[rf.TensorKey.content]["membership"].shape[-1] == 2
    fresh = model.encode(source, strata="validate")

    model.eval()
    old_loss = model.training_step(retained, 0)["loss"]
    new_loss = model.training_step(fresh, 0)["loss"]
    assert torch.isfinite(old_loss)
    torch.testing.assert_close(old_loss, new_loss)
    old_loss.backward()
    gradient = model.nodes[ADDRESS].decoder.linears["content"].weight.grad
    assert gradient is not None and gradient.shape == (8, 8)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0
    assert (retained == pristine).all()
    assert_counts(model, "set", dict.fromkeys("abcde", 1))


def test_late_registered_category_subclass_uses_its_registered_bind_component():
    category = rf.TENSORFIELDS["category"]
    name = "vocabulary_growth_label"
    assert name not in rf.TENSORFIELDS
    extension = rf.Extension(name=name, types=(str,))
    calls = []
    try:

        @extension.register
        class Request(rf.Category):
            type: Literal["vocabulary_growth_label"] = "vocabulary_growth_label"

        for component, implementation in category.components.items():
            if component not in {rf.Component.Request, rf.Component.bind}:
                extension.register(implementation, component)

        @extension.register
        def bind(module, field, observation, binding, *, address, strata, resize):
            assert module.schema.requests[address].type == name
            assert isinstance(binding, VocabularyBatch)
            calls.append((strata, binding.labels.to_pylist(), observation is not None))
            category.component(rf.Component.bind)(
                module, field, observation, binding, address=address, strata=strata, resize=resize
            )

        model = rf.Model(
            d_model=8,
            n_layers=1,
            n_heads=2,
            value=Request(p_unavailable=0.0, mask=True),
        )
        fields = model.encode(table("category", ["a", "b", "c"]), strata="train")
        assert_shapes(model, "category", 4)
        assert_counts(model, "category", dict.fromkeys("abc", 1))
        assert_encoding(model, "category", fields[ADDRESS], ["a", "b", "c"], inputs=False)
        model.encode(table("category", ["never-trained"]), strata="validate")
        assert calls == [(rf.Strata.train, ["a", "b", "c"], True), (rf.Strata.validate, ["never-trained"], False)]
        assert_counts(model, "category", dict.fromkeys("abc", 1))
    finally:
        rf.TENSORFIELDS.pop(name)


class TrainingTrace(Callback):
    """Record optimizer progress and worker identity across late admissions."""

    def __init__(self, kind):
        self.kind = kind
        self.batches = []
        self.workers = []
        self.optimizers = []
        self.weights = []

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        assert not batch.bindings
        labels = batch.source["value"].to_pylist()
        if self.kind == "set":
            labels = [value[0] for value in labels]
        assert_encoding(pl_module, self.kind, batch.tensors[ADDRESS], labels, inputs=False)
        self.batches.append((trainer.current_epoch, batch_idx, tuple(labels)))
        self.optimizers.append(trainer.optimizers[0])
        if trainer.train_dataloader.num_workers:
            self.workers.append(tuple(worker.pid for worker in trainer.train_dataloader._iterator._workers))
        node = pl_module.nodes[ADDRESS]
        parameter = (
            node.embedder.embeddings["cluster"].weight
            if self.kind == "cluster"
            else node.decoder.linears["content"].weight
        )
        self.weights.append(parameter.detach().clone())

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        assert outputs is not None and torch.isfinite(outputs["loss"])
        node = pl_module.nodes[ADDRESS]
        parameter = (
            node.embedder.embeddings["cluster"].weight
            if self.kind == "cluster"
            else node.decoder.linears["content"].weight
        )
        optimizer = trainer.optimizers[0]
        assert any(parameter is item for group in optimizer.param_groups for item in group["params"])
        assert optimizer.state[parameter]["step"].item() == batch_idx + 1
        assert optimizer.state[parameter]["exp_avg"].shape == parameter.shape
        assert optimizer.state[parameter]["exp_avg_sq"].shape == parameter.shape
        assert not torch.equal(parameter, self.weights[-1])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("workers", [0, 2], ids=["main-process", "persistent-workers"])
def test_one_fit_admits_late_labels_without_restarting_epoch_or_optimizer(kind, workers, tmp_path):
    model = build(kind)
    model.optimizer = rf.adamw(learning_rate=1e-3)
    labels = ["a", "b", "c", "a", "d", "e", "f", "b"]
    data = rf.ArrowDataModule(
        model=model,
        train=table(kind, labels, supervise=True),
        validate=table(kind, ["validation-only", "a"], supervise=True),
        shuffle=False,
        num_workers={"train": workers, "validate": 0},
        persistent_workers={"train": bool(workers), "validate": False},
        pin_memory=False,
    )
    trace = TrainingTrace(kind)
    trainer = Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=1,
        limit_train_batches=4,
        limit_val_batches=1,
        num_sanity_val_steps=0,
        reload_dataloaders_every_n_epochs=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        callbacks=[trace],
        default_root_dir=tmp_path,
    )
    trainer.fit(model, datamodule=data)

    assert trainer.global_step == 4
    assert [(epoch, index) for epoch, index, _ in trace.batches] == [(0, index) for index in range(4)]
    consumed = [label for _, _, batch_labels in trace.batches for label in batch_labels]
    assert Counter(consumed) == Counter(labels)
    assert all(optimizer is trace.optimizers[0] for optimizer in trace.optimizers)
    assert [weight.shape[0] for weight in trace.weights] == ([3, 5, 9, 9] if kind == "cluster" else [2, 4, 8, 8])
    if workers:
        assert len(trace.workers) == 4
        assert len(set(trace.workers[0])) == workers
        assert all(pids == trace.workers[0] for pids in trace.workers)
    assert REQUESTS[kind].vocabulary(model, ADDRESS) == tuple(dict.fromkeys(consumed))
    assert_shapes(model, kind, 8)
    assert_counts(model, kind, Counter(labels))


@pytest.mark.parametrize("kind", KINDS)
def test_lightning_checkpoint_resume_restores_training_state_before_further_growth(kind, tmp_path):
    def schedule(module, optimizer):
        return {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5), "interval": "step"}

    model = build(kind)
    model.optimizer = rf.adamw(learning_rate=1e-3, fused=False)
    model.scheduler = schedule
    schema = model.schema.model_dump(mode="json")
    labels = ["a", "b", "c", "a"]
    data_options = dict(shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False)
    data = rf.ArrowDataModule(model=model, train=table(kind, labels, supervise=True), **data_options)
    trainer_options = dict(
        accelerator="cpu",
        devices=1,
        limit_train_batches=2,
        limit_val_batches=0,
        num_sanity_val_steps=0,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        default_root_dir=tmp_path,
    )
    trainer = Trainer(max_epochs=1, **trainer_options)
    trainer.fit(model, datamodule=data)
    assert trainer.global_step == 2
    assert_shapes(model, kind, 4)
    assert_counts(model, kind, Counter(labels))

    path = tmp_path / "training.ckpt"
    trainer.save_checkpoint(path)
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    assert checkpoint["schema"] == schema
    assert checkpoint["global_step"] == 2
    assert len(checkpoint["optimizer_states"]) == len(checkpoint["lr_schedulers"]) == 1
    optimizer = trainer.optimizers[0]
    saved_state = deepcopy(model.state_dict())
    weights = {name: value.detach().clone() for name, value in model.named_parameters()}
    moments = {name: deepcopy(optimizer.state.get(value, {})) for name, value in model.named_parameters()}
    scheduler_state = deepcopy(trainer.lr_scheduler_configs[0].scheduler.state_dict())
    assert scheduler_state["last_epoch"] == 2
    rates = [group["lr"] for group in optimizer.param_groups]
    assert rates == pytest.approx([1e-3 * 0.5**2] * len(rates))
    growing = (
        "nodes./value.embedder.embeddings.cluster.weight"
        if kind == "cluster"
        else "nodes./value.decoder.linears.content.weight"
    )
    assert moments[growing]["step"].item() == 2
    assert moments[growing]["exp_avg"].count_nonzero() > 0
    assert moments[growing]["exp_avg_sq"].count_nonzero() > 0

    class ResumeTrace(Callback):
        def __init__(self):
            self.restored = False
            self.batches = []

        def on_train_start(self, trainer, pl_module):
            assert trainer.global_step == 2
            assert trainer.current_epoch == 1
            assert_shapes(pl_module, kind, 4)
            assert_counts(pl_module, kind, Counter(labels))
            assert REQUESTS[kind].vocabulary(pl_module, ADDRESS) == ("a", "b", "c")
            assert_state(pl_module.state_dict(), saved_state)
            self.optimizer = trainer.optimizers[0]
            self.scheduler = trainer.lr_scheduler_configs[0].scheduler
            assert self.scheduler.optimizer is self.optimizer
            assert self.scheduler.state_dict() == scheduler_state
            assert [group["lr"] for group in self.optimizer.param_groups] == rates
            optimized = {id(value) for group in self.optimizer.param_groups for value in group["params"]}
            assert optimized == {id(value) for value in pl_module.parameters() if value.requires_grad}
            for name, value in pl_module.named_parameters():
                assert_state(self.optimizer.state.get(value, {}), moments[name])
            self.restored = True

        def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
            assert self.restored
            self.batches.append(batch_idx)
            assert trainer.current_epoch == 1
            assert trainer.global_step == 2 + batch_idx
            assert trainer.optimizers[0] is self.optimizer
            assert trainer.lr_scheduler_configs[0].scheduler is self.scheduler
            assert self.scheduler.optimizer is self.optimizer
            assert not batch.bindings
            assert_shapes(pl_module, kind, 8)
            assert_encoding(
                pl_module, kind, batch.tensors[ADDRESS], more[2 * batch_idx : 2 * batch_idx + 2], inputs=False
            )
            if batch_idx != 0:
                return

            assert REQUESTS[kind].vocabulary(pl_module, ADDRESS) == tuple("abcde")
            assert_counts(pl_module, kind, {"a": 2, "b": 1, "c": 1, "d": 0, "e": 0})
            assert self.scheduler.state_dict() == scheduler_state
            assert [group["lr"] for group in self.optimizer.param_groups] == rates
            # Admission precedes this hook, but no resumed optimizer step has
            # occurred: old rows and moments must be exact, including the tail.
            for name, value in pl_module.named_parameters():
                old = weights[name]
                current = self.optimizer.state.get(value, {})
                previous = moments[name]
                assert current.keys() == previous.keys()
                if value.shape == old.shape:
                    torch.testing.assert_close(value, old, rtol=0, atol=0)
                    assert_state(current, previous)
                    continue
                destinations = torch.arange(old.shape[0])
                if kind == "cluster" and name == growing:
                    destinations[-1] = value.shape[0] - 1
                torch.testing.assert_close(value[destinations], old, rtol=0, atol=0)
                introduced = torch.ones(value.shape[0], dtype=torch.bool)
                introduced[destinations] = False
                for key, tensor in previous.items():
                    if key == "step":
                        torch.testing.assert_close(current[key], tensor, rtol=0, atol=0)
                    else:
                        torch.testing.assert_close(current[key][destinations], tensor, rtol=0, atol=0)
                        assert current[key][introduced].count_nonzero() == 0

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            assert outputs is not None and torch.isfinite(outputs["loss"])
            value = dict(pl_module.named_parameters())[growing]
            state = self.optimizer.state[value]
            assert state["step"].item() == trainer.global_step == 3 + batch_idx
            assert self.scheduler.last_epoch == trainer.global_step
            assert state["exp_avg"].shape == state["exp_avg_sq"].shape == value.shape
            assert state["exp_avg"][3:5].count_nonzero() > 0
            assert state["exp_avg_sq"][3:5].count_nonzero() > 0

    restored = build(kind)
    restored.optimizer = rf.adamw(learning_rate=1e-3, fused=False)
    restored.scheduler = schedule
    assert restored.schema.model_dump(mode="json") == schema
    assert_shapes(restored, kind, 1)
    assert REQUESTS[kind].vocabulary(restored, ADDRESS) == ()
    more = ["d", "e", "f", "a"]
    resumed_data = rf.ArrowDataModule(model=restored, train=table(kind, more, supervise=True), **data_options)
    trace = ResumeTrace()
    resumed = Trainer(max_epochs=2, callbacks=[trace], **trainer_options)
    resumed.fit(restored, datamodule=resumed_data, ckpt_path=path)

    assert trace.restored and trace.batches == [0, 1]
    assert resumed.global_step == 4
    assert resumed.current_epoch == 2
    assert trace.scheduler.last_epoch == 4
    assert [group["lr"] for group in trace.optimizer.param_groups] == pytest.approx([1e-3 * 0.5**4] * len(rates))
    assert_shapes(restored, kind, 8)
    assert REQUESTS[kind].vocabulary(restored, ADDRESS) == tuple("abcdef")
    assert_counts(restored, kind, Counter(labels + more))
    assert restored.schema.model_dump(mode="json") == schema
