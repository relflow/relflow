import copy
import uuid
from types import SimpleNamespace
from typing import Literal

import pyarrow as pa
import pytest
import torch
from lightning.pytorch import Callback
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.trainer.states import TrainerFn
from tensordict import TensorDict

import relflow as rf
from relflow.data.arrow import Encoded
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.tensorfields import grouping
from relflow.tensorfields.extensions import number
from relflow.tensorfields.extensions.group import resolve_group_embedder, vocabulary_logits


@pytest.fixture
def case(monkeypatch):
    model = rf.Model.xs(d_model=8, batch_size=256, identity=rf.Group(bounds=2, mask=True, p_unavailable=0))
    model.encode(pa.table({"identity": ["a", "b", "absent"]}), strata="train")
    embedder = resolve_group_embedder(model, "/identity")
    weight = embedder.embeddings[TensorKey.group.name].weight
    with torch.no_grad():
        weight[:3].copy_(torch.tensor([[2.0, -2], [-2, 2], [0, 1]]))
        weight[-1].copy_(torch.tensor([0.5, -0.5]))

    def decode(pooled):
        return TensorDict(
            {
                TensorKey.state: pooled.new_zeros((pooled.shape[0], len(Tokens))),
                TensorKey.group: pooled.new_tensor([0.0, 5.0]).expand(pooled.shape[0], -1),
            }
        )

    monkeypatch.setattr(model.nodes["/identity"].decoder, "decode", decode)
    train = pa.table({"identity": ["a", "a", "a"]})
    validate = pa.table({"identity": ["a", "a"]})
    return model, embedder, weight, train, validate


def optimizer_history(weight):
    optimizer = torch.optim.AdamW([weight], amsgrad=True)
    optimizer.state[weight] = {
        "step": torch.tensor(13.0),
        "exp_avg": torch.full_like(weight, 3),
        "exp_avg_sq": torch.full_like(weight, 5),
        "max_exp_avg_sq": torch.full_like(weight, 7),
    }
    weight.grad = torch.full_like(weight, 11)
    return optimizer


def test_only_group_can_refine_its_own_hidden_identity_objective(case, monkeypatch):
    model, embedder, weight, train, validate = case
    encoded = []
    encode = model.encode

    def observed_encode(table, *, strata):
        assert strata == Strata.validate
        result = encode(table, strata=strata)
        assert result["/identity"].state.ne(Tokens.valued).all()
        assert result["/identity"].targets[TensorKey.state].eq(Tokens.valued).all()
        encoded.append(table)
        return result

    def forbidden(*args, **kwargs):
        pytest.fail("Group refinement must not access the test loop or ordinary loss hooks")

    monkeypatch.setattr(model, "encode", observed_encode)
    monkeypatch.setattr(model, "test", forbidden)
    monkeypatch.setitem(rf.TENSORFIELDS["group"].components, rf.Component.loss, forbidden)
    vocabulary = embedder.vocab.snapshot()

    event = grouping.refine(model, "/identity", train, validate)

    assert event["accepted"] and event["changed_ids"] == 1
    assert event["objective"] == "group_identity_nll"
    assert event["training_targets"] == 3 and event["validation_targets"] == 2
    assert event["proposed"] < event["before"]
    assert weight[0].argmax().item() == 1
    assert [table.num_rows for table in encoded] == [3, 2]
    assert embedder.vocab.snapshot() == vocabulary


def test_accepted_own_nll_matches_shared_likelihood_and_preserves_unobserved_optimizer_rows(case):
    model, embedder, weight, train, validate = case
    model.train()
    model.nodes["/identity"].decoder.eval()
    modes = {layer: layer.training for layer in model.modules()}
    parameters = {name: value.detach().clone() for name, value in model.named_parameters() if value is not weight}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    rng = torch.get_rng_state().clone()
    before = weight.detach().clone()
    optimizer = optimizer_history(weight)

    event = grouping.refine(model, "/identity", train, validate, [optimizer])

    assert event["accepted"]
    probabilities = torch.tensor([[0.0, 5.0]]).softmax(-1)
    expected = -vocabulary_logits(embedder, probabilities)[0, 0].item()
    assert event["proposed"] == pytest.approx(expected)
    torch.testing.assert_close(weight[0].logsumexp(-1), before[0].logsumexp(-1))
    torch.testing.assert_close(weight[1:], before[1:], rtol=0, atol=0)
    assert optimizer.state[weight]["step"].item() == 13
    for name, value in optimizer.state[weight].items():
        if name != "step":
            assert not value[0].any()
            assert value[1:].eq({"exp_avg": 3, "exp_avg_sq": 5, "max_exp_avg_sq": 7}[name]).all()
    assert weight.grad is not None
    assert not weight.grad[0].any() and weight.grad[1:].eq(11).all()
    for name, value in model.named_parameters():
        if value is not weight:
            torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    assert all(layer.training == training for layer, training in modes.items())
    assert torch.equal(torch.get_rng_state(), rng)


def test_training_membership_proposal_is_rejected_when_own_validation_nll_worsens(case, monkeypatch):
    model, _, weight, train, validate = case

    def decode(pooled):
        # Different fixed context predictions isolate train proposal from validation acceptance.
        logits = [0.0, 5.0] if pooled.shape[0] == train.num_rows else [5.0, 0.0]
        return TensorDict(
            {
                TensorKey.state: pooled.new_zeros((pooled.shape[0], len(Tokens))),
                TensorKey.group: pooled.new_tensor(logits).expand(pooled.shape[0], -1),
            }
        )

    monkeypatch.setattr(model.nodes["/identity"].decoder, "decode", decode)
    before = weight.detach().clone()
    optimizer = optimizer_history(weight)
    history = copy.deepcopy(optimizer.state[weight])

    event = grouping.refine(model, "/identity", train, validate, [optimizer])

    assert not event["accepted"] and event["changed_ids"] == 0
    assert event["proposed"] > event["before"]
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
    for name, value in history.items():
        torch.testing.assert_close(optimizer.state[weight][name], value)
    assert weight.grad is not None and weight.grad.eq(11).all()


@pytest.mark.parametrize("failure", ["exception", "nonfinite"])
def test_failed_own_head_restores_parameters_buffers_modes_and_rng(case, monkeypatch, failure):
    model, _, _, train, validate = case
    decoder = model.nodes["/identity"].decoder
    original = decoder.decode
    calls = 0

    def decode(pooled):
        nonlocal calls
        calls += 1
        torch.rand(3)
        if calls == 3:
            if failure == "exception":
                raise RuntimeError("Group counterfactual failed")
            result = original(pooled)
            result[TensorKey.group].fill_(float("nan"))
            return result
        return original(pooled)

    monkeypatch.setattr(decoder, "decode", decode)
    model.train()
    decoder.eval()
    frozen = decoder.linears[TensorKey.state.name].weight
    frozen.requires_grad_(False)
    parameters = {name: value.detach().clone() for name, value in model.named_parameters()}
    buffers = {name: value.detach().clone() for name, value in model.named_buffers()}
    modes = {layer: layer.training for layer in model.modules()}
    rng = torch.get_rng_state().clone()

    event = grouping.refine(model, "/identity", train, validate)

    assert not event["accepted"] and "reason" in event
    for name, value in model.named_parameters():
        torch.testing.assert_close(value, parameters[name], rtol=0, atol=0)
    for name, value in model.named_buffers():
        torch.testing.assert_close(value, buffers[name], rtol=0, atol=0)
    assert all(layer.training == training for layer, training in modes.items())
    assert not frozen.requires_grad
    assert torch.equal(torch.get_rng_state(), rng)


def test_single_identity_likelihood_ties_keep_assignment_and_optimizer_history():
    model = rf.Model.xs(d_model=8, batch_size=8, identity=rf.Group(bounds=3, mask=True, p_unavailable=0))
    table = pa.table({"identity": ["only", "only"]})
    model.encode(table, strata="train")
    embedder = resolve_group_embedder(model, "/identity")
    weight = embedder.embeddings[TensorKey.group.name].weight
    with torch.no_grad():
        weight[0].copy_(torch.tensor([1.0, -1.0, 0.0]))
    before = weight.detach().clone()
    optimizer = optimizer_history(weight)

    event = grouping.refine(model, "/identity", table, table, [optimizer])

    assert not event["accepted"]
    torch.testing.assert_close(weight, before, rtol=0, atol=0)
    assert optimizer.state[weight]["exp_avg"].eq(3).all()
    assert weight.grad is not None and weight.grad.eq(11).all()


@pytest.mark.parametrize("blocked", ["frozen", "override", "empty_validation"])
def test_ineligible_refinement_keeps_assignments_and_skips_head(case, monkeypatch, blocked):
    model, embedder, weight, train, validate = case
    if blocked == "frozen":
        weight.requires_grad_(False)
    elif blocked == "override":
        embedder._override_depth = 1
    else:
        validate = validate.slice(0, 0)

    def forbidden(*args, **kwargs):
        pytest.fail("ineligible Group refinement must not evaluate its head")

    monkeypatch.setattr(grouping, "evidence", forbidden)
    before = weight.detach().clone()
    event = grouping.refine(model, "/identity", train, validate)
    assert not event["accepted"] and "reason" in event
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_plain_visible_group_does_not_borrow_another_fields_objective(monkeypatch):
    model = rf.Model.xs(d_model=8, identity=rf.Group(bounds=2), target=rf.Number(mask=True))
    table = pa.table({"identity": ["a", "b"], "target": [1.0, 2.0]})
    model.encode(table, strata="train")

    def forbidden(*args, **kwargs):
        pytest.fail("Group without its own objective must not borrow supervised field losses")

    monkeypatch.setitem(rf.TENSORFIELDS["number"].components, rf.Component.loss, forbidden)
    event = grouping.refine(model, "/identity", table, table)
    assert not event["accepted"] and "Group objective" in event["reason"]


def test_unknown_and_null_group_targets_supply_no_evidence_or_vocabulary_admission(case):
    model, embedder, weight, train, _ = case
    validate = pa.table({"identity": ["unknown", None]})
    vocabulary = embedder.vocab.snapshot()
    before = weight.detach().clone()

    event = grouping.refine(model, "/identity", train, validate)

    assert not event["accepted"]
    assert event["reason"] == "no finite known Group validation targets"
    assert embedder.vocab.snapshot() == vocabulary
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_nested_group_evidence_includes_multiple_identities_and_repeated_coordinates():
    model = rf.Model.xs(
        d_model=8,
        batch_size=8,
        items=rf.Branch(identity=rf.Group(bounds=2, mask=True, p_unavailable=0)),
    )
    train = pa.Table.from_pylist([{"items": [{"identity": "a"}, {"identity": "b"}, {"identity": "a"}]}])
    model.encode(train, strata="train")
    validation = pa.Table.from_pylist(
        [
            {"items": [{"identity": "a"}, {"identity": "b"}, {"identity": "a"}]},
            {"items": [{"identity": "unknown"}, {"identity": None}]},
            {"items": []},
        ],
        schema=train.schema,
    )

    identities, context, errors = grouping.evidence(model, "/items/identity", grouping.calibrate(model, validation))

    assert identities.tolist() == [0, 1, 0]
    assert context.shape == (3, 2) and errors.shape == (3,)
    assert torch.isfinite(errors).all()
    empty = pa.Table.from_pylist([{"items": []}], schema=train.schema)
    identities, context, errors = grouping.evidence(model, "/items/identity", grouping.calibrate(model, empty))
    assert identities.numel() == errors.numel() == 0 and context.shape == (0, 2)


def test_unrelated_third_party_decoder_and_loss_are_not_called():
    name = f"unrelated_group_{uuid.uuid4().hex}"
    extension = rf.Extension(name=name, types=(int | float,))

    @extension.register(rf.Component.Request)
    class Request(number.Request):
        type: Literal[name] = name

    @extension.register(rf.Component.Decoder)
    class Decoder(number.Decoder):
        def decode(self, pooled):
            pytest.fail("Group refinement must not decode an unrelated supervised target")

    @extension.register(rf.Component.loss)
    def loss(*args, **kwargs):
        pytest.fail("Group refinement must not invoke third-party supervision")

    for component, implementation in number.number.components.items():
        if component not in {rf.Component.Request, rf.Component.Decoder, rf.Component.loss}:
            extension.register(component, implementation)
    try:
        model = rf.Model.xs(
            d_model=8,
            batch_size=8,
            identity=rf.Group(bounds=2, mask=True, p_unavailable=0),
            unrelated=Request(mask=True),
        )
        table = pa.table({"identity": ["a", "b"], "unrelated": [1.0, 2.0]})
        model.encode(table, strata="train")
        event = grouping.refine(model, "/identity", table, table)
        assert event["objective"] == "group_identity_nll"
    finally:
        rf.TENSORFIELDS.pop(name)


def test_ordinary_supervised_backpropagation_still_trains_visible_group_assignments(monkeypatch):
    model = rf.Model.xs(d_model=8, identity=rf.Group(bounds=2, p_unavailable=0), target=rf.Number(mask=True))
    source = pa.table({"identity": ["a", "b", "a"], "target": [1.0, -1.0, 2.0]})
    inputs = model.encode(source, strata="train")
    monkeypatch.setattr(model, "track", lambda *args, value, **kwargs: value)
    prediction = next(value for value in model(inputs, strata=Strata.train) if value.address == "/target")

    number.loss(model, prediction, inputs["/target"], Strata.train).backward()

    gradient = resolve_group_embedder(model, "/identity").embeddings[TensorKey.group.name].weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert bool(gradient[:2].ne(0).any())


@pytest.mark.parametrize("objective,validation", [(False, True), (True, False), (False, False)])
def test_callback_without_own_objective_or_validation_retains_no_calibration_or_collectives(
    monkeypatch, objective, validation
):
    model = rf.Model.xs(
        d_model=8,
        identity=rf.Group(bounds=2, mask=rf.Mask(reconstruct=True) if objective else ()),
        target=rf.Number(mask=True),
    )
    table = pa.table({"identity": ["a", "b"], "target": [1.0, 2.0]})
    callback = grouping.Refinement()
    saved = {"last_step": 50, "events": [{"accepted": False, "step": 50}]}
    callback.load_state_dict(copy.deepcopy(saved))
    callback.train.add(table)
    callback.validate.add(table)
    trainer = SimpleNamespace(enable_validation=validation)
    warnings = []

    def forbidden(*args, **kwargs):
        pytest.fail("inactive Group refinement must not retain data, run collectives, or inspect cadence")

    monkeypatch.setattr(grouping, "world_size", forbidden)
    monkeypatch.setattr(grouping, "synchronize_epoch_metrics", forbidden)
    monkeypatch.setattr(grouping, "refine", forbidden)
    monkeypatch.setattr(grouping.Calibration, "add", forbidden)
    monkeypatch.setattr(grouping.Calibration, "gather", forbidden)
    monkeypatch.setattr(
        grouping, "logger", SimpleNamespace(bind=lambda **kwargs: SimpleNamespace(warning=warnings.append))
    )

    callback.on_fit_start(trainer, model)
    callback.on_train_start(trainer, model)
    batch = Encoded(TensorDict({}), table)
    callback.on_train_batch_end(trainer, model, None, batch, 0)
    callback.on_validation_batch_end(trainer, model, None, batch, 0)
    callback.on_validation_epoch_start(trainer, model)

    assert callback.train.rows == callback.validate.rows == 0
    assert callback.train.size == callback.validate.size == 0
    assert callback.train.tables == callback.validate.tables == []
    assert callback.state_dict() == saved
    assert bool(warnings) == (objective and not validation)


def test_callback_refines_own_hidden_group_at_cadence_and_resumes_state(case, monkeypatch):
    model, _, _, train, validate = case
    callback = grouping.Refinement()
    trainer = SimpleNamespace(
        enable_validation=True,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_should_accumulate=lambda: False)),
        sanity_checking=False,
        global_step=0,
        optimizers=[],
    )
    callback.on_fit_start(trainer, model)
    callback.on_train_batch_end(trainer, model, None, Encoded(TensorDict({}), train), 0)
    callback.on_validation_batch_end(trainer, model, None, Encoded(TensorDict({}), validate), 0)
    phases = []

    def synchronize(trainer):
        phases.append("flush_previous_metrics")

    def refine(module, address, training, validation, optimizers):
        assert address == "/identity" and training.equals(train) and validation.equals(validate)
        phases.append("refine")
        return {"accepted": True, "changed_ids": 1}

    monkeypatch.setattr(grouping, "synchronize_epoch_metrics", synchronize)
    monkeypatch.setattr(grouping, "refine", refine)
    callback.on_validation_epoch_start(trainer, model)
    trainer.global_step = 199
    callback.on_validation_epoch_start(trainer, model)
    assert not phases
    trainer.global_step = 200
    callback.on_validation_epoch_start(trainer, model)
    assert phases == ["flush_previous_metrics", "refine"]
    assert callback.events[-1]["step"] == 200
    resumed = grouping.Refinement()
    resumed.load_state_dict(copy.deepcopy(callback.state_dict()))
    resumed.on_fit_start(trainer, model)
    resumed.on_train_batch_end(trainer, model, None, Encoded(TensorDict({}), train), 0)
    resumed.on_validation_batch_end(trainer, model, None, Encoded(TensorDict({}), validate), 0)
    resumed.on_validation_epoch_start(trainer, model)
    assert len(phases) == 2
    trainer.global_step = 400
    resumed.on_validation_epoch_start(trainer, model)
    assert len(phases) == 4 and resumed.last_step == 400
    trainer.state.fn = TrainerFn.VALIDATING
    trainer.global_step = 600
    resumed.on_validation_epoch_start(trainer, model)
    assert len(phases) == 4
    trainer.state.fn = TrainerFn.FITTING
    trainer.sanity_checking = True
    resumed.on_validation_epoch_start(trainer, model)
    assert len(phases) == 4


def test_callback_waits_for_accumulated_optimizer_step_without_advancing_cadence(case, monkeypatch):
    model, _, weight, train, validate = case
    callback = grouping.Refinement()
    callback.last_step = 200
    trainer = SimpleNamespace(
        enable_validation=True,
        state=SimpleNamespace(fn=TrainerFn.FITTING),
        fit_loop=SimpleNamespace(epoch_loop=SimpleNamespace(_should_accumulate=lambda: True)),
        sanity_checking=False,
        global_step=400,
        optimizers=[],
    )
    callback.on_fit_start(trainer, model)
    callback.on_train_batch_end(trainer, model, None, Encoded(TensorDict({}), train), 0)
    callback.on_validation_batch_end(trainer, model, None, Encoded(TensorDict({}), validate), 0)

    def forbidden(*args, **kwargs):
        pytest.fail("pending accumulation must not flush metrics or replace Group assignments")

    monkeypatch.setattr(grouping, "synchronize_epoch_metrics", forbidden)
    monkeypatch.setattr(grouping, "refine", forbidden)
    before = weight.detach().clone()

    callback.on_validation_epoch_start(trainer, model)

    assert callback.last_step == 200 and callback.events == []
    torch.testing.assert_close(weight, before, rtol=0, atol=0)


def test_fit_checkpoint_metrics_and_callback_state_describe_refined_mapping(case, monkeypatch, tmp_path):
    model, _, weight, train, validate = case
    decoder = model.nodes["/identity"].decoder
    monkeypatch.setattr(decoder, "decode", type(decoder).decode.__get__(decoder))
    observed = []

    def refine(module, address, training, validation, optimizers):
        with torch.no_grad():
            weight[0, 1] = 10 + module.global_step
        return {"accepted": True, "changed_ids": 1}

    class Inspection(Callback):
        def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
            if not trainer.sanity_checking:
                observed.append((trainer.global_step, weight[0, 1].item()))

        def on_validation_epoch_end(self, trainer, pl_module):
            pl_module.log("mapping_freshness", weight[0, 1], on_step=False, on_epoch=True)

    monkeypatch.setattr(grouping, "INTERVAL", 1)
    monkeypatch.setattr(grouping, "refine", refine)
    checkpoint = ModelCheckpoint(dirpath=tmp_path, monitor="mapping_freshness", mode="max", save_top_k=1)
    datamodule = rf.ArrowDataModule(
        model=model,
        train=train,
        validate=validate,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        datamodule,
        batch_size=2,
        learning_rate=1e-6,
        fused=False,
        max_epochs=1,
        max_steps=2,
        limit_train_batches=2,
        limit_val_batches=1,
        num_sanity_val_steps=1,
        val_check_interval=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        early_stopping=False,
        rollback=False,
        callbacks=[Inspection(), checkpoint],
        enable_progress_bar=False,
        enable_model_summary=False,
        logger=False,
    )

    assert observed == [(1, 11), (2, 12)]
    assert checkpoint.best_model_score.item() == 12
    saved = torch.load(checkpoint.best_model_path, weights_only=True)
    refinement = saved["callbacks"]["Refinement"]
    assert refinement["last_step"] == 2
    assert [event["step"] for event in refinement["events"]] == [1, 2]
    torch.testing.assert_close(
        saved["state_dict"]["nodes./identity.embedder.embeddings.group.weight"][0, 1], torch.tensor(12.0)
    )


def test_calibration_bounds_rows_and_bytes_without_retaining_full_source_buffers():
    source = pa.table({"text": ["x" * 100 for _ in range(20)]})
    calibration = grouping.Calibration(limit=8, byte_limit=source.slice(0, 4).nbytes)
    calibration.add(source)
    calibration.add(source)
    assert 0 < calibration.rows <= 4 and calibration.size <= calibration.byte_limit
    retained = calibration.gather(8, calibration.byte_limit)
    assert retained is not None and retained.equals(source.slice(0, calibration.rows))
    assert retained["text"].chunk(0).buffers()[-1].address != source["text"].chunk(0).buffers()[-1].address
