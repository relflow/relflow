from __future__ import annotations

import subprocess
import sys
from copy import deepcopy
from textwrap import dedent
from types import SimpleNamespace

import lightning.pytorch as lit
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.structs.enums import Strata
from relflow.tensorfields.base import TENSORFIELDS


@pytest.fixture
def descent():
    pytest.importorskip("torchjd")
    pytest.importorskip("quadprog")
    from relflow.architecture import jacobian

    return jacobian


def model(*, jacobian=False, second=True):
    fields = {"context": rf.Number, "first": rf.Boolean(mask=True, weight=3.0)}
    if second:
        fields["second"] = rf.Boolean(mask=True, weight=0.5)
    return rf.Model(
        **fields,
        jacobian=jacobian,
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        batch_size=2,
    )


def gradients(losses, parameters):
    """Reference Jacobian assembled from independent scalar backward traversals."""

    rows = []
    for loss in losses:
        values = torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)
        rows.append(
            torch.cat(
                [
                    (torch.zeros_like(parameter) if value is None else value).reshape(-1)
                    for parameter, value in zip(parameters, values, strict=True)
                ]
            )
        )
    return torch.stack(rows)


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("entry", ["tree", "schema", "preset", "yaml"])
def test_construction_keeps_jacobian_as_runtime_policy(tmp_path, entry, policy):
    fields = {"context": rf.Number, "first": rf.Boolean(mask=True)}
    if entry == "tree":
        candidate = rf.Model(d_model=8, n_layers=1, n_heads=2, jacobian=policy, **fields)
    elif entry == "schema":
        schema = rf.Schema.from_tree(d_model=8, n_layers=1, n_heads=2, **fields)
        candidate = rf.Model(schema, jacobian=policy)
    elif entry == "preset":
        candidate = rf.Model.xs(d_model=8, jacobian=policy, **fields)
    else:
        path = tmp_path / "model.yaml"
        path.write_text(
            dedent(
                """
                preset: xs
                d_model: 8
                fields:
                  context:
                    type: number
                  first:
                    type: boolean
                    mask: true
                """
            )
        )
        candidate = rf.Model.from_yaml(path)
        assert candidate.jacobian is False
        candidate.jacobian = policy

    assert candidate.jacobian is policy
    assert tuple(candidate.schema.requests) == ("/context", "/first")
    assert "jacobian" not in candidate.schema.model_dump()


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("preset", ["xs", "sm", "md", "lg", "xl"])
def test_preset_factories_forward_jacobian_without_persisting_it(preset, policy):
    candidate = getattr(rf.Model, preset)(
        d_model=8, n_layers=1, n_heads=2, jacobian=policy, first=rf.Boolean(mask=True)
    )

    assert candidate.jacobian is policy
    assert "jacobian" not in candidate.preset.model_dump()
    assert "jacobian" not in candidate.schema.model_dump()
    assert tuple(candidate.schema.requests) == ("/first",)


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("entry", ["yaml", "override"])
def test_templates_reject_jacobian_training_flags(tmp_path, entry, policy):
    path = tmp_path / "model.yaml"
    content = "preset: xs\nd_model: 8\nfields: {first: {type: boolean, mask: true}}\n"
    if entry == "yaml":
        content += f"jacobian: {str(policy).lower()}\n"
    path.write_text(content)

    error = ValueError if entry == "yaml" else TypeError
    with pytest.raises(error, match="jacobian"):
        rf.Model.from_yaml(path, **({"jacobian": policy} if entry == "override" else {}))


@pytest.mark.integration
def test_optional_dependency_is_only_required_for_enabled_training(tmp_path):
    script = dedent(
        """
        import importlib.abc
        import sys

        class BlockTorchJD(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "torchjd" or fullname.startswith("torchjd."):
                    raise ModuleNotFoundError("blocked optional TorchJD", name=fullname)

        sys.meta_path.insert(0, BlockTorchJD())
        import pyarrow as pa
        import relflow as rf

        table = pa.table({"context": [1.0, 2.0], "first": [True, False], "second": [False, True]})
        model = rf.Model(
            context=rf.Number, first=rf.Boolean(mask=True), second=rf.Boolean(mask=True),
            d_model=8, n_layers=1, n_heads=2, dropout=0.0,
        )
        assert model.jacobian is False
        model.on_fit_start()
        model.training_step(model.encode(table, strata="train"), 0)["loss"].backward()
        assert model.predict(table).num_rows == 2

        model.jacobian = True
        assert model.predict(table).num_rows == 2
        assert model.validation_step(model.encode(table, strata="validate"), 0)["loss"].isfinite()
        assert model.test_step(model.encode(table, strata="test"), 0)["loss"].isfinite()
        model.save(sys.argv[1])
        loaded = rf.Model.load(sys.argv[1])
        assert loaded.jacobian is False
        loaded.on_fit_start()
        assert loaded.predict(table).num_rows == 2
        assert not any(name == "torchjd" or name.startswith("torchjd.") for name in sys.modules)
        assert "relflow.architecture.jacobian" not in sys.modules
        loaded.jacobian = True
        try:
            loaded.on_fit_start()
        except ImportError as error:
            assert "relflow[torchjd]" in str(error), error
        else:
            raise AssertionError("enabled training did not report the missing optional dependency")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(tmp_path / "model.ckpt")],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_one_objective_preserves_exact_gradients_without_building_a_jacobian(descent, monkeypatch):
    eager = model(second=False)
    enabled = deepcopy(eager)
    enabled.jacobian = True

    def forbidden(*args, **kwargs):
        raise AssertionError("one objective must use ordinary scalar backward")

    monkeypatch.setattr(torch.autograd, "grad", forbidden)
    monkeypatch.setattr(descent, "UPGradWeighting", forbidden)
    table = pa.table({"context": [1.0, 2.0], "first": [True, False]})
    losses = []
    for candidate in (eager, enabled):
        monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
        result = candidate.training_step(candidate.encode(table, strata=Strata.train), 0)
        assert set(result) == {"loss"}
        losses.append(result["loss"].detach())
        result["loss"].backward()

    torch.testing.assert_close(losses[0], losses[1], rtol=0, atol=0)
    for before, after in zip(eager.parameters(), enabled.parameters(), strict=True):
        if before.grad is None:
            assert after.grad is None
        else:
            torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_conflicting_objectives_match_full_jacobian_upgrad_with_unit_preferences(descent, dtype):
    from torchjd.aggregation import UPGrad

    module = torch.nn.Module()
    module.register_parameter("values", torch.nn.Parameter(torch.tensor([0.2, -0.3, 0.4], dtype=dtype)))
    module.optimizer = None
    values = module.values
    losses = [values[0] + values[2].square(), -values[0] + values[1], -values[1] + values[2]]
    parameters = tuple(module.parameters())
    jacobian = gradients(losses, parameters)
    expected = UPGrad(pref_vector=torch.ones(len(losses), dtype=dtype))(jacobian)
    assert not torch.allclose(expected, jacobian.sum(dim=0))

    combined = descent.combine(module, losses, values.sum() * 0.0)
    assert all(parameter.grad is None for parameter in parameters)
    assert all(not hasattr(parameter, "jac") for parameter in parameters)
    combined.backward()

    torch.testing.assert_close(values.grad, expected, rtol=2e-5, atol=2e-6)


@pytest.mark.parametrize("constant", [0.0, 7.0])
def test_constant_objective_contributes_a_zero_gradient_row(descent, constant):
    module = torch.nn.Module()
    module.register_parameter("value", torch.nn.Parameter(torch.tensor(2.0)))
    module.optimizer = None
    losses = [module.value.square(), module.value.new_tensor(constant)]

    descent.combine(module, losses, module.value * 0.0).backward()

    torch.testing.assert_close(module.value.grad, torch.tensor(4.0))


def test_weighted_schema_objectives_match_independent_full_jacobian(descent, monkeypatch):
    from torchjd.aggregation import UPGrad

    candidate = model(jacobian=True)
    monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
    inputs = candidate.encode(
        pa.table({"context": [1.0, 2.0], "first": [True, False], "second": [False, True]}),
        strata=Strata.train,
    )
    losses = []
    for prediction in candidate(inputs, strata=Strata.train):
        request = candidate.schema.requests[prediction.address]
        loss = TENSORFIELDS[request.type].loss(
            module=candidate,
            prediction=prediction,
            batch=inputs[prediction.address],
            strata=Strata.train,
        )
        losses.append(loss * request.weight)
    parameters = tuple(parameter for parameter in candidate.parameters() if parameter.requires_grad)
    jacobian = gradients(losses, parameters)
    expected = UPGrad(pref_vector=torch.ones(2))(jacobian)

    result = candidate.training_step(inputs, 0)
    assert set(result) == {"loss"}
    result["loss"].backward()
    actual = torch.cat(
        [
            (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).reshape(-1)
            for parameter in parameters
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)


def test_sparse_nested_objectives_preserve_geometry_and_log_the_raw_loss_sum(descent, monkeypatch):
    from torchjd.aggregation import UPGrad

    candidate = rf.Model.xs(
        d_model=8,
        dropout=0.0,
        context=rf.Number,
        history=rf.Branch(
            length=8,
            value=rf.Number,
            target=rf.Boolean(mask=rf.Mask(query="selected", skip=True, dropout=False, reconstruct=True), weight=3.0),
        ),
        answer=rf.Boolean(mask=True, weight=0.5),
    )
    candidate.jacobian = True
    source = pa.Table.from_pylist(
        [
            {
                "context": 1.0,
                "history": [
                    {"value": 2.0, "target": True, "selected": True},
                    {"value": 4.0, "target": False, "selected": False},
                    {"value": 6.0, "target": False, "selected": True},
                ],
                "answer": False,
            },
            {"context": 2.0, "history": [], "answer": True},
        ]
    )
    monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
    inputs = candidate.encode(source, strata=Strata.train)
    losses = []
    for prediction in candidate(inputs, strata=Strata.train):
        request = candidate.schema.requests[prediction.address]
        losses.append(
            TENSORFIELDS[request.type].loss(
                module=candidate, prediction=prediction, batch=inputs[prediction.address], strata=Strata.train
            )
            * request.weight
        )
    parameters = tuple(parameter for parameter in candidate.parameters() if parameter.requires_grad)
    expected = UPGrad(pref_vector=torch.ones(2))(gradients(losses, parameters))
    recorded = {}

    def track(names, value, **kwargs):
        recorded[tuple(names)] = value.detach().clone()
        return value

    monkeypatch.setattr(candidate, "track", track)
    result = candidate.training_step(inputs, 0)
    result["loss"].backward()
    actual = torch.cat(
        [
            (torch.zeros_like(parameter) if parameter.grad is None else parameter.grad).reshape(-1)
            for parameter in parameters
        ]
    )
    torch.testing.assert_close(actual, expected, rtol=2e-4, atol=2e-6)
    total = torch.stack(losses).sum().detach()
    torch.testing.assert_close(recorded[("loss", Strata.train)], total)
    torch.testing.assert_close(recorded[("loss", "train_epoch")], total)


@pytest.mark.integration
def test_training_grows_vocabulary_rows_without_stale_optimizer_geometry(descent, tmp_path):
    candidate = rf.Model.xs(
        d_model=8,
        dropout=0.0,
        batch_size=2,
        context=rf.Category(p_unavailable=0.0),
        first=rf.Category(mask=True, p_unavailable=0.0),
        second=rf.Boolean(mask=True),
    )
    candidate.jacobian = True
    table = pa.table(
        {
            "context": [f"context-{i}" for i in range(12)],
            "first": [f"label-{i}" for i in range(12)],
            "second": [False, True] * 6,
        }
    )
    data = rf.ArrowDataModule(model=candidate, train=table, shuffle=False, num_workers=0)
    sizes, identities = [], []

    class Audit(lit.Callback):
        def on_after_backward(self, trainer, pl_module):
            decoder = pl_module.nodes["/first"].decoder.linears["content"]
            sizes.append(decoder.out_features)
            identities.append(id(decoder.weight))
            owned = {id(parameter) for group in trainer.optimizers[0].param_groups for parameter in group["params"]}
            assert owned == {id(parameter) for parameter in pl_module.parameters() if parameter.requires_grad}
            assert decoder.weight.grad is not None
            assert all(
                torch.isfinite(parameter.grad).all()
                for parameter in pl_module.parameters()
                if parameter.grad is not None
            )

    candidate.fit(
        data,
        callbacks=[Audit()],
        max_epochs=1,
        accelerator="cpu",
        devices=1,
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=False,
        default_root_dir=tmp_path,
        early_stopping=False,
        rollback=False,
        fused=False,
    )
    assert candidate.trainer.global_step == len(sizes) == 6
    assert max(sizes) >= 12 and sizes[-1] > sizes[0]
    assert len(set(identities)) > 1
    assert rf.Category.vocabulary(candidate, "/first") == tuple(f"label-{i}" for i in range(12))


def test_optimizer_subset_controls_geometry_and_frozen_parameters_are_excluded(descent):
    from torchjd.aggregation import UPGrad

    module = torch.nn.Module()
    module.register_parameter("owned", torch.nn.Parameter(torch.tensor([0.2, -0.3], dtype=torch.float64)))
    module.register_parameter("omitted", torch.nn.Parameter(torch.tensor(0.5, dtype=torch.float64)))
    module.register_parameter("frozen", torch.nn.Parameter(torch.tensor(1.0, dtype=torch.float64), requires_grad=False))
    module.optimizer = torch.optim.SGD([module.owned], lr=0.1)
    losses = [
        module.owned[0] + 10.0 * module.omitted + module.frozen,
        -module.owned[0] + module.owned[1] + 10.0 * module.omitted,
    ]
    expected = UPGrad(pref_vector=torch.ones(2, dtype=torch.float64))(gradients(losses, (module.owned,)))
    wrong = UPGrad(pref_vector=torch.ones(2, dtype=torch.float64))(gradients(losses, (module.owned, module.omitted)))[
        :2
    ]
    assert not torch.allclose(expected, wrong)
    before = module.owned.detach().clone()
    omitted = module.omitted.detach().clone()

    descent.combine(module, losses, module.owned.sum() * 0.0).backward()
    torch.testing.assert_close(module.owned.grad, expected, rtol=1e-7, atol=1e-9)
    assert module.frozen.grad is None
    module.optimizer.step()
    torch.testing.assert_close(module.owned, before - 0.1 * expected)
    torch.testing.assert_close(module.omitted, omitted, rtol=0, atol=0)


def test_precision_scaling_does_not_change_task_geometry(descent):
    jacobian = torch.tensor([[1.0, 2.0, -3.0], [-2.0, 1.0, 4.0]], dtype=torch.float64)
    scale = 1024.0
    parameters = [torch.zeros(2, dtype=jacobian.dtype), torch.zeros((), dtype=jacobian.dtype)]
    rows = [(row[:2] * scale, row[2] * scale) for row in jacobian]

    geometry = descent.gramian(rows, parameters, scale)

    assert geometry.dtype == torch.float64
    torch.testing.assert_close(geometry, jacobian @ jacobian.T, rtol=0, atol=0)


@pytest.mark.parametrize("fail", [False, True])
def test_compiler_backward_settings_are_scoped_to_jacobian_regions(monkeypatch, fail):
    from torch._functorch import config

    settings = []

    def compile(function, **options):
        def execute(*args, **kwargs):
            settings.append((config.donated_buffer, config.backward_pass_autocast))
            if fail:
                raise RuntimeError("compiler execution failed")
            return function(*args, **kwargs)

        return execute

    monkeypatch.setattr(torch, "compile", compile)
    original = (config.donated_buffer, config.backward_pass_autocast)
    with config.patch(donated_buffer=True, backward_pass_autocast="same_as_forward"):
        for enabled in (True, False):
            candidate = model(jacobian=enabled).compile(backend="eager")
            encoder = candidate.nodes["/"].encoder
            inputs = torch.randn(2, 3, 8)
            present = torch.ones(2, 3, dtype=torch.bool)
            if fail:
                with pytest.raises(RuntimeError, match="compiler execution failed"):
                    encoder.compute(inputs, present)
            else:
                encoder.compute(inputs, present)
            assert (config.donated_buffer, config.backward_pass_autocast) == (True, "same_as_forward")
    assert settings == [(False, "off"), (True, "same_as_forward")]
    assert (config.donated_buffer, config.backward_pass_autocast) == original


@pytest.mark.integration
@pytest.mark.parametrize("autocast", [False, True])
def test_aot_compilation_preserves_repeated_objective_backward(descent, monkeypatch, autocast):
    from torch._dynamo.backends.registry import lookup_backend
    from torch._functorch import config

    eager = model(jacobian=True)
    batch = eager.encode(
        pa.table({"context": [1.0, 2.0], "first": [True, False], "second": [False, True]}),
        strata=Strata.train,
    )
    # Encoding observes training data, including Number normalization state.
    compiled = deepcopy(eager)
    aot_eager = lookup_backend("aot_eager")
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        assert config.donated_buffer is False
        assert config.backward_pass_autocast == "off"
        return aot_eager(graph, inputs)

    original = (config.donated_buffer, config.backward_pass_autocast)
    torch._dynamo.reset()
    try:
        compiled.compile(backend=backend)
        losses, accumulated = [], []
        for candidate in (eager, compiled):
            monkeypatch.setattr(candidate, "log", lambda *args, **kwargs: None)
            candidate.on_fit_start()
            values = []
            for index in range(2):
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
                    result = candidate.training_step(deepcopy(batch), index)
                values.append(result["loss"].detach().clone())
                result["loss"].backward()
            losses.append(values)
            accumulated.append({name: parameter.grad for name, parameter in candidate.named_parameters()})
        assert graphs
        assert any(value is not None and value.abs().sum() > 0 for value in accumulated[1].values())
        torch.testing.assert_close(losses[1], losses[0], rtol=2e-4, atol=2e-6)
        torch.testing.assert_close(accumulated[1], accumulated[0], rtol=2e-4, atol=2e-6)
        assert (config.donated_buffer, config.backward_pass_autocast) == original
    finally:
        torch._dynamo.reset()


def test_enabling_jacobian_after_compilation_requires_recompiling(descent):
    candidate = model().compile(backend="eager")
    candidate.jacobian = True

    with pytest.raises(ValueError, match=r"jacobian=True.*model\.compile.*again"):
        candidate.on_fit_start()

    candidate.compile(backend="eager")
    candidate.on_fit_start()


def test_failed_objective_probe_restores_compiler_and_autocast_settings(descent, monkeypatch):
    from torch._functorch import config

    module = torch.nn.Module()
    module.register_parameter("value", torch.nn.Parameter(torch.tensor(2.0)))
    module.optimizer = None

    def fail(*args, **kwargs):
        assert config.donated_buffer is False
        assert not torch.is_autocast_enabled("cpu")
        raise RuntimeError("objective probe failed")

    monkeypatch.setattr(torch.autograd, "grad", fail)
    original = (config.donated_buffer, config.backward_pass_autocast)
    with config.patch(donated_buffer=True), torch.autocast("cpu", dtype=torch.bfloat16):
        with pytest.raises(RuntimeError, match="objective probe failed"):
            descent.combine(module, [module.value.square(), -module.value], module.value * 0.0)
        assert config.donated_buffer is True
        assert torch.is_autocast_enabled("cpu")
    assert (config.donated_buffer, config.backward_pass_autocast) == original
    assert module.value.grad is None


def test_nonfinite_objective_geometry_fails_before_backward(descent):
    module = torch.nn.Module()
    module.register_parameter("value", torch.nn.Parameter(torch.tensor(2.0)))
    module.optimizer = None

    with pytest.raises(FloatingPointError, match="gradients are not finite"):
        descent.combine(module, [module.value * float("inf"), -module.value], module.value * 0.0)

    assert module.value.grad is None


def test_solver_runtime_failure_is_reported_without_mutating_gradients(descent, monkeypatch):
    module = torch.nn.Module()
    module.register_parameter("value", torch.nn.Parameter(torch.tensor(2.0)))
    module.optimizer = None

    def fail(**kwargs):
        raise RuntimeError("solver failed unexpectedly")

    monkeypatch.setattr(descent, "UPGradWeighting", fail)
    with pytest.raises(RuntimeError, match="could not resolve finite objective weights") as caught:
        descent.combine(module, [module.value.square(), -module.value], module.value * 0.0)

    assert str(caught.value.__cause__) == "solver failed unexpectedly"
    assert module.value.grad is None


def test_unsharded_strategy_is_required_for_jacobian_training(descent):
    candidate = model(jacobian=True)
    candidate._trainer = SimpleNamespace(strategy=object())

    with pytest.raises(ValueError, match="single-device training and DDP"):
        candidate.on_fit_start()


@pytest.mark.parametrize("policy", [False, True])
def test_checkpoint_round_trip_omits_policy_and_preserves_predictions(tmp_path, policy):
    candidate = model(jacobian=policy)
    table = pa.table({"context": [1.0, 2.0], "first": [True, False], "second": [False, True]})
    expected = candidate.predict(table)
    path = tmp_path / "model.ckpt"

    candidate.save(path)
    saved = torch.load(path, weights_only=False)
    assert "jacobian" not in saved
    native = {}
    candidate.on_save_checkpoint(native)
    assert "jacobian" not in native
    restored = rf.Model.load(path)

    assert restored.jacobian is False
    assert restored.predict(table).equals(expected)
    assert restored.state_dict().keys() == candidate.state_dict().keys()


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize(
    "metadata",
    [{}, {"jacobian": False}, {"jacobian": True}, {"jacobian": "true"}, {"jacobian": None}],
    ids=["absent", "false", "true", "invalid-string", "invalid-none"],
)
def test_checkpoint_restore_preserves_explicit_session_policy(policy, metadata):
    candidate = model(jacobian=policy)
    checkpoint = {"state_dict": deepcopy(candidate.state_dict())}
    candidate.on_save_checkpoint(checkpoint)
    checkpoint.update(version="0.2.2", batch_size=7, **metadata)

    candidate.on_load_checkpoint(checkpoint)
    assert candidate.jacobian is policy
    assert candidate.version == "0.2.2"

    candidate.restore_checkpoint_state(checkpoint)
    assert candidate.jacobian is policy
    assert candidate.version == "0.2.2"
    assert candidate.batch_size == 7


@pytest.mark.parametrize("stale_policy", [True, "true"])
def test_fresh_load_ignores_stale_checkpoint_training_policy(tmp_path, stale_policy):
    candidate = model(jacobian=True)
    path = tmp_path / "stale.ckpt"
    candidate.save(path)
    saved = torch.load(path, weights_only=False)
    saved["jacobian"] = stale_policy
    torch.save(saved, path)

    assert rf.Model.load(path).jacobian is False


@pytest.mark.parametrize("policy", [False, True])
@pytest.mark.parametrize("invalid", ["version", "weights"])
def test_failed_checkpoint_restore_preserves_model_and_training_policy(invalid, policy):
    candidate = model(jacobian=policy).eval()
    state = deepcopy(candidate.state_dict())
    original = (
        candidate.schema,
        candidate.nodes,
        candidate.example_input_array,
        candidate.version,
        candidate.batch_size,
    )
    checkpoint = {"state_dict": deepcopy(state)}
    candidate.on_save_checkpoint(checkpoint)
    checkpoint.update(version="0.2.2", batch_size=7, jacobian="ignored")
    if invalid == "version":
        checkpoint["version"] = 1
        exception = ValueError
    else:
        del checkpoint["state_dict"][next(iter(checkpoint["state_dict"]))]
        exception = RuntimeError

    with pytest.raises(exception):
        candidate.restore_checkpoint_state(checkpoint)

    assert candidate.schema is original[0]
    assert candidate.nodes is original[1]
    assert candidate.example_input_array is original[2]
    assert (candidate.version, candidate.batch_size) == original[3:]
    assert candidate.jacobian is policy
    assert not candidate.training
    for name, expected in state.items():
        actual = candidate.state_dict()[name]
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        else:
            assert actual == expected


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_sparse_parameter_geometry_matches_dense_reference(descent, dtype):
    generator = torch.Generator().manual_seed(4721)
    parameters = [torch.randn(n, dtype=dtype, generator=generator) for n in (7, 3, 5, 2, 4)]
    support = [(0, 1, 2), (0,), (2,), (), (0, 2)]
    rows = [
        tuple(
            torch.randn(p.shape, dtype=dtype, generator=generator) if task in used else None
            for p, used in zip(parameters, support, strict=True)
        )
        for task in range(3)
    ]
    dense = torch.stack(
        [
            torch.cat(
                [
                    torch.zeros_like(p) if gradient is None else gradient
                    for p, gradient in zip(parameters, row, strict=True)
                ]
            )
            for row in rows
        ]
    )

    actual = descent.gramian(rows, parameters, 8.0)

    torch.testing.assert_close(actual, (dense / 8.0) @ (dense / 8.0).T)


def test_mixed_precision_geometry_retains_double_precision_from_later_parameters(descent):
    parameters = [torch.zeros(1, dtype=torch.float32), torch.zeros(1, dtype=torch.float64)]
    rows = [
        (torch.tensor([1.0]), torch.tensor([1e100], dtype=torch.float64)),
        (None, torch.tensor([-2e100], dtype=torch.float64)),
    ]

    actual = descent.gramian(rows, parameters, 1.0)

    assert actual.dtype == torch.float64
    assert torch.isfinite(actual).all()
    expected = torch.tensor([[1e200, -2e200], [-2e200, 4e200]], dtype=torch.float64)
    torch.testing.assert_close(actual, expected)


def test_sparse_extension_gradients_fail_with_an_actionable_error(descent):
    module = torch.nn.Module()
    module.lookup = torch.nn.Embedding(8, 2, sparse=True)
    module.optimizer = None
    values = module.lookup(torch.tensor([1, 3]))

    with pytest.raises(ValueError, match="requires dense objective gradients.*sparse=False"):
        descent.combine(module, [values.square().sum(), values.sum()], values.sum() * 0.0)

    assert module.lookup.weight.grad is None


def test_missing_local_objectives_skip_probes_but_preserve_backward_edges(descent, monkeypatch):
    from relflow.architecture import runtime

    candidate = model(jacobian=True)
    table = pa.table({"context": [1.0, 2.0], "first": [True, False], "second": [False, True]})
    inputs = candidate.encode(table, strata=Strata.train)
    first, second = candidate.schema.objectives
    monkeypatch.setattr(runtime, "participation", lambda *args: ({first}, {first, second}))
    monkeypatch.setattr(candidate, "track", lambda names, value, **kwargs: value)
    original = descent.combine
    captured = []

    def combine(module, losses, anchor):
        captured.extend(losses)
        assert anchor.requires_grad
        return original(module, losses, anchor)

    monkeypatch.setattr(descent, "combine", combine)
    result = candidate.training_step(inputs, 0)
    assert len(captured) == 2
    assert sum(loss.requires_grad for loss in captured) == 1
    assert captured[1].item() == 0.0
    result["loss"].backward()
    decoder = candidate.nodes[str(second)].decoder
    assert all(parameter.grad is not None for parameter in decoder.parameters() if parameter.requires_grad)
    assert all(
        torch.count_nonzero(parameter.grad) == 0 for parameter in decoder.parameters() if parameter.grad is not None
    )
