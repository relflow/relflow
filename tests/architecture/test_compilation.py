import pickle
from copy import deepcopy
from types import MethodType

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.mutations import MutationLockCallback
from relflow.architecture.pool import LearnedQueryCrossAttention


@pytest.fixture
def model():
    return rf.Model(
        value=rf.Number,
        history=rf.Branch(length=2, attention=None, value=rf.Number),
        label=rf.Boolean(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=0.0,
        batch_size=2,
    )


@pytest.fixture
def compiler(monkeypatch):
    calls = []

    def compile(function, **options):
        record = {"function": function, "options": options, "executions": 0}
        calls.append(record)

        def compiled(*args, **kwargs):
            record["executions"] += 1
            return function(*args, **kwargs)

        return compiled

    monkeypatch.setattr(torch, "compile", compile)
    return calls


def regions(model):
    return [module for module in model.modules() if isinstance(module, (BranchEncoder, LearnedQueryCrossAttention))]


@pytest.mark.parametrize("encoders,pools", [(True, True), (True, False), (False, True), (False, False)])
def test_compile_selects_tensor_regions_without_changing_model_state(model, compiler, encoders, pools):
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    modules = {name: id(module) for name, module in model.named_modules()}
    state = deepcopy(model.state_dict())
    schema = model.schema.model_dump(mode="python")
    eager = {module: module.compute for module in regions(model)}

    assert model.compile(encoders=encoders, pools=pools) is model

    selected = {record["function"].__self__ for record in compiler}
    expected = {
        module
        for module in eager
        if (encoders and isinstance(module, BranchEncoder) and len(module.encoder))
        or (pools and isinstance(module, LearnedQueryCrossAttention))
    }
    assert selected == expected
    assert model.nodes["record/history"].encoder not in selected
    if pools:
        assert model.nodes["record/label"].decoder.pool in selected
    for module in eager.keys() - selected:
        assert module.compute == eager[module]
    assert parameters == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert modules == {name: id(module) for name, module in model.named_modules()}
    assert model.schema.model_dump(mode="python") == schema
    torch.testing.assert_close(model.state_dict(), state, rtol=0, atol=0)


def test_inductor_defaults_and_explicit_options_are_forwarded_without_mutating_input(model, compiler):
    model.compile()
    defaults = {
        "fallback_random": True,
        "triton.cudagraphs": False,
        "shape_padding": False,
        "comprehensive_padding": False,
        "inplace_padding": False,
    }
    assert compiler
    assert all(
        record["options"] == {"backend": "inductor", "fullgraph": True, "dynamic": None, "options": defaults}
        for record in compiler
    )

    compiler.clear()
    options = {"max_autotune": True, "shape_padding": True}
    model.compile(dynamic=True, options=options)
    assert options == {"max_autotune": True, "shape_padding": True}
    assert all(record["options"]["fullgraph"] is True for record in compiler)
    assert all(record["options"]["dynamic"] is True for record in compiler)
    assert all(record["options"]["options"] == defaults | options for record in compiler)


def test_recompiling_replaces_the_policy_and_both_disabled_restores_eager(model, compiler):
    eager = {module: module.compute for module in regions(model)}
    model.compile(pools=False)
    encoder_bindings = {module: module.compute for module in eager if module.compute != eager[module]}
    assert encoder_bindings

    compiler.clear()
    model.compile(encoders=False, backend="eager", dynamic=False)
    assert compiler
    assert all(isinstance(record["function"].__self__, LearnedQueryCrossAttention) for record in compiler)
    assert all(module.compute == eager[module] for module in encoder_bindings)
    assert all(record["function"].__self__ in eager for record in compiler)
    assert all(record["options"]["options"] is None for record in compiler)

    compiler.clear()
    model.compile(encoders=False, pools=False)
    assert not compiler
    assert all(module.compute == function for module, function in eager.items())


def test_failed_compiler_construction_keeps_every_previous_binding(model, compiler, monkeypatch):
    model.compile()
    bindings = {module: module.compute for module in regions(model)}
    attempted = []

    def fail(function, **options):
        attempted.append(function)
        if len(attempted) == 2:
            raise RuntimeError("compiler setup failed")
        return function

    monkeypatch.setattr(torch, "compile", fail)
    with pytest.raises(RuntimeError, match="compiler setup failed"):
        model.compile(backend="eager")
    assert len(attempted) == 2
    assert all(module.compute == function for module, function in bindings.items())


@pytest.mark.parametrize("switches", [{"encoders": None}, {"pools": 1}])
def test_invalid_region_switches_leave_compilation_unchanged(model, compiler, switches):
    model.compile()
    bindings = {module: module.compute for module in regions(model)}
    compiler.clear()
    with pytest.raises(TypeError, match="Boolean region switches"):
        model.compile(**switches)
    assert not compiler
    assert all(module.compute == function for module, function in bindings.items())


def test_compilation_cannot_change_during_a_lightning_training_loop(model, compiler):
    model.compile()
    bindings = {module: module.compute for module in regions(model)}
    compiler.clear()
    callback = MutationLockCallback()
    callback.on_train_start(None, model)
    try:
        with pytest.raises(RuntimeError, match="compile"):
            model.compile(encoders=False, pools=False)
    finally:
        callback.on_train_end(None, model)
    assert not compiler
    assert all(module.compute == function for module, function in bindings.items())
    model.compile(encoders=False, pools=False)
    assert all(module.compute.__self__ is module for module in regions(model))


def test_compile_preserves_custom_compute_and_custom_owner_classes(model, compiler, monkeypatch):
    encoder = model.nodes["record"].encoder

    def compute(self, inputs, present):
        return inputs * 3

    custom = MethodType(compute, encoder)
    monkeypatch.setattr(encoder, "compute", custom)

    class CustomPool(LearnedQueryCrossAttention):
        pass

    model.add_module("custom_pool", CustomPool(1, 8, 2, 0.0))
    pool_compute = model.custom_pool.compute
    model.compile()
    selected = {record["function"].__self__ for record in compiler}
    assert encoder not in selected and model.custom_pool not in selected
    assert encoder.compute == custom
    assert model.custom_pool.compute == pool_compute
    model.compile(encoders=False, pools=False)
    assert encoder.compute == custom
    assert model.custom_pool.compute == pool_compute


def test_compute_borrowed_from_another_owner_remains_bound_to_that_owner(model, compiler, monkeypatch):
    encoder = model.nodes["record"].encoder
    donor = BranchEncoder(model.schema, "record")
    monkeypatch.setattr(encoder, "compute", donor.compute)
    inputs = torch.randn(2, 3, 8)
    present = torch.ones(2, 3, dtype=torch.bool)
    expected = donor.compute(inputs, present)

    model.compile()
    selected = {record["function"].__self__ for record in compiler}
    assert encoder not in selected and donor not in selected
    assert encoder.compute.__self__ is donor
    torch.testing.assert_close(encoder.compute(inputs, present), expected, rtol=0, atol=0)
    model.compile(encoders=False, pools=False)
    assert encoder.compute.__self__ is donor


@pytest.mark.parametrize("operation", ["contracts", "update", "reset", "restore"])
def test_lifecycle_changes_return_regions_to_eager(model, compiler, operation):
    checkpoint = {"state_dict": deepcopy(model.state_dict())}
    model.on_save_checkpoint(checkpoint)
    model.compile()
    assert compiler
    if operation == "contracts":
        model.reset_contracts()
    elif operation == "update":
        model.update(rf.where("name") == "value", embed=True)
    elif operation == "reset":
        model.reset(rf.where("name") == "value")
    else:
        model.restore_checkpoint_state(checkpoint)
    assert all(module.compute.__self__ is module for module in regions(model))


def test_native_lightning_weight_resume_keeps_explicit_compilation(model, compiler):
    checkpoint = {"state_dict": deepcopy(model.state_dict())}
    model.on_save_checkpoint(checkpoint)
    model.compile()
    bindings = {module: module.compute for module in regions(model)}
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    with torch.no_grad():
        next(model.parameters()).add_(1)

    model.on_load_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    assert all(module.compute == function for module, function in bindings.items())
    assert parameters == {name: id(parameter) for name, parameter in model.named_parameters()}
    torch.testing.assert_close(model.state_dict(), checkpoint["state_dict"], rtol=0, atol=0)


def test_checkpoint_saves_weights_and_schema_without_compilation(model, compiler, tmp_path):
    eager = tmp_path / "eager.ckpt"
    model.save(eager)
    model.compile()
    compiled = tmp_path / "compiled.ckpt"
    model.save(compiled)
    before = torch.load(eager, weights_only=False)
    after = torch.load(compiled, weights_only=False)
    torch.testing.assert_close(after.pop("state_dict"), before.pop("state_dict"), rtol=0, atol=0)
    assert after == before
    restored = rf.Model.load(compiled)
    assert all(module.compute.__self__ is module for module in regions(restored))
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("operation", ["deepcopy", "pickle"])
def test_copy_and_pickle_drop_compilation_and_bind_the_new_owners(model, compiler, operation):
    model.compile()
    bindings = {module: module.compute for module in regions(model)}
    restored = deepcopy(model) if operation == "deepcopy" else pickle.loads(pickle.dumps(model))
    assert all(module.compute.__self__ is module for module in regions(restored))
    assert not set(regions(model)) & set(regions(restored))
    assert all(module.compute == function for module, function in bindings.items())
    torch.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("dropout", [0.0, 0.2])
def test_real_graph_backend_reaches_training_step_and_preserves_accumulation(monkeypatch, dropout):
    eager = rf.Model(
        value=rf.Number,
        label=rf.Boolean(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
        dropout=dropout,
        batch_size=2,
    )
    batch = eager.encode(pa.table({"value": [1.0, 5.0], "label": [False, True]}), strata="train")
    compiled = deepcopy(eager)
    graphs = []
    executions = []

    def backend(graph, inputs):
        graphs.append(graph)

        def execute(*args):
            executions.append(graph)
            return graph.forward(*args)

        return execute

    torch._dynamo.reset()
    try:
        compiled.compile(backend=backend)
        losses, gradients, random_states = [], [], []
        for model in (eager, compiled):
            monkeypatch.setattr(model, "log", lambda *args, **kwargs: None)
            torch.manual_seed(1729)
            retained = []
            for _ in range(2):
                result = model.training_step(deepcopy(batch), 0)
                retained.append(result["loss"].detach().clone())
                result["loss"].backward()
            losses.append(retained)
            gradients.append({name: parameter.grad for name, parameter in model.named_parameters()})
            random_states.append(torch.get_rng_state())
        assert graphs and len(executions) >= 2 * len(graphs)
        torch.testing.assert_close(losses[1], losses[0], rtol=0, atol=0)
        torch.testing.assert_close(gradients[1], gradients[0], rtol=0, atol=0)
        assert any(value is not None and value.abs().sum() > 0 for value in gradients[1].values())
        assert torch.equal(random_states[1], random_states[0])
    finally:
        torch._dynamo.reset()


@pytest.mark.parametrize("boundary", ["encoder", "pool"])
def test_descendant_hooks_added_after_warmup_run_eager_then_compilation_resumes(model, boundary):
    graphs, executions, hooks = [], [], []

    def backend(graph, inputs):
        graphs.append(graph)

        def execute(*args):
            executions.append(graph)
            return graph.forward(*args)

        return execute

    model.eval()
    if boundary == "encoder":
        owner = model.nodes["record"].encoder
        descendant = owner.encoder[0].ffn_norm
    else:
        owner = model.nodes["record/label"].decoder.pool
        descendant = owner.norm
    inputs = torch.randn(2, 3, 8)
    present = torch.ones(2, 3, dtype=torch.bool)
    arguments = (inputs, present) if boundary == "encoder" else (inputs, present, None, None)

    def hook(module, inputs, output):
        hooks.append(module)
        return output + 1

    handle = None
    torch._dynamo.reset()
    try:
        model.compile(backend=backend)
        with torch.no_grad():
            first = owner.compute(*arguments)
            assert len(graphs) == len(executions) == 1
            handle = descendant.register_forward_hook(hook)
            changed = owner.compute(*arguments)
            assert hooks == [descendant]
            assert len(graphs) == len(executions) == 1
            assert not torch.equal(first, changed)
            handle.remove()
            handle = None
            restored = owner.compute(*arguments)
            assert len(graphs) == 1 and len(executions) == 2
            torch.testing.assert_close(restored, first, rtol=0, atol=0)
    finally:
        if handle is not None:
            handle.remove()
        torch._dynamo.reset()
