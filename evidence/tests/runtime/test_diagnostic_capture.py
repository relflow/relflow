"""Return snapshots preserve proof results, RNG state, and paired observations."""

from pathlib import Path
from types import ModuleType

import numpy as np
import pyarrow.ipc as ipc
import pytest

from evidence.diagnostics.capture import Capture, load


def test_return_capture_preserves_values_and_snapshots_mutable_inputs(tmp_path):
    source = tmp_path / "proof.py"
    text = """import numpy as np
def score(rows, predicted):
    actual = np.asarray([row['target'] for row in rows])
    return float(np.mean((predicted-actual)**2))
def run():
    rows = [{'target': 1.0}, {'target': 2.0}]
    predicted = np.array([1.1, 1.8])
    measured = score(rows, predicted)
    rows[0]['target'] = 9.0
    return measured, rows
"""
    source.write_text(text)
    module = ModuleType("captured_test_proof")
    exec(compile(text, str(source), "exec"), module.__dict__)
    expected = module.run()
    output = tmp_path / "observations"
    output.mkdir()
    capture = Capture(source, output)
    rng_before = np.random.get_state()
    capture.start(module)
    try:
        actual = module.run()
    finally:
        capture.close()
    rng_after = np.random.get_state()
    assert actual == expected
    np.testing.assert_array_equal(rng_before[1], rng_after[1])
    assert rng_before[2:] == rng_after[2:]
    assert [frame["function"] for frame in capture.frames] == ["score", "run"]
    snapshots = [frame["values"]["rows"]["artifact"] for frame in capture.frames]
    assert snapshots[0] != snapshots[1]
    with ipc.open_file(output / snapshots[0]) as reader:
        assert reader.read_all().to_pylist() == [{"target": 1.0}, {"target": 2.0}]
    with ipc.open_file(output / snapshots[1]) as reader:
        assert reader.read_all().to_pylist() == [{"target": 9.0}, {"target": 2.0}]
    # Duplicate snapshots share content-addressed data without losing call identity.
    assert capture.frames[1]["values"]["predicted"] == capture.frames[0]["values"]["predicted"]


def test_snapshot_keeps_nonfinite_values_explicit(tmp_path):
    capture = Capture(Path(__file__), tmp_path)
    assert capture.value(float("nan")) == {"nonfinite": "nan"}
    assert capture.value(None) is None
    assert capture.value(np.array([float("nan"), float("inf")]))["shape"] == [2]


def test_parameter_sequence_snapshot_retains_gradients(tmp_path):
    import torch

    parameters = [torch.tensor([float(index)], requires_grad=True) for index in range(40)]
    capture = Capture(Path(__file__), tmp_path)
    snapshot = capture.value(parameters)
    assert len(snapshot) == 40
    assert all(parameter.requires_grad for parameter in parameters)
    sum(parameter.sum() for parameter in parameters).backward()
    assert all(parameter.grad.item() == 1 for parameter in parameters)
    with ipc.open_file(tmp_path / snapshot[7]["artifact"]) as reader:
        assert reader.read_all()["value"].to_pylist() == [7.0]


@pytest.mark.integration
def test_proof_module_remains_importable_by_spawned_seed_workers(tmp_path):
    from concurrent.futures import ProcessPoolExecutor
    from multiprocessing import get_context

    source = tmp_path / "diagnostic_spawn_fixture.py"
    source.write_text("def trial(seed):\n    return seed * seed\n")
    proof = load(source)
    with ProcessPoolExecutor(max_workers=1, mp_context=get_context("spawn")) as pool:
        assert list(pool.map(proof.trial, [17, 117, 217])) == [289, 13689, 47089]
