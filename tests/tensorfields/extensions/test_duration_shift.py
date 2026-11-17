"""Duration measures gaps between records of one Branch collection from a single timestamp source."""

import pyarrow as pa
import pytest

import relflow as rf
from relflow.structs.enums import Tokens

P, V = Tokens.padded, Tokens.valued


def history(*times):
    item = pa.struct([("occurred_at", pa.string())])
    return pa.table({"transactions": pa.array([[{"occurred_at": time} for time in times]], pa.list_(item))})


SOURCE = history("2026-01-01T00:00:00", "2026-01-01T00:00:05", "2026-01-01T00:00:12")


@pytest.mark.parametrize(("shift", "content", "state"), [(1, [0, 5, 7], [P, V, V]), (2, [0, 0, 12], [P, P, V])])
def test_shift_measures_elapsed_time_since_an_earlier_record(shift, content, state):
    class Transaction(rf.Branch):
        elapsed = rf.Duration(query="occurred_at", shift=shift, granularity="seconds")

    class FraudDetector(rf.Model):
        transactions = Transaction(length=64, overflow="tail")

    field = FraudDetector(preset="small", d_model=8).encode(SOURCE)["/transactions/elapsed"]

    assert field.content.reshape(-1).tolist() == content
    assert field.state.reshape(-1).tolist() == state


def test_records_without_a_predecessor_are_never_targets():
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(elapsed=rf.Duration(query="occurred_at", shift=1, granularity="seconds", mask=True)),
    )

    field = model.encode(SOURCE, strata="train")["/transactions/elapsed"]

    assert field.targets[rf.TensorKey.state].reshape(-1).tolist() == [P, V, V]
    assert rf.Duration.normalization(model, "/transactions/elapsed")["count"] == 2


@pytest.mark.parametrize(
    "options",
    [
        {"query": "at", "shift": 0},
        {"query": "at", "shift": True},
        {"query": "at", "shift": 1.5},
        {"query": ("start", "end"), "shift": 1},
        {"query": "at", "start": "2026-01-01", "shift": 1},
        {"query": "at", "end": "2026-01-01", "shift": 1},
    ],
)
def test_shift_rejects_invalid_offsets_and_endpoint_combinations(options):
    with pytest.raises(ValueError, match="shift"):
        rf.Duration(**options)


def test_yaml_and_checkpoint_round_trips_preserve_shift(tmp_path):
    path = tmp_path / "duration.yaml"
    path.write_text(
        """
preset: small
d_model: 8
fields:
  transactions:
    type: branch
    fields:
      elapsed: {type: duration, query: occurred_at, shift: 2, granularity: seconds}
"""
    )
    model = rf.Model.from_yaml(path)
    model.encode(SOURCE, strata="train")
    checkpoint = tmp_path / "duration.ckpt"
    model.save(checkpoint)
    restored = rf.Model.load(checkpoint)

    for loaded in (model, restored):
        field = loaded.encode(SOURCE)["/transactions/elapsed"]
        assert field.content.reshape(-1).tolist() == [0, 0, 12]
    assert rf.Duration.normalization(restored, "/transactions/elapsed") == rf.Duration.normalization(
        model, "/transactions/elapsed"
    )
