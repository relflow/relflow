"""Duration derives adjacent elapsed times while preserving Branch alignment."""

import pyarrow as pa
import pytest

import relflow as rf
from relflow.structs.enums import Tokens

P, N, V = Tokens.padded, Tokens.null, Tokens.valued


def history(*times):
    item = pa.struct([("occurred_at", pa.string())])
    return pa.table({"transactions": pa.array([[{"occurred_at": time} for time in times]], pa.list_(item))})


SOURCE = history("2026-01-01T00:00:00", "2026-01-01T00:00:05", "2026-01-01T00:00:12")


def test_delta_keeps_one_value_per_record_and_a_leading_null():
    class Transaction(rf.Branch):
        elapsed = rf.Duration(query="occurred_at", delta=True, granularity="seconds")

    class FraudDetector(rf.Model):
        transactions = Transaction(length=64, overflow="tail")

    field = FraudDetector(preset="small", d_model=8).encode(SOURCE)["/transactions/elapsed"]
    assert field.content.reshape(-1).tolist() == [0, 5, 7]
    assert field.state.reshape(-1).tolist() == [N, V, V]


def test_delta_uses_the_field_name_when_query_is_omitted():
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(length=3, occurred_at=rf.Duration(delta=True, granularity="seconds")),
    )
    field = model.encode(SOURCE)["/transactions/occurred_at"]
    assert field.content.reshape(-1).tolist() == [0, 5, 7]
    assert field.state.reshape(-1).tolist() == [N, V, V]


def test_delta_accepts_a_normalized_single_query_and_current_branch_absolute_path():
    for query in (("occurred_at",), "/transactions/occurred_at"):
        model = rf.Model(
            preset="small",
            d_model=8,
            transactions=rf.Branch(elapsed=rf.Duration(query=query, delta=True, granularity="seconds")),
        )
        field = model.encode(SOURCE)["/transactions/elapsed"]
        assert field.content.reshape(-1).tolist() == [0, 5, 7]
        assert field.state.reshape(-1).tolist() == [N, V, V]


def test_first_null_is_not_a_numeric_target():
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(
            elapsed=rf.Duration(query="occurred_at", delta=True, granularity="seconds", predict=True)
        ),
    )
    field = model.encode(SOURCE, strata="train")["/transactions/elapsed"]
    assert field.targets[rf.TensorKey.state].reshape(-1).tolist() == [N, V, V]
    assert rf.Duration.normalization(model, "/transactions/elapsed")["count"] == 2


def test_delta_keeps_the_original_predecessor_before_a_tail_crop():
    source = history(*(f"2026-01-01T00:00:0{second}" for second in range(7)))
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(
            length=2, overflow="tail", elapsed=rf.Duration(query="occurred_at", delta=True, granularity="seconds")
        ),
    )
    field = model.encode(source)["/transactions/elapsed"]
    assert field.state.reshape(-1).tolist() == [V, V]
    assert field.content.reshape(-1).tolist() == [1, 1]


def test_delta_preserves_null_records_and_missing_timestamps():
    source = pa.table(
        {
            "transactions": pa.array(
                [
                    [
                        {"occurred_at": "2026-01-01T00:00:00"},
                        None,
                        {"occurred_at": "2026-01-01T00:00:02"},
                        {"occurred_at": None},
                        {"occurred_at": "2026-01-01T00:00:04"},
                        {"occurred_at": "2026-01-01T00:00:06"},
                    ]
                ],
                type=pa.list_(pa.struct([("occurred_at", pa.string())])),
            )
        }
    )
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(length=6, elapsed=rf.Duration(query="occurred_at", delta=True, granularity="seconds")),
    )
    field = model.encode(source)["/transactions/elapsed"]
    assert field.state.reshape(-1).tolist() == [N, P, N, N, N, V]
    assert field.content.reshape(-1).tolist() == [0, 0, 0, 0, 0, 2]


def test_delta_keeps_negative_durations_when_timestamps_are_out_of_order():
    source = history("2026-01-01T00:00:05", "2026-01-01T00:00:02")
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(length=2, elapsed=rf.Duration(query="occurred_at", delta=True, granularity="seconds")),
    )
    field = model.encode(source)["/transactions/elapsed"]
    assert field.state.reshape(-1).tolist() == [N, V]
    assert field.content.reshape(-1).tolist() == [0, -3]


def test_delta_does_not_gather_unrelated_record_fields():
    timestamps = SOURCE["transactions"].chunk(0).values.field("occurred_at")
    unrelated = pa.RunEndEncodedArray.from_arrays(pa.array([2, 3], type=pa.int16()), pa.array(["a", "b"]))
    records = pa.StructArray.from_arrays([timestamps, unrelated], names=["occurred_at", "unrelated"])
    source = pa.table({"transactions": pa.ListArray.from_arrays(pa.array([0, 3], type=pa.int32()), records)})
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(length=3, elapsed=rf.Duration(query="occurred_at", delta=True, granularity="seconds")),
    )
    field = model.encode(source)["/transactions/elapsed"]
    assert field.state.reshape(-1).tolist() == [N, V, V]
    assert field.content.reshape(-1).tolist() == [0, 5, 7]


def test_delta_requires_a_branch_and_local_timestamp_query():
    with pytest.raises(ValueError, match="requires a Branch"):
        rf.Model(preset="small", d_model=8, elapsed=rf.Duration(query="at", delta=True))
    with pytest.raises(ValueError, match="own Branch"):
        rf.Model(preset="small", d_model=8, events=rf.Branch(elapsed=rf.Duration(query="../at", delta=True)))


def test_delta_rejects_an_array_inside_an_individual_branch_record():
    model = rf.Model(preset="small", d_model=8, events=rf.Branch(elapsed=rf.Duration(query="at", delta=True)))
    source = pa.table({"events": [[{"at": ["2026-01-01", "2026-01-02"]}]]})
    with pytest.raises(ValueError, match="one timestamp per Branch record"):
        model.encode(source)


def test_delta_respects_nullable_false():
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(length=3, elapsed=rf.Duration(query="occurred_at", delta=True, nullable=False)),
    )
    with pytest.raises(ValueError, match="nullable=False"):
        model.encode(SOURCE)


@pytest.mark.parametrize(
    "options",
    [
        {"query": "at", "delta": 1},
        {"query": "at", "delta": "true"},
        {"query": (), "delta": True},
        {"query": ("start", "end"), "delta": True},
        {"query": "at", "start": "2026-01-01", "delta": True},
        {"query": "at", "end": "2026-01-01", "delta": True},
        {"query": "at", "shift": 1},
    ],
)
def test_delta_rejects_invalid_options(options):
    with pytest.raises(ValueError):
        rf.Duration(**options)


@pytest.mark.integration
def test_delta_trains_and_predicts_with_worker_batches(tmp_path):
    records = pa.Table.from_pylist(
        [
            {
                "transactions": [
                    {"occurred_at": f"2026-01-01T00:00:{second:02d}", "amount": float(row + second)}
                    for second in (0, 5, 12)
                ]
            }
            for row in range(4)
        ]
    )
    model = rf.Model(
        preset="small",
        d_model=8,
        transactions=rf.Branch(
            length=2,
            overflow="tail",
            amount=rf.Number,
            elapsed=rf.Duration(query="occurred_at", delta=True, granularity="seconds", predict=True),
        ),
    )
    data = rf.ArrowDataModule(
        train=records, validate=records, num_workers=1, persistent_workers=False, pin_memory=False
    )
    model.fit(
        data,
        batch_size=2,
        max_epochs=1,
        limit_train_batches=1,
        limit_val_batches=1,
        logger=False,
        enable_checkpointing=False,
        rollback=False,
        early_stopping=False,
        enable_progress_bar=False,
        default_root_dir=str(tmp_path),
    )
    assert rf.Duration.normalization(model, "/transactions/elapsed")["count"] == 4
    output = model.predict(records, batch_size=1).to_dicts()
    assert len(output) == 4
    assert all(len(row["predictions"]["transactions"]) == 2 for row in output)
    assert all(item["elapsed"]["inferred"] for row in output for item in row["predictions"]["transactions"])


def test_yaml_and_checkpoint_round_trips_preserve_delta(tmp_path):
    path = tmp_path / "duration.yaml"
    path.write_text("""
preset: small
d_model: 8
fields:
  transactions:
    type: branch
    fields:
      elapsed: {type: duration, query: occurred_at, delta: true, granularity: seconds}
""")
    model = rf.Model.from_yaml(path)
    model.encode(SOURCE, strata="train")
    restored = rf.Model.load(model.save(tmp_path / "duration.ckpt"))
    for loaded in (model, restored):
        field = loaded.encode(SOURCE)["/transactions/elapsed"]
        assert field.content.reshape(-1).tolist() == [0, 5, 7]
        assert field.state.reshape(-1).tolist() == [N, V, V]
    assert rf.Duration.normalization(restored, "/transactions/elapsed") == rf.Duration.normalization(
        model, "/transactions/elapsed"
    )
