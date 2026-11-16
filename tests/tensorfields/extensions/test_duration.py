from datetime import datetime, timezone

import pyarrow as pa
import pytest
import torch
import yaml

import relflow as rf
from relflow.tensorfields.extensions.duration import duration
from relflow.tensorfields.extensions.number import number


def test_duration_uses_numbers_modeling_components_without_a_second_learner():
    assert rf.TENSORFIELDS["duration"].Request is rf.Duration
    for role in (
        rf.Component.TensorField,
        rf.Component.Embedder,
        rf.Component.Decoder,
        rf.Component.observe,
        rf.Component.learn,
        rf.Component.loss,
        rf.Component.output,
        rf.Component.write,
    ):
        assert duration.component(role) is number.component(role)
    assert duration.callback_factories == number.callback_factories


@pytest.mark.parametrize("mask", [False, True])
def test_duration_observes_computed_values_in_source_units_before_input_masking(mask):
    model = rf.Model(preset="small", d_model=8, elapsed=rf.Duration(query=("start", "end"), mask=mask))
    records = pa.table({"start": ["2026-01-01"] * 4, "end": ["2026-01-02T12:00:00", "2026-01-01", "2025-12-31", None]})

    field = model.encode(records, strata="train")["/elapsed"]
    content = field.targets[rf.TensorKey.content] if mask else field.content

    torch.testing.assert_close(content.flatten(), torch.tensor([1.5, 0, -1, 0], dtype=torch.float64))
    stats = rf.Duration.normalization(model, "/elapsed")
    assert stats["count"] == 3
    assert stats["mean"] == pytest.approx(0.5 / 3)
    if not mask:
        assert field.state.flatten().tolist() == [rf.Tokens.valued] * 3 + [rf.Tokens.null]


@pytest.mark.parametrize(("endpoint", "expected"), [("start", 2.5), ("end", -2.5)])
def test_duration_pairs_one_query_with_either_configured_endpoint(endpoint, expected):
    model = rf.Model(preset="small", d_model=8, elapsed=rf.Duration(query="observed_at", **{endpoint: "2026-01-01"}))

    field = model.encode(pa.table({"observed_at": ["2026-01-03T12:00:00"]}))["/elapsed"]

    assert field.content.item() == expected


def test_duration_implicit_query_reads_the_same_named_source_with_a_static_endpoint():
    model = rf.Model(preset="small", d_model=8, elapsed=rf.Duration(start="2026-01-01"))

    field = model.encode(pa.table({"elapsed": ["2026-01-02T12:00:00"]}))["/elapsed"]

    assert field.content.item() == 1.5


def test_two_static_endpoints_preserve_the_batch_size_without_source_columns():
    model = rf.Model(preset="small", d_model=8, elapsed=rf.Duration(start="2026-01-01", end="2026-01-04"))

    field = model.encode(pa.table({"unrelated": [1, 2, 3]}))["/elapsed"]

    assert field.content.shape == (3, 1)
    assert field.content.flatten().tolist() == [3.0] * 3
    assert model.schema.requests[rf.Address("elapsed")].query == ()


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"query": "at"},
        {"query": ("start", "end"), "start": "2026-01-01"},
        {"query": ("a", "b", "c")},
        {"query": (), "start": "2026-01-01"},
        {"query": "at", "start": "2026-01-01", "end": "2026-01-02"},
    ],
)
def test_duration_rejects_ambiguous_endpoint_orientation(options):
    with pytest.raises(ValueError, match="Duration"):
        rf.Duration(**options)


@pytest.mark.parametrize("unit", ["months", "years", "day", "unknown"])
def test_duration_rejects_unsupported_granularity(unit):
    with pytest.raises(ValueError, match="granularity"):
        rf.Duration(query=("start", "end"), granularity=unit)


@pytest.mark.parametrize("value", [5, 0.5, True, "", "   ", []])
def test_duration_static_endpoints_are_explicit_temporal_values(value):
    with pytest.raises(ValueError, match="endpoints"):
        rf.Duration(query="at", start=value)


def test_duration_yaml_preserves_tuple_queries_units_and_configured_timestamps(tmp_path):
    model = rf.Model(
        preset="small",
        d_model=8,
        elapsed=rf.Duration(query=("start", "end"), granularity="hours"),
        since=rf.Duration(query="end", start=datetime(2026, 1, 1, tzinfo=timezone.utc)),
    )
    path = tmp_path / "duration.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "preset": "small",
                "d_model": 8,
                "fields": {
                    name: model.schema.requests[rf.Address(name)].model_dump(mode="json", exclude={"name"})
                    for name in ("elapsed", "since")
                },
            }
        )
    )
    restored = rf.Model.from_yaml(path)
    records = pa.table({"start": ["2026-01-01T00:00:00Z"], "end": ["2026-01-02T12:00:00Z"]})

    original = model.encode(records)
    reloaded = restored.encode(records)

    assert restored.schema.requests[rf.Address("elapsed")].query == ("start", "end")
    assert reloaded["/elapsed"].content.item() == 36
    for address in ("/elapsed", "/since"):
        torch.testing.assert_close(original[address].content, reloaded[address].content)


def test_duration_broadcasts_account_dates_after_transaction_cropping_without_losing_holes():
    records = pa.Table.from_pylist(
        [
            {
                "account": {"created_at": "2026-01-01"},
                "transactions": [{"posted_at": "2026-01-02"}, None, {"posted_at": "2026-01-05"}],
            },
            {"account": {"created_at": "2026-01-02"}, "transactions": []},
            {"account": {"created_at": "2026-01-02"}, "transactions": None},
            {"account": {"created_at": None}, "transactions": [{"posted_at": "2026-01-03"}]},
        ]
    )
    model = rf.Model(
        preset="small",
        d_model=8,
        embed=True,
        transactions=rf.Branch(
            length=2, overflow="tail", age=rf.Duration(query=("../account.created_at", "posted_at"), embed=True)
        ),
    )

    field = model.encode(records)["/transactions/age"]

    assert field.content.reshape(4, 2).tolist() == [[0, 4], [0, 0], [0, 0], [0, 0]]
    assert field.state.reshape(4, 2).tolist() == [
        [rf.Tokens.padded, rf.Tokens.valued],
        [rf.Tokens.padded] * 2,
        [rf.Tokens.padded] * 2,
        [rf.Tokens.null, rf.Tokens.padded],
    ]
    assert len(model.predict(records)) == len(records)


def test_duration_checkpoint_retains_source_configuration_and_numeric_statistics(tmp_path):
    model = rf.Model(preset="small", d_model=8, elapsed=rf.Duration(start="2026-01-01", granularity="hours"))
    records = pa.table({"elapsed": ["2026-01-02", "2026-01-04"]})
    model.encode(records, strata="train")
    checkpoint = tmp_path / "duration.ckpt"
    model.save(checkpoint)

    restored = rf.Model.load(checkpoint)

    assert restored.schema.model_dump(mode="json") == model.schema.model_dump(mode="json")
    assert rf.Duration.normalization(restored, "/elapsed") == rf.Duration.normalization(model, "/elapsed")
    torch.testing.assert_close(restored.encode(records)["/elapsed"].content, model.encode(records)["/elapsed"].content)


@pytest.mark.integration
def test_duration_can_train_and_predict_with_number_objectives(tmp_path):
    records = pa.table(
        {
            "input": [1.0, 2.0, 3.0, 4.0],
            "start": ["2026-01-01"] * 4,
            "end": ["2026-01-02", "2026-01-03", "2026-01-04", "2026-01-05"],
        }
    )
    model = rf.Model(preset="small", d_model=8, input=rf.Number, elapsed=rf.Duration(query=("start", "end"), mask=True))
    data = rf.ArrowDataModule(
        train=records, validate=records, num_workers=0, persistent_workers=False, pin_memory=False
    )
    model.fit(
        data,
        batch_size=4,
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

    output = model.predict(records.drop(["end"]))
    poisoned = records.set_column(records.schema.get_field_index("end"), "end", pa.array(["2050-01-01"] * 4))

    assert len(output) == 4
    assert "elapsed" in output["predictions"].struct.fields
    assert output.equals(model.predict(poisoned))
