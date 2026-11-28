from __future__ import annotations

import polars as pl
import pyarrow as pa
import pytest

import relflow as rf
from relflow.data.ragged import coalesce


@pytest.mark.parametrize(
    ("options", "selection", "reconstruct", "skip", "sampled"),
    [
        ({"predict": True}, "all", True, True, False),
        ({"hide": "eligible"}, "query", False, False, False),
        ({"hide": 0.5}, "rate", False, False, True),
        ({"hide": ("eligible", 0.5)}, "union", False, False, True),
        ({"task": rf.reconstruct()}, "all", True, False, False),
        ({"task": rf.reconstruct(query="eligible")}, "query", True, False, False),
        ({"task": rf.reconstruct(rate=0.5)}, "rate", True, False, True),
        ({"task": rf.reconstruct(query="eligible", rate=0.5)}, "intersection", True, False, True),
        ({"task": rf.reconstruct(ablate=True)}, "all", True, True, False),
        ({"task": rf.reconstruct(query="eligible", ablate=True)}, "query", True, True, False),
        ({"task": rf.reconstruct(rate=0.5, ablate=True)}, "rate", True, True, True),
    ],
)
@pytest.mark.parametrize("strata", list(rf.Strata))
def test_mask_configuration_matrix(options, selection, reconstruct, skip, sampled, strata):
    size = 256
    eligible = [index % 2 == 0 for index in range(size)]
    source = pa.table({"value": [float(index) for index in range(size)], "eligible": eligible})
    schema = rf.Schema.from_tree(d_model=8, n_layers=1, n_heads=2, value=rf.Number(**options))
    projection = coalesce(source, schema, strata, seed=19, epoch=3).fields["/value"].single()
    inputs, targets = projection.split()

    active = strata == rf.Strata.train or (reconstruct and (strata != rf.Strata.predict or not sampled))
    chosen = (inputs.dense.reshape(-1) != rf.Tokens.valued).tolist()
    if not active:
        assert not any(chosen)
    elif selection == "all":
        assert all(chosen)
    elif selection == "query":
        assert chosen == eligible
    elif selection == "rate":
        assert 0 < sum(chosen) < size
    elif selection == "intersection":
        assert 0 < sum(chosen) < sum(eligible)
        assert not any(chosen[1::2])
    else:
        assert all(chosen[::2])
        assert 0 < sum(chosen[1::2]) < size // 2

    expected = [active and reconstruct and value for value in chosen]
    assert projection.trainable.tolist() == ([False] * size if strata == rf.Strata.predict else expected)
    assert projection.inferred.tolist() == (expected if strata == rf.Strata.predict else [False] * size)
    assert targets.placement.to_pylist() == [index for index, value in enumerate(projection.trainable) if value]
    assert projection.present.tolist() == ([not value for value in chosen] if skip else [True] * size)


def test_overlapping_reconstruction_rules_keep_one_answer_at_its_original_position():
    model = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        events=rf.Branch(
            value=rf.Number(task=(rf.reconstruct(query="hide_it"), rf.reconstruct(query="omit_it", ablate=True)))
        ),
    )
    rows = [
        {"value": 5, "hide_it": True, "omit_it": False},
        None,
        {"value": 8, "hide_it": True, "omit_it": True},
        {"value": 9, "hide_it": False, "omit_it": True},
    ]
    source = pa.table({"events": [rows, [], None]})
    inputs = model.encode(source, strata="train")
    field = inputs["/events/value"]
    assert field.present[0].flatten().tolist() == [True, False, False, False]
    assert field.trainable[0].flatten().tolist() == [True, False, True, True]
    assert field.targets["content"][field.trainable].tolist() == [5, 8, 9]
    geom = inputs["/"]["/events"]
    assert geom.lengths.flatten().tolist() == [4, 0, 0]
    assert geom.slots[0].flatten().tolist() == [True, False, True, True]


@rf.preprocess
def threshold(frame: pl.DataFrame, *, cutoff: float) -> pl.DataFrame:
    return frame.with_columns((pl.col("amount") > cutoff).fill_null(False).alias("mask_amount"))


def test_arrow_preprocessor_selector_and_partial_binding():
    prepare = threshold.partial(cutoff=1_000.0)
    configured = rf.Model(
        preset="small",
        d_model=8,
        amount=rf.Number(task=rf.reconstruct(query="mask_amount")),
    )
    source = pa.table({"amount": [100.0, 2_000.0, None]})

    field = configured.encode(
        source,
        preprocess=prepare,
        strata="predict",
    )[rf.Address("/amount")]

    assert field.present[:, 0].all()
    assert field.state[:, 0].tolist() == [rf.Tokens.valued, rf.Tokens.masked, rf.Tokens.null]


def test_arrow_preprocessor_attaches_to_data_module():
    configured = rf.Model(
        preset="small",
        d_model=8,
        batch_size=3,
        amount=rf.Number(task=rf.reconstruct(query="mask_amount")),
    )
    data = rf.ArrowDataModule(
        model=configured,
        validate=pa.table({"amount": [100.0, 2_000.0, None]}),
        preprocessor=threshold.partial(cutoff=1_000.0),
        shuffle=False,
    )

    encoded = next(iter(data.val_dataloader()))
    field = encoded.tensors[rf.Address("/amount")]

    assert field.present[:, 0].all()
    assert field.state[:, 0].tolist() == [rf.Tokens.valued, rf.Tokens.masked, rf.Tokens.null]


@rf.preprocess
def prediction_policy(frame: pl.DataFrame, *, strata: rf.Strata) -> pl.DataFrame:
    values = pl.col("deny_amount").fill_null(False) if strata == rf.Strata.predict else pl.lit(False)
    return frame.with_columns(values.alias("mask_amount"))


def test_stratum_aware_selector():
    configured = rf.Model(
        preset="small",
        d_model=8,
        amount=rf.Number(task=rf.reconstruct(query="mask_amount")),
    )
    source = pa.table({"amount": [1.0, 2.0], "deny_amount": [True, False]})

    training = configured.encode(source, preprocess=prediction_policy, strata="train")
    prediction = configured.encode(source, preprocess=prediction_policy, strata="predict")

    assert training[rf.Address("/amount")].present[:, 0].tolist() == [True, True]
    assert prediction[rf.Address("/amount")].state[:, 0].ne(rf.Tokens.masked).tolist() == [False, True]


EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("kind", pa.large_string()),
            pa.field("amount", pa.float64()),
        ]
    )
)


@rf.preprocess
def refunds(frame: pl.DataFrame) -> pl.DataFrame:
    selected = (pl.element().struct.field("amount") < 0).fill_null(False).alias("mask_event")
    return frame.with_columns(pl.col("events").list.eval(pl.element().struct.with_fields(selected)))


def events() -> pa.Table:
    return pa.Table.from_pylist(
        [
            {
                "events": [
                    {"kind": "sale", "amount": 5.0},
                    {"kind": "refund", "amount": -2.0},
                    {"kind": "unknown", "amount": None},
                ]
            }
        ],
        schema=pa.schema([pa.field("events", EVENTS)]),
    )


def test_arrow_selector_is_atomic_for_a_branch():
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=3,
            task=rf.reconstruct(query="mask_event"),
            kind=rf.Category,
            amount=rf.Number,
        ),
    )

    encoded = configured.encode(events(), preprocess=refunds, strata="predict")
    amount = encoded[rf.Address("/events/amount")]
    kind = encoded[rf.Address("/events/kind")]

    assert amount.inferred.tolist() == kind.inferred.tolist() == [[[False, True, False]]]


def test_shared_selector_can_drive_different_leaf_effects():
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=3,
            kind=rf.Category(
                hide="mask_event",
            ),
            amount=rf.Number(
                task=rf.reconstruct(query="mask_event"),
            ),
        ),
    )

    encoded = configured.encode(events(), preprocess=refunds, strata="predict")
    amount = encoded[rf.Address("/events/amount")]
    kind = encoded[rf.Address("/events/kind")]

    assert amount.inferred.tolist() == [[[False, True, False]]]
    assert kind.present.tolist() == [[[True, True, True]]]
    assert amount.state.tolist() == [[[rf.Tokens.valued, rf.Tokens.masked, rf.Tokens.null]]]
    assert kind.state.tolist() == [[[rf.Tokens.valued, rf.Tokens.valued, rf.Tokens.valued]]]


@rf.preprocess
def recent(frame: pl.DataFrame) -> pl.DataFrame:
    selected = (pl.int_range(pl.len()) >= pl.len() - 2).alias("mask_event")
    return frame.with_columns(pl.col("events").list.eval(pl.element().struct.with_fields(selected)))


def test_last_n_selector():
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=3,
            amount=rf.Number(task=rf.reconstruct(query="mask_event")),
        ),
    )

    field = configured.encode(
        events(),
        preprocess=recent,
        strata="predict",
    )[rf.Address("/events/amount")]

    assert field.inferred.tolist() == [[[False, True, True]]]


RISK_EVENTS = pa.large_list(
    pa.struct(
        [
            pa.field("amount", pa.float64()),
            pa.field("risk", pa.float64()),
        ]
    )
)


@rf.preprocess
def top_risk(frame: pl.DataFrame, *, count: int = 2) -> pl.DataFrame:
    selected = (
        pl.element().struct.field("risk").fill_null(float("-inf")).rank("ordinal", descending=True) <= count
    ).alias("mask_event")
    return frame.with_columns(pl.col("events").list.eval(pl.element().struct.with_fields(selected)))


def test_exact_k_selector():
    source = pa.Table.from_pylist(
        [
            {
                "events": [
                    {"amount": 10.0, "risk": 0.1},
                    {"amount": 20.0, "risk": 0.9},
                    {"amount": 30.0, "risk": 0.4},
                ]
            }
        ],
        schema=pa.schema([pa.field("events", RISK_EVENTS)]),
    )
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=3,
            amount=rf.Number(task=rf.reconstruct(query="mask_event")),
        ),
    )

    field = configured.encode(
        source,
        preprocess=top_risk,
        strata="predict",
    )[rf.Address("/events/amount")]

    assert field.inferred.tolist() == [[[False, True, True]]]


@pytest.mark.parametrize(
    ("preprocess", "datatype"),
    [
        (refunds, EVENTS),
        (recent, EVENTS),
        (top_risk, RISK_EVENTS),
    ],
)
def test_nested_selectors_preserve_all_null_schema(preprocess, datatype):
    source = pa.Table.from_arrays(
        [pa.nulls(2, type=datatype)],
        names=["events"],
    )
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=3,
            amount=rf.Number(task=rf.reconstruct(query="mask_event")),
        ),
    )

    field = configured.encode(
        source,
        preprocess=preprocess,
        strata="predict",
    )[rf.Address("/events/amount")]

    assert not field.present.any()


@rf.preprocess
def broadcast(frame: pl.DataFrame) -> pl.DataFrame:
    denied = pl.col("events").list.eval(pl.element().struct.with_fields(pl.lit(True).alias("mask_event")))
    allowed = pl.col("events").list.eval(pl.element().struct.with_fields(pl.lit(False).alias("mask_event")))
    return frame.with_columns(pl.when(pl.col("deny_events").fill_null(False)).then(denied).otherwise(allowed))


def test_observation_selector_broadcasts_into_nested_records():
    source = pa.Table.from_pylist(
        [
            {
                "events": [
                    {"kind": "sale", "amount": 1.0},
                    {"kind": "sale", "amount": 2.0},
                ],
                "deny_events": True,
            },
            {
                "events": [{"kind": "sale", "amount": 3.0}],
                "deny_events": False,
            },
        ],
        schema=pa.schema(
            [
                pa.field("events", EVENTS),
                pa.field("deny_events", pa.bool_()),
            ]
        ),
    )
    configured = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=2,
            task=rf.reconstruct(query="mask_event"),
            amount=rf.Number,
        ),
    )

    field = configured.encode(
        source,
        preprocess=broadcast,
        strata="predict",
    )[rf.Address("/events/amount")]

    assert field.inferred.tolist() == [
        [[True, True]],
        [[False, False]],
    ]


def test_train_only_query_is_not_required_during_prediction():
    configured = rf.Model(preset="small", d_model=8, value=rf.Number(hide="eligible"))

    field = configured.encode(
        pa.table({"value": [1.0, 2.0]}),
        strata="predict",
    )[rf.Address("/value")]

    assert field.state[:, 0].tolist() == [rf.Tokens.valued, rf.Tokens.valued]


def test_unconditionally_masked_query_target_can_be_absent_at_prediction():
    configured = rf.Model(
        preset="small",
        d_model=8,
        feature=rf.Number,
        label=rf.Boolean(query="labels.value", predict=True),
    )

    source = pa.table({"feature": [1.0, 2.0]})
    field = configured.encode(source, strata="predict")["/label"]
    assert field.inferred.all()
    assert len(configured.predict(source)) == 2
