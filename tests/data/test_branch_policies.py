"""Branch policies choose rows once and preserve their ownership in nested data."""

from itertools import pairwise

import numpy as np
import polars as pl
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture import runtime
from relflow.data import ragged
from relflow.data.iterables import encode


def test_field_budget_skips_unselected_targets_decoders_and_losses():
    names = [f"x{i}" for i in range(16)]
    source = pa.table({name: [float(i), None, float(i + 1)] for i, name in enumerate(names)})
    model = rf.Model(
        d_model=8,
        n_heads=2,
        n_layers=1,
        task=rf.reconstruct(fields=3, ablate=True),
        fields=dict.fromkeys(names, rf.Number),
    )
    batch = model.encode(source, strata="train", seed=82)
    chosen = {address for address in model.schema.objectives if batch[address].trainable.any()}
    assert len(chosen) == 3
    assert all(batch[address].targets.is_empty() == (address not in chosen) for address in model.schema.objectives)
    for address in model.schema.objectives:
        assert batch[address].present.flatten().tolist() == [address not in chosen] * 3
        if address in chosen:
            assert batch[address].targets[rf.TensorKey.state].flatten().tolist() == [
                rf.Tokens.valued,
                rf.Tokens.null,
                rf.Tokens.valued,
            ]
    predictions = model(batch, strata="train")
    assert {p.address for p in predictions} == chosen
    loss = model.training_step(batch, 0)["loss"]
    assert torch.isfinite(loss)
    loss.backward()
    for address in model.schema.objectives:
        assert any(p.grad is not None for p in model.nodes[address].decoder.parameters()) == (address in chosen)
    # A peer can select a field whose answer storage is empty on this rank.
    model.zero_grad(set_to_none=True)
    peer = next(a for a in model.schema.objectives if a not in chosen)
    participating = chosen | {peer}
    predictions = runtime.forward(model, batch, strata=rf.Strata.train, participating=participating)
    runtime.objective(
        model, batch, predictions, strata=rf.Strata.train, local=chosen, participating=participating, batch_idx=0
    ).backward()
    peer_gradients = [p.grad for p in model.nodes[peer].decoder.parameters() if p.grad is not None]
    assert peer_gradients and all(g.count_nonzero() == 0 for g in peer_gradients)
    assert model.schema.decodes == []
    predicted = model.encode(source, strata="predict")
    assert all(predicted[a].present.all() and predicted[a].targets.is_empty() for a in model.schema.objectives)


def test_branch_field_draws_are_reproducible_and_independent_of_encoder_omission():
    names = [f"x{i}" for i in range(8)]
    events = [{**dict.fromkeys(names, i), "keep": i % 3 == 0, "always": i} for i in range(64)]
    source = pa.table({"events": [events, [], None, [None, *events[:2]]]})

    def schema(ablate):
        return rf.Schema.from_tree(
            d_model=8,
            n_layers=1,
            n_heads=2,
            events=rf.Branch(
                task=rf.reconstruct(fields=3, rate=0.8, span=2, query="keep", ignore="always", ablate=ablate),
                always=rf.Number,
                unused=rf.Number(active=False),
                fields=dict.fromkeys(names, rf.Number),
            ),
        )

    masked, ablated = schema(False), schema(True)

    def selected(spec, split, epoch, ordinal):
        batch = encode(source, spec, split, {}, seed=38, epoch=epoch, ordinal=ordinal).tensors
        assert not batch["/events/always"].trainable.any()
        return {a: b.trainable for a in spec.reconstruct if (b := batch[a]).trainable.any()}

    original = selected(masked, rf.Strata.train, 0, 0)
    omitted = selected(ablated, rf.Strata.train, 0, 0)
    assert len(original) == 3 and original.keys() == omitted.keys()
    for address, positions in original.items():
        assert torch.equal(positions, omitted[address])
        assert not positions[1:3].any() and not positions[3, 0, 0]
        assert not positions[0, 0, 1::3].any() and not positions[0, 0, 2::3].any()
    assert any(selected(masked, rf.Strata.train, epoch, 0).keys() != original.keys() for epoch in range(1, 5))
    assert any(selected(masked, rf.Strata.train, 0, ordinal).keys() != original.keys() for ordinal in range(1, 5))
    for split in (rf.Strata.validate, rf.Strata.test):
        first, later = selected(masked, split, 0, 2), selected(masked, split, 4, 2)
        assert len(first) == 3 and first.keys() == later.keys()
        assert all(torch.equal(first[a], later[a]) for a in first)


def test_mask_unions_shared_branch_decisions_and_preserves_exempt_subtrees():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            hide=("flag", 0.4),
            ignore=("planned", "details"),
            units=rf.Number,
            value=rf.Number,
            planned=rf.Boolean,
            details=rf.Branch(score=rf.Number(hide="private")),
        ),
    )
    rows = [
        {
            "units": index,
            "value": index,
            "planned": True,
            "flag": index % 2 == 0,
            "details": [{"score": index, "private": index % 3 == 0}],
        }
        for index in range(128)
    ]
    records = pa.table({"events": [rows]})
    batch = model.encode(records, strata="train", seed=61)
    chosen = batch["/events/units"].state.flatten() == rf.Tokens.masked
    assert chosen[::2].all() and 0 < chosen[1::2].sum() < 64
    assert torch.equal(batch["/events/units"].state, batch["/events/value"].state)
    assert batch["/events/units"].present.all()
    assert (batch["/events/planned"].state == rf.Tokens.valued).all()
    assert (batch["/events/details/score"].state.flatten() == rf.Tokens.masked).tolist() == [
        index % 3 == 0 for index in range(128)
    ]
    assert model.schema.objectives == []
    for strata in ("validate", "test", "predict"):
        retained = model.encode(records, strata=strata)
        assert (retained["/events/units"].state == rf.Tokens.valued).all()
        assert (retained["/events/details/score"].state == rf.Tokens.valued).all()


@pytest.mark.parametrize("ablate", [False, True])
def test_sampled_spans_share_targets_stop_at_query_boundaries_and_repeat_during_validation(ablate):
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            task=rf.reconstruct(rate=0.4, query="eligible", span=3, ignore="planned", ablate=ablate),
            first=rf.Number,
            details=rf.Branch(second=rf.Number),
            planned=rf.Number(task=rf.reconstruct(query="held_out")),
        ),
    )
    rows = [
        {
            "first": index,
            "details": [{"second": index}],
            "planned": index,
            "eligible": index % 16 != 15,
            "held_out": index % 5 == 0,
        }
        for index in range(256)
    ]
    source = pa.table({"events": [rows, rows]})

    def targets(strata, epoch):
        batch = model.encode(source, strata=strata, seed=903, epoch=epoch)
        chosen = batch["/events/first"].trainable.reshape(2, 256)
        expected = ~chosen if ablate else torch.ones_like(chosen)
        assert torch.equal(batch["/events/first"].present.reshape(2, 256), expected)
        assert torch.equal(batch["/events/details/second"].present.reshape(2, 256), expected)
        assert batch["/events/planned"].present.all()
        assert torch.equal(chosen, batch["/events/details/second"].trainable.reshape(2, 256))
        assert batch["/events/planned"].trainable.reshape(2, 256).tolist() == [[row["held_out"] for row in rows]] * 2
        return chosen

    baseline = targets("validate", 0)
    assert torch.equal(baseline, targets("validate", 12))
    assert not torch.equal(targets("train", 0), targets("train", 12))
    assert not baseline[:, 15::16].any()
    assert 0 < baseline.sum() < baseline.numel()
    for parent in baseline.reshape(-1, 16).tolist():
        for index, selected in enumerate(parent[:13]):
            if selected and (index == 0 or not parent[index - 1]):
                assert all(parent[index : index + 3])


def test_ablation_deactivates_whole_nested_collections_per_parent_only_in_training():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        customers=rf.Branch(
            credit=rf.Number(predict=True),
            orders=rf.Branch(ablate=0.5, items=rf.Branch(units=rf.Number(predict=True))),
        ),
    )
    orders = [{"items": [{"units": 3}]}, None, {"items": [{"units": 4}, {"units": 6}]}]
    source = pa.table({"customers": [[{"credit": i, "orders": orders} for i in range(64)]] * 2})
    batch = model.encode(source, strata="train", seed=49)
    geometry = batch["/"]["/customers/orders"]
    lengths = geometry.lengths.flatten()
    assert set(lengths.tolist()) == {0, 3}
    retained = lengths == 3
    assert torch.equal(geometry.present.flatten(), retained)
    assert geometry.slots.reshape(-1, 3)[retained].tolist() == [[True, False, True]] * retained.sum().item()
    nested = batch["/"]["/customers/orders/items"].lengths.reshape(-1, 3)
    assert nested[retained].tolist() == [[1, 0, 2]] * retained.sum().item()
    assert not nested[~retained].any()
    assert torch.equal(batch["/customers/orders/items/units"].trainable.reshape(-1, 6).sum(-1), 3 * retained)
    assert batch["/customers/credit"].trainable.all()
    for strata in ("validate", "test", "predict"):
        held_out = model.encode(source, strata=strata)
        assert (held_out["/"]["/customers/orders"].lengths == 3).all()
        assert held_out["/"]["/customers/orders/items"].lengths.reshape(-1, 3).tolist() == [[1, 0, 2]] * 128


def test_complete_ablation_removes_training_objectives_and_prediction_restores_collection():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        orders=rf.Branch(ablate=1.0, items=rf.Branch(units=rf.Number(predict=True))),
    )
    source = pa.table({"orders": [[{"items": [{"units": 6}, {"units": 7}]}], [], None]})
    batch = model.encode(source, strata="train")
    assert not batch["/"]["/orders"].present.any()
    assert batch["/"]["/orders"].slots.shape[-1] == 0
    assert not batch["/orders/items/units"].trainable.any()
    assert model.training_step(batch, 0) is None
    recovered = model.predict(source)["predictions"].struct.field("orders").to_list()
    assert len(recovered[0][0]["items"]) == 2
    assert recovered[1:] == [[], None]


def test_ablation_checks_overflow_only_for_retained_parents_across_arrow_chunks():
    model = rf.Model(
        preset="small",
        events=rf.Branch(length=1, overflow="error", ablate=0.5, name=rf.Category),
    )
    source = pa.table({"events": [[{"name": str(i)}] for i in range(32)]})
    keep = model.encode(source, strata="train", seed=7)["/"]["/events"].valid.flatten().tolist()
    assert any(keep) and not all(keep)

    # Independent dictionaries force the collection reader to retain chunk boundaries.
    def chunk(start, stop):
        names = [str(i) for i in range(start, stop) for _ in range(1 if keep[i] else 2)]
        records = pa.StructArray.from_arrays([pa.array(names).dictionary_encode()], names=["name"])
        offsets = np.r_[0, np.cumsum([1 if retain else 2 for retain in keep[start:stop]])]
        return pa.ListArray.from_arrays(pa.array(offsets, type=pa.int32()), records)

    dropped = keep.index(False)
    cuts = sorted({0, dropped, dropped + 1, 16, 32})
    chunked = pa.table({"events": pa.chunked_array([chunk(start, stop) for start, stop in pairwise(cuts)])})
    encoded = model.encode(chunked, strata="train", seed=7)
    assert encoded["/"]["/events"].valid.flatten().tolist() == keep
    assert encoded["/events/name"].present.flatten().tolist() == keep
    with pytest.raises(ValueError, match="overflow"):
        model.encode(chunked, strata="validate")


@pytest.mark.parametrize("value", [True, 1, "offline", -0.1, 1.01, float("nan")])
def test_ablation_requires_a_probability(value):
    with pytest.raises((TypeError, ValueError)):
        rf.Branch(ablate=value, units=rf.Number)
    with pytest.raises((TypeError, ValueError)):
        rf.Number(ablate=value)


@pytest.mark.parametrize("ignore", ["missing", "details.score", ("units", "units"), "", ["units"]])
def test_branch_exemptions_require_distinct_direct_child_names(ignore):
    with pytest.raises((TypeError, ValueError)):
        rf.Branch(hide=0.2, ignore=ignore, units=rf.Number)
    with pytest.raises((TypeError, ValueError)):
        rf.Branch(task=rf.reconstruct(rate=0.2, ignore=ignore), units=rf.Number)


def test_reconstruction_span_stops_at_null_rows_and_parent_boundaries(monkeypatch):
    def first_draw_only(layout, **kwargs):
        draws = np.full(len(layout.records), np.iinfo(np.uint64).max, dtype=np.uint64)
        draws[0] = 0
        return draws

    monkeypatch.setattr(ragged, "scores", first_draw_only)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(task=rf.reconstruct(rate=0.5, span=6), value=rf.Number),
    )
    source = pa.table({"events": [[{"value": 1}, {"value": 2}, None, {"value": 3}], [{"value": 4}]]})
    chosen = model.encode(source, strata="validate")["/events/value"].trainable
    assert chosen.reshape(2, 4).tolist() == [[True, True, False, False], [False] * 4]


def test_root_and_span_restrictions():
    with pytest.raises(TypeError, match="Set predict on a leaf or child branch"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, predict=True, value=rf.Number)
    with pytest.raises(TypeError, match="Set ablate on a leaf or child branch"):
        rf.Model(d_model=8, n_layers=1, n_heads=2, ablate=1.0, value=rf.Number)
    with pytest.raises(ValueError, match=r"reconstruct\(span=.*child branch"):
        rf.Number(task=rf.reconstruct(rate=0.2, span=2))
    with pytest.raises(ValueError, match=r"reconstruct\(ignore=.*branch or model"):
        rf.Number(task=rf.reconstruct(ignore="child"))


def test_policy_names_can_also_be_children_and_task_survives_schema_edits():
    class Responses(rf.Branch):
        task = rf.Number
        predict = rf.Boolean

    model = rf.Model(preset="small", fields={"task": Responses(predict=True)}, predict=rf.Boolean, ablate=rf.Number)
    assert {"/predict", "/ablate"} <= model.schema.requests.keys()
    assert model.schema.objectives == ["/task/task", "/task/predict"]
    model.update(rf.where("address") == "/task", predict=False, task=rf.reconstruct(query="review"))
    model.update(rf.where("address") == "/task/task", ablate=1.0)
    restored = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert restored.branches["/task"].task == rf.reconstruct(query="review")
    assert restored.requests["/task/task"].ablate == 1.0
    assert restored.requests["/task/predict"].embedded
    with pytest.raises((TypeError, ValueError), match="pretrain"):
        rf.Number(**{"pretrain": rf.reconstruct()})


def test_branch_prediction_matches_leaf_targets_and_preserves_supplied_rows():
    def model(branch_target):
        return rf.Model(
            preset="small",
            d_model=8,
            context=rf.Number,
            events=rf.Branch(
                predict=branch_target,
                score=rf.Number(predict=not branch_target),
                checks=rf.Branch(passed=rf.Boolean(predict=not branch_target)),
            ),
        )

    source = pa.table(
        {
            "context": [1, 2, 3],
            "events": [[{"score": 4, "checks": [{"passed": False}, None]}, None, {"score": 2, "checks": []}], [], None],
        }
    )
    missing = pa.table(
        {
            "context": [1, 2, 3],
            "events": [[{"checks": [{}, None]}, None, {"checks": []}], [], None],
        }
    )
    grouped, individual = model(True), model(False)
    individual.load_state_dict(grouped.state_dict())
    restored = rf.Schema.model_validate_json(grouped.schema.model_dump_json())
    assert restored.objectives == restored.decodes == ["/events/score", "/events/checks/passed"]
    assert all(not restored.requests[path].embedded for path in restored.decodes)
    for split in ("train", "validate", "test"):
        a, b = grouped.encode(source, strata=split), individual.encode(source, strata=split)
        for path in restored.decodes:
            assert not a[path].present.any()
            assert torch.equal(a[path].trainable, b[path].trainable)
    result = grouped.predict(missing)
    assert result.equals(grouped.predict(source))
    assert result.equals(individual.predict(missing))
    rows = result["predictions"].struct.field("events").to_list()
    assert rows[1:] == [[], None] and rows[0][1] is None
    first, _, last = rows[0]
    assert first["score"]["inferred"] and first["checks"][0]["passed"]["inferred"]
    assert first["checks"][1] is None and last["checks"] == []
    with pytest.raises(ValueError, match="below the model root"):
        grouped.update(rf.where("address") == "/", predict=True)


def test_leaf_ablation_samples_each_target_independently_before_observation():
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(predict=True, score=rf.Number(ablate=0.5), other=rf.Number(ablate=0.5), kept=rf.Number),
        missing=rf.Number(ablate=1.0, nullable=False, task=rf.reconstruct()),
    )
    events = [None if i % 7 == 0 else dict(score=i, other=i, kept=i) for i in range(128)]
    source = pa.table({"events": [events, [], None]})
    batch = model.encode(source, strata="train", seed=73)
    keep = batch["/events/score"].trainable
    sibling = batch["/events/other"].trainable
    assert not batch["/missing"].present.any() and not batch["/missing"].trainable.any()
    assert torch.equal(batch["/events/kept"].trainable.flatten(), batch["/"]["/events"].slots.flatten())
    assert (keep & ~sibling).any() and (sibling & ~keep).any()
    assert rf.Number.normalization(model, "/events/score")["count"] == keep.sum().item()
    assert torch.equal(keep, model.encode(source, strata="train", seed=73)["/events/score"].trainable)
    assert not torch.equal(keep, model.encode(source, strata="train", seed=73, epoch=1)["/events/score"].trainable)
    source = source.append_column("missing", pa.array([10, 11, 12]))
    for split in ("validate", "test", "predict"):
        fields = model.encode(source, strata=split)
        flag = "inferred" if split == "predict" else "trainable"
        assert torch.equal(getattr(fields["/events/score"], flag), getattr(fields["/events/kept"], flag))
        assert getattr(fields["/missing"], flag).all()


def test_ablated_duration_uses_original_predecessors_and_crop_positions():
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=12,
            overflow="tail",
            full=rf.Duration(query="at", delta=True),
            partial=rf.Duration(query="at", delta=True, ablate=0.5),
        ),
    )
    rows = [None if i % 6 == 0 else {"at": f"2026-01-{i + 1:02}"} for i in range(22)]
    source = pa.table({"events": [rows, None, [], rows]})
    fields = model.encode(source, strata="train", seed=9)
    full, partial = fields["/events/full"], fields["/events/partial"]
    chosen = partial.present
    assert chosen.any() and (full.present & ~chosen).any()
    assert torch.equal(partial.state[chosen], full.state[chosen])
    torch.testing.assert_close(partial.content[chosen], full.content[chosen])


@pytest.mark.parametrize("span", [None, 1, 4])
def test_full_rate_reconstructs_every_eligible_branch_row_at_prediction(span):
    model = rf.Model(
        preset="small",
        events=rf.Branch(
            task=rf.reconstruct(query="eligible", rate=1.0, span=span),
            value=rf.Number,
        ),
    )
    source = pa.table({"events": [[{"eligible": True, "value": None}, {"eligible": False, "value": 2}]]})
    assert model.schema.decodes == ["/events/value"]
    assert model.encode(source, strata="predict")["/events/value"].inferred.flatten().tolist() == [True, False]


@pytest.mark.parametrize(
    "options", [{"rate": True}, {"rate": 1.01}, {"rate": -0.1}, {"span": 2}, {"rate": 0.2, "span": 0}, {"query": "["}]
)
def test_reconstruction_rejects_invalid_selectors(options):
    with pytest.raises((TypeError, ValueError)):
        rf.reconstruct(**options)


def test_zero_rate_and_exemptions_survive_inspection_and_schema_edits():
    model = rf.Model(
        preset="small",
        items=rf.Branch(
            task=(rf.reconstruct(rate=0.0), rf.reconstruct(query="held", ignore="kind")),
            value=rf.Number,
            kind=rf.Category,
        ),
    )
    (binding,) = model.schema.tasks
    assert binding.owners == ("/items",) and binding.targets == ("/items/value",)
    assert binding.task == rf.reconstruct(query="held", ignore="kind")
    original = model.schema.model_dump_json()
    with pytest.raises(ValueError, match="ignored child"):
        model.delete(rf.where("address") == "/items/kind")
    with pytest.raises(ValueError, match="root is always present"):
        model.update(rf.where("address") == "/", ablate=0.5)
    assert model.schema.model_dump_json() == original
    model.update(rf.where("address") == "/items", task=rf.reconstruct(rate=0.0))
    model.delete(rf.where("address") == "/items/kind")
    assert model.schema.objectives == []


@pytest.mark.parametrize("strata", ["train", "validate", "test", "predict"])
def test_composed_reconstruction_unions_each_rules_own_selection_and_exemptions(strata):
    held = rf.reconstruct(query="held", ignore=("price", "context"))
    sampled = rf.reconstruct(rate=0.2, ignore="context")
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            hide="redacted",
            ignore="context",
            task=sampled,
            units=rf.Number,
            price=rf.Number,
            context=rf.Number(task=rf.reconstruct(query="review")),
        ),
    )
    count = 128
    rows = [
        {"units": i, "price": i, "context": i, "redacted": i % 3 == 0, "held": i % 4 == 0, "review": i % 7 == 0}
        for i in range(count)
    ]
    source = pa.table({"events": [rows]})
    baseline = model.encode(source, strata=strata, seed=71, epoch=1)
    model.update(rf.where("address") == "/events", task=(held, sampled))
    restored = rf.Schema.model_validate_json(model.schema.model_dump_json())
    assert restored.branches["/events"].task == (held, sampled)
    assert restored.decodes == ["/events/units", "/events/context"]
    batch = model.encode(source, strata=strata, seed=71, epoch=1)
    held_flags = torch.tensor([row["held"] for row in rows])
    redaction = torch.tensor([row["redacted"] and strata == "train" for row in rows])
    review = torch.tensor([row["review"] for row in rows])
    routed = "inferred" if strata == "predict" else "trainable"

    def selection(fields, name):
        return getattr(fields[f"/events/{name}"], routed).flatten()

    sampled_flags = selection(baseline, "units")
    assert torch.equal(selection(batch, "price"), sampled_flags)
    assert torch.equal(selection(batch, "units"), sampled_flags | held_flags)
    assert torch.equal(selection(batch, "context"), review)
    if strata == "predict":
        assert not sampled_flags.any()
    else:
        assert (sampled_flags & ~held_flags).any() and (held_flags & ~sampled_flags).any()
    assert torch.equal(
        batch["/events/units"].state.flatten() == rf.Tokens.masked, sampled_flags | held_flags | redaction
    )
    assert torch.equal(batch["/events/price"].state.flatten() == rf.Tokens.masked, sampled_flags | redaction)
    if strata == "validate":
        replay = model.encode(source, strata=strata, seed=71, epoch=19)
        assert torch.equal(selection(replay, "units"), selection(batch, "units"))


@pytest.mark.parametrize("split", ["train", "validate", "test", "predict"])
def test_preprocessor_redacts_ordered_rows_and_preserves_null_and_empty_collections(split):
    @rf.preprocess
    def redact_events(frame: pl.DataFrame) -> pl.DataFrame:
        event = pl.element()
        return frame.with_columns(pl.col("events").list.eval(pl.when(~event.struct.field("hidden")).then(event)))

    model = rf.Model(preset="small", events=rf.Branch(length=4, value=rf.Number))
    source = pl.DataFrame(
        {
            "events": [
                [{"value": 1, "hidden": False}, {"value": 99, "hidden": True}, None, {"value": 4, "hidden": False}],
                [],
                None,
            ]
        }
    )
    batch = model.encode(source, preprocess=redact_events, strata=split)
    geometry = batch["/"]["/events"]
    assert geometry.lengths.flatten().tolist() == [4, 0, 0]
    assert geometry.valid.flatten().tolist() == [True, True, False]
    assert geometry.slots.reshape(3, 4)[0].tolist() == [True, False, False, True]
    assert batch["/events/value"].content.reshape(3, 4)[0].tolist() == [1.0, 0.0, 0.0, 4.0]
