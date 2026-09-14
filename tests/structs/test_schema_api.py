import json
import pickle

import pytest

import relflow as rf
from relflow.structs.enums import TensorKey


def test_schema_pickle_discards_derived_selection_predicates():
    model = rf.Model(id=rf.Category(size=32), d_model=8, n_layers=1, n_heads=2)
    selected = model.select(rf.where("type") == "category")

    restored = pickle.loads(pickle.dumps(model.schema))

    assert selected == [model.schema.requests["record/id"]]
    assert restored._selection_cache == {}
    assert restored.select(rf.where("type") == "category") == [restored.requests["record/id"]]


def test_model_constructor_supports_direct_binding_and_opt_in_queries():
    model = rf.Model(
        job_code=rf.Category(query='source["job code"]', description="Job code from OpenML", size=128),
        amount=rf.Number(),
        label=rf.Category(mask=True, embed=False, topk=[2, 3]),
        d_model=32,
        n_layers=2,
        n_heads=4,
        batch_size=8,
    )
    params = model.schema

    assert model.batch_size == 8
    assert params.d_model == 32
    assert params.fields.name == "record"
    assert params.fields.length == 1
    assert params.fields.n_layers == 2
    assert params.fields.n_heads == 4

    job = params.requests["record/job_code"]
    assert job.name == "job_code"
    assert job.description == "Job code from OpenML"
    assert job.query == 'source["job code"]'
    assert job.size == 128

    amount = params.requests["record/amount"]
    assert amount.query is None
    assert amount.active is True
    assert amount.embed is False

    label = params.requests["record/label"]
    assert label.mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert label.embed is False
    assert label.topk == [2, 3]
    assert params.reconstruct == ["record/label"]


def test_model_uses_constructor_without_from_tree_alternative():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4, reduction=rf.Mean())

    assert model.schema.requests["record/amount"].name == "amount"
    assert not hasattr(rf.Model, "from_tree")


def test_model_constructor_rejects_duplicate_sources():
    with pytest.raises(ValueError, match="duplicate field name"):
        rf.Model(amount=rf.Number(), fields={"amount": rf.Number()}, d_model=16, n_layers=1, n_heads=4)


def test_model_constructor_accepts_branch_nodes_with_optional_leaf_queries():
    model = rf.Model(
        transactions=rf.Branch(
            amount=rf.Number(),
            merchant_code=rf.Category(query='source["merchant code"]', description="merchant code", size=32),
            length=4,
        ),
        d_model=16,
        n_layers=1,
        n_heads=4,
    )
    params = model.schema

    assert "record/transactions" in params.branches

    amount = params.requests["record/transactions/amount"]
    assert amount.query is None
    assert params.shapes["record/transactions/amount"] == (1, 4)

    merchant = params.requests["record/transactions/merchant_code"]
    assert merchant.name == "merchant_code"
    assert merchant.description == "merchant code"
    assert merchant.query == 'source["merchant code"]'
    assert merchant.size == 32


def test_branch_mask_shorthand_normalizes_and_exports_public_api():
    policy = rf.Mask(query="recent", rate=0.5)
    branch = rf.Branch(amount=rf.Number(), length=4, mask=policy)
    model = rf.Model(transactions=branch, d_model=16, n_layers=1, n_heads=4)

    bound = model.schema.branches["record/transactions"]
    assert bound.mask == (policy,)


def test_branch_mask_validation_rejects_invalid_bound_configs():
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        rf.Schema.model_validate(
            {
                "d_model": 16,
                "fields": {
                    "name": "record",
                    "type": "branch",
                    "length": 1,
                    "masks": [{"count": 1}],
                    "fields": [{"name": "amount", "type": "number"}],
                },
            }
        )

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        rf.Model(
            transactions=rf.Branch(amount=rf.Number(), length=2, mask=rf.Mask.model_validate({"offset": 2})),
            d_model=16,
            n_layers=1,
            n_heads=4,
        )

    with pytest.raises(TypeError, match="entries must be Mask"):
        rf.Model(transactions=rf.Branch(amount=rf.Number(), length=2, mask=[0.5]), d_model=16, n_layers=1, n_heads=4)


def test_model_constructor_accepts_root_branch_options():
    model = rf.Model(
        amount=rf.Number(),
        d_model=16,
        n_layers=2,
        n_heads=4,
        name="events",
        description="event records",
        embed=True,
        attention=None,
        reduction=rf.Attention(n_outputs=3, n_layers=2),
        dropout=0.2,
    )
    params = model.schema

    assert params.fields.name == "events"
    assert params.fields.description == "event records"
    assert params.fields.embed is True
    assert params.fields.attention is None
    assert params.fields.length == 1
    assert params.fields.reduction == rf.Attention(n_outputs=3, n_layers=2)
    assert params.fields.dropout == 0.2
    assert params.embed == ["events"]
    assert params.shapes["events/amount"] == (1,)


def test_disabled_attention_round_trips_as_null() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        attention=None,
        items=rf.Branch(length=2, attention=None, value=rf.Number),
    )
    schema = model.schema

    serialized = schema.model_dump_json(round_trip=True)
    payload = json.loads(serialized)

    assert payload["fields"]["attention"] is None
    assert payload["fields"]["fields"][0]["attention"] is None
    restored = rf.Schema.model_validate_json(serialized)
    assert restored.model_dump(mode="python") == schema.model_dump(mode="python")
    assert all(branch.attention is None for branch in restored.branches.values())


@pytest.mark.parametrize("nested", [False, True])
def test_schema_rejects_string_none_attention(nested: bool) -> None:
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        items=rf.Branch(length=2, value=rf.Number),
    )
    payload = schema.model_dump(mode="python")
    branch = payload["fields"]["fields"][0] if nested else payload["fields"]
    branch["attention"] = "none"

    with pytest.raises(ValueError, match="attention"):
        rf.Schema.model_validate(payload)


@pytest.mark.parametrize("constructor", [rf.Model, rf.Schema.from_tree])
def test_tree_constructors_reject_children_that_are_not_definitions(constructor) -> None:
    with pytest.raises(TypeError, match="tree field 'unexpected'.*must be a Branch, Leaf, or Leaf class"):
        constructor(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4, unexpected=2)


def test_schema_rejects_embedding_a_branch_without_active_output() -> None:
    with pytest.raises(ValueError, match="embed=True but no active descendant output"):
        rf.Model(
            empty=rf.Branch(embed=True, value=rf.Number(active=False)),
            target=rf.Number(mask=True),
            d_model=8,
            n_layers=1,
            n_heads=2,
        )


def test_model_constructor_rejects_root_length_argument():
    with pytest.raises(TypeError, match="tree field 'length'"):
        rf.Model(amount=rf.Number(), d_model=16, n_layers=2, n_heads=4, length=3)


def test_model_constructor_accepts_root_mask():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=2, n_heads=4, mask=True)

    assert model.schema.fields.mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert model.schema.reconstruct == ["record/amount"]


def test_model_select_returns_nodes_and_update_refreshes_cached_role_views():
    model = rf.Model(amount=rf.Number(), label=rf.Category(mask=True, embed=False), d_model=16, n_layers=1, n_heads=4)
    params = model.schema

    numeric = rf.where("type") == "number"
    assert model.select(numeric) == model.select(rf.where("type") == "number")

    model.update(numeric, weight=2.0)
    assert params.requests["record/amount"].weight == 2.0

    model.update(rf.where("name") == "amount", description="Transaction amount")
    assert model.select(rf.where("description") == "Transaction amount") == [params.requests["record/amount"]]

    reconstruct = rf.where("reconstruct")
    assert model.select(reconstruct, include_root=False) == [params.requests["record/label"]]

    model.update(rf.where("name") == "amount", mask=True)
    assert params.requests["record/amount"].mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert model.select(reconstruct, include_root=False) == [
        params.requests["record/amount"],
        params.requests["record/label"],
    ]

    model.update(rf.where("name") == "amount", mask=False)
    assert params.requests["record/amount"].mask == ()
    assert model.select(reconstruct, include_root=False) == [params.requests["record/label"]]


def test_schema_helper_classmethods_back_public_dsl():
    predicate = rf.NodePredicate.from_callable("amount-name", lambda node: node.name == "amount")
    attribute = rf.NodeAttribute.named("name")

    assert rf.predicate("amount-name", lambda node: node.name == "amount").key == predicate.key
    assert rf.where("name") == attribute


def test_schema_select_returns_nodes_and_accepts_boolean_predicates():
    model = rf.Model(amount=rf.Number(), memo=rf.Number(active=False), d_model=16, n_layers=1, n_heads=4)
    params = model.schema

    active = params.select(rf.where("active"), include_root=False)
    inactive = params.select(~rf.where("active"), include_root=False)

    assert isinstance(active, list)
    assert active == [params.requests["record/amount"]]
    assert inactive == [params.requests["record/memo"]]

    model.update(rf.where("name") == "memo", mask=True)
    assert params.requests["record/memo"].mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)
    assert params.select(rf.where("reconstruct"), include_root=False) == []

    with pytest.raises(TypeError, match="Python 'not where"):
        not rf.where("active")


def test_model_update_can_deactivate_and_reactivate_leaf_nodes():
    model = rf.Model(
        amount=rf.Number(), memo=rf.Number(active=False, mask=0.5, embed=True), d_model=16, n_layers=1, n_heads=4
    )
    params = model.schema

    assert "record/memo" in params.requests
    assert "record/memo" not in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == []
    inactive = model.select(lambda node: getattr(node, "active", True) is False)
    assert inactive[0].address == "record/memo"

    model.update(rf.where("name") == "memo", active=True)

    assert "record/memo" in params.requests
    assert "record/memo" in params.active_requests
    assert "record/memo" in model.nodes
    assert params.embed == ["record/memo"]


def test_model_update_applies_validated_values_before_rebuilding_modules():
    model = rf.Model(label=rf.Category(size=8, topk=[2]), d_model=16, n_layers=1, n_heads=4)
    address = "record/label"
    before = model.nodes[address]

    model.update(rf.where("name") == "label", size=16, topk=[3, 2])

    request = model.schema.requests[address]
    assert request.size == 16
    assert request.topk == [2, 3]
    assert model.nodes[address] is not before
    assert model.nodes[address].embedder.size == 16
    assert model.nodes[address].embedder.embeddings[TensorKey.content.name].num_embeddings == 16


def test_model_update_disables_attention_without_removing_reduction() -> None:
    model = rf.Model(
        d_model=8,
        n_layers=2,
        n_heads=2,
        items=rf.Branch(length=3, first=rf.Number, second=rf.Number, reduction=rf.Attention(n_outputs=2)),
    )
    address = "record/items"
    before = model.nodes[address].encoder
    reduction = {name: value.detach().clone() for name, value in before.pool.state_dict().items()}
    assert before.coordinate_encoder is not None
    assert len(before.encoder) == 1

    model.update(rf.where("address") == address, attention=None)

    encoder = model.nodes[address].encoder
    assert model.schema.branches[address].attention is None
    assert encoder.coordinate_encoder is None
    assert len(encoder.encoder) == 0
    assert model.schema.branches[address].reduction == rf.Attention(n_outputs=2)
    assert encoder.pool.state_dict().keys() == reduction.keys()
    for name, value in encoder.pool.state_dict().items():
        assert value.equal(reduction[name])


def test_model_update_uses_current_schema_when_selection_cache_is_stale():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4)
    predicate = rf.where("name") == "amount"

    assert model.select(predicate) == [model.schema.requests["record/amount"]]

    request = model.schema.requests["record/amount"]
    request.name = "renamed"

    model.update(predicate, weight=2.0)

    assert request.weight == 1.0
    assert "record/amount" not in model.schema.requests
    assert "record/renamed" in model.schema.requests
    assert "record/amount" not in model.nodes
    assert "record/renamed" in model.nodes


def test_model_extend_appends_fields_under_one_selected_array_and_rebuilds_modules():
    model = rf.Model(transactions=rf.Branch(amount=rf.Number(), length=4), d_model=16, n_layers=1, n_heads=4)
    params = model.schema

    model.extend(rf.where("address") == "record/transactions", risk_score=rf.Number())

    assert "record/transactions/risk_score" in params.requests
    assert "record/transactions/risk_score" in model.nodes


def test_model_extend_appends_category_field_and_preserves_existing_vocabulary():
    model = rf.Model(label=rf.Category(size=10), d_model=16, n_layers=1, n_heads=4)

    label_vocab = model.nodes["record/label"].embedder.vocab
    label_vocab.extend(["alpha", "beta"])

    model.extend(rf.where("name") == "record", caretaker=rf.Category(size=10))

    assert "record/caretaker" in model.schema.requests
    assert "record/caretaker" in model.nodes
    assert model.nodes["record/label"].embedder.vocab.snapshot() == ["alpha", "beta"]
    assert model.nodes["record/caretaker"].embedder.vocab.snapshot() == []


def test_model_extend_defaults_to_root_when_only_one_array_matches():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4)

    model.extend(risk_score=rf.Number())

    assert "record/risk_score" in model.schema.requests
    assert "record/risk_score" in model.nodes


def test_model_delete_removes_nodes_permanently_and_rebuilds_modules():
    model = rf.Model(amount=rf.Number(), risk_score=rf.Number(), d_model=16, n_layers=1, n_heads=4)
    params = model.schema

    model.delete(rf.where("name") == "risk_score")

    assert "record/risk_score" not in params.requests
    assert "record/risk_score" not in model.nodes


def test_model_delete_rejects_removing_the_final_request():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4)

    with pytest.raises(ValueError, match="every request"):
        model.delete(rf.where("name") == "amount")


def test_model_reset_reinitializes_runtime_node_without_changing_schema():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4)
    before = model.nodes["record/amount"]

    model.reset(rf.where("name") == "amount")

    assert model.nodes["record/amount"] is not before
    assert "record/amount" in model.schema.requests


def test_model_override_temporarily_updates_schema_and_rebuilds_modules():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4)
    before = model.nodes["record/amount"]

    with model.override(rf.where("name") == "amount", active=False):
        assert "record/amount" not in model.schema.active_requests
        assert model.nodes["record/amount"] is not before

    assert "record/amount" in model.schema.active_requests


def test_model_override_mask_restores_original_policy():
    model = rf.Model(amount=rf.Number(mask=0.25), d_model=16, n_layers=1, n_heads=4)
    request = model.schema.requests["record/amount"]

    with model.override(rf.where("name") == "amount", mask=True):
        assert request.mask == (rf.Mask(skip=True, dropout=False, reconstruct=True),)

    assert request.mask == (rf.Mask(rate=0.25),)


def test_model_mutations_are_blocked_inside_training_loop_lock():
    model = rf.Model(amount=rf.Number(), d_model=16, n_layers=1, n_heads=4)
    lock = rf.MutationLockCallback()

    lock.on_train_start(trainer=None, pl_module=model)
    try:
        with pytest.raises(RuntimeError, match="active loop: train"):
            model.update(rf.where("name") == "amount", weight=2.0)
    finally:
        lock.on_train_end(trainer=None, pl_module=model)

    model.update(rf.where("name") == "amount", weight=2.0)
    assert model.schema.requests["record/amount"].weight == 2.0
