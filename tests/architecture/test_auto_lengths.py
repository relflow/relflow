from copy import deepcopy

import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.rotary import coordinates
from relflow.structs.enums import TensorKey


def test_structural_ranks_keep_finite_gaps_and_record_local_dynamic_offsets():
    shape = (2, 1, 3, 2)
    lengths = (
        torch.ones(2, dtype=torch.int64),
        torch.tensor([[2], [3]]),
        torch.tensor([[[2, 1, 0]], [[1, 2, 2]]]),
    )
    dynamic = coordinates(shape, (1, None, None), lengths, device=torch.device("cpu"))
    capped = coordinates(shape, (1, None, 4), lengths, device=torch.device("cpu"))

    assert dynamic[0, 0].tolist() == [[0, 1], [2, 3], [3, 4]]
    assert dynamic[1, 0].tolist() == [[0, 1], [1, 2], [3, 4]]
    assert capped[0, 0].tolist() == [[0, 1], [4, 5], [8, 9]]


@pytest.mark.parametrize("reduction", [rf.Attention(), rf.Mean()])
def test_nested_automatic_context_is_independent_of_batchmates(reduction):
    torch.manual_seed(109)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        embed=True,
        reduction=reduction,
        events=rf.Branch(
            reduction=None,
            amount=rf.Number,
            flag=rf.Boolean,
            items=rf.Branch(reduction=None, value=rf.Number, category=rf.Boolean),
            capped=rf.Branch(length=3, overflow="tail", reduction=None, value=rf.Number),
        ),
        answer=rf.Number(mask=True),
    ).eval()
    for module in model.modules():
        projection = getattr(module, "mass_projection", None)
        direction = getattr(module, "mass_direction", None)
        if projection is not None:
            with torch.no_grad():
                projection.weight.normal_(std=0.1)
                direction.normal_(std=0.1)

    short = {
        "events": [
            None,
            {
                "amount": 2.0,
                "flag": None,
                "items": [None, {"value": 3.0, "category": True}],
                "capped": [{"value": 4.0}],
            },
        ],
        "answer": 1.0,
    }
    long = {
        "events": [
            {
                "amount": float(index),
                "flag": True,
                "items": [{"value": float(item), "category": False} for item in range(index + 1)],
                "capped": [{"value": float(item)} for item in range(5)],
            }
            for index in range(7)
        ],
        "answer": 0.0,
    }
    parameters = {name: id(parameter) for name, parameter in model.named_parameters()}
    schema = model.schema.model_dump()
    reference = None
    for records, index in [([short], 0), ([short, long], 0), ([long, short], 1), ([short], 0)]:
        batch = model.encode(pa.Table.from_pylist(records), strata="predict")
        actual = {
            prediction.address: {
                key: value[index].detach().clone()
                for key, value in prediction.payload.items()
                if isinstance(value, torch.Tensor)
            }
            for prediction in model(batch, strata="predict")
        }
        if reference is None:
            reference = actual
        else:
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    assert parameters == {name: id(parameter) for name, parameter in model.named_parameters()}
    assert model.schema.model_dump() == schema


def test_dynamic_repeated_targets_share_queries_and_keep_optimizer_parameters():
    torch.manual_seed(113)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            reduction=None,
            items=rf.Branch(length=3, reduction=None, amount=rf.Number, label=rf.Number(mask=True)),
        ),
    )
    decoder = model.nodes["/events/items/label"].decoder
    assert decoder.pool.queries.shape == (3, 8)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    parameters = [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
    for length in (2, 6, 1):
        records = [
            {
                "events": [
                    {"items": [{"amount": float(index), "label": 2.0 * index} for index in range(2)]}
                    for _ in range(length)
                ]
            }
        ]
        batch = model.encode(pa.Table.from_pylist(records), strata="train")
        result = model.training_step(batch, batch_idx=0)
        assert result is not None
        assert torch.isfinite(result["loss"])
        result["loss"].backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        assert decoder.pool.queries.shape == (3, 8)
        assert parameters == [id(parameter) for group in optimizer.param_groups for parameter in group["params"]]
        assert parameters == [id(parameter) for parameter in model.parameters()]
    assert optimizer.state[decoder.pool.queries]["exp_avg"].shape == (3, 8)


def test_repeated_targets_keep_coordinates_when_neighboring_axes_grow():
    torch.manual_seed(117)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(
            reduction=None,
            items=rf.Branch(length=3, reduction=None, amount=rf.Number, label=rf.Number(mask=True, embed=True)),
        ),
    ).eval()
    short = {"events": [{"items": [{"amount": 1.0, "label": 2.0}] * 2}] * 2}
    long = {"events": [{"items": [{"amount": 3.0, "label": 4.0}] * 3}] * 5}
    reference = None
    for records, index in [([short], 0), ([short, long], 0), ([long, short], 1)]:
        batch = model.encode(pa.Table.from_pylist(records), strata="predict")
        prediction = model(batch, strata="predict")[0]
        selected = prediction.payload[TensorKey.inferred][index].reshape(-1)
        actual = {key: value[index][selected] for key, value in prediction.payload.items() if value.is_floating_point()}
        if reference is None:
            reference = actual
        else:
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)
    assert not torch.allclose(reference[TensorKey.embedding][0], reference[TensorKey.embedding][2])


@pytest.mark.parametrize("records", [[], [{"events": []}, {"events": None}]])
def test_empty_automatic_axes_support_backward_without_spurious_gradients(records):
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(reduction=None, amount=rf.Number, label=rf.Number(mask=True, embed=True)),
    )
    schema = pa.schema([("events", pa.list_(pa.struct([("amount", pa.float64()), ("label", pa.float64())])))])
    batch = model.encode(pa.Table.from_pylist(records, schema=schema), strata="predict")

    predictions = model(batch, strata="predict")

    assert len(predictions) == 1
    prediction = predictions[0]
    assert prediction.payload[TensorKey.embedding].shape == (len(records), 0, 8)
    sum(value.sum() for value in prediction.payload.values() if value.is_floating_point()).backward()
    decoder = model.nodes["/events/label"].decoder
    assert all(parameter.grad is not None for parameter in decoder.parameters())
    assert all(parameter.grad is None or not parameter.grad.any() for parameter in model.parameters())


def test_packed_and_dense_dynamic_positions_agree_with_nested_holes():
    torch.manual_seed(127)
    packed = rf.Model(
        d_model=8,
        n_layers=2,
        n_heads=2,
        embed=True,
        events=rf.Branch(reduction=None, amount=rf.Number, flag=rf.Boolean),
    ).eval()
    dense = deepcopy(packed)
    handles = [
        module.register_forward_hook(lambda owner, inputs, output: None)
        for module in dense.modules()
        if isinstance(module, BranchEncoder)
        for module in module.encoder
    ]
    observations = pa.Table.from_pylist(
        [
            {"events": [{"amount": 1.0, "flag": True}, None]},
            {"events": [{"amount": float(value), "flag": None} for value in range(5)]},
        ]
    )
    try:
        actual = packed(packed.encode(observations, strata="predict"), strata="predict")
        expected = dense(dense.encode(observations, strata="predict"), strata="predict")
    finally:
        for handle in handles:
            handle.remove()
    for left, right in zip(actual, expected, strict=True):
        torch.testing.assert_close(left.payload, right.payload, rtol=1e-5, atol=1e-5)


def test_finite_decoder_heritage_retains_original_flat_memory_positions():
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        reduction=None,
        events=rf.Branch(
            length=2,
            reduction=None,
            amount=rf.Number,
            target=rf.Number(mask=True, decoder_position=True),
        ),
        answer=rf.Number(mask=True, decoder_position=True),
    ).eval()
    observed = []

    def capture(owner, args, kwargs):
        observed.append((args[0].shape, kwargs["memory_positions"]))

    handles = [
        model.nodes[address].decoder.pool.register_forward_pre_hook(capture, with_kwargs=True)
        for address in ("/events/target", "/answer")
    ]
    table = pa.Table.from_pylist(
        [{"events": [{"amount": 1.0, "target": 2.0}, {"amount": 3.0, "target": 4.0}], "answer": 5.0}]
    )
    try:
        model(model.encode(table, strata="predict"), strata="predict")
    finally:
        for handle in handles:
            handle.remove()
    # Memory keeps the encoder's item-major numbering: the events block interleaves
    # amount and target per item (0, 2 | 1, 3) and each heritage parcel continues
    # after the previous span, so the storage order is not the position order.
    expected = {
        11: [0, 2, 1, 3, 4, 5, 7, 6, 8, 9, 10],  # /events/target: root, events, and its own leaf
        6: [0, 2, 1, 3, 4, 5],  # /answer: root and its own leaf
    }
    assert len(observed) == 2
    for shape, positions in observed:
        torch.testing.assert_close(positions, torch.tensor([expected[shape[1]]]), rtol=0, atol=0)


def test_three_dynamic_axes_preserve_ordered_scalar_and_repeated_predictions():
    torch.manual_seed(139)
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        reduction=None,
        customers=rf.Branch(
            reduction=None,
            orders=rf.Branch(
                reduction=None,
                items=rf.Branch(reduction=None, amount=rf.Number, target=rf.Number(mask=True, embed=True)),
            ),
        ),
        answer=rf.Number(mask=True, decoder_position=True),
    ).eval()

    def record(customers, orders, items):
        return {
            "customers": [
                {
                    "orders": [
                        {"items": [{"amount": float(index), "target": float(index + 1)} for index in range(items)]}
                        for _ in range(orders)
                    ]
                }
                for _ in range(customers)
            ],
            "answer": 1.0,
        }

    short, long = record(2, 2, 2), record(3, 4, 5)
    reference = None
    for records, index in [([short], 0), ([short, long], 0), ([long, short], 1)]:
        batch = model.encode(pa.Table.from_pylist(records), strata="predict")
        actual = {}
        for prediction in model(batch, strata="predict"):
            selected = prediction.payload[TensorKey.inferred][index].reshape(-1)
            actual[prediction.address] = {
                key: value[index][selected] for key, value in prediction.payload.items() if value.is_floating_point()
            }
        if reference is None:
            reference = actual
        else:
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-5)


def test_encoded_models_pass_structural_geometry_to_custom_compute(monkeypatch):
    model = rf.Model(d_model=8, n_layers=1, n_heads=2, amount=rf.Number, answer=rf.Number(mask=True)).eval()
    encoder = model.nodes["/"].encoder
    pool = model.nodes["/answer"].decoder.pool
    encode, compute = encoder.compute, pool.compute
    calls = []

    def custom_encoder(inputs, present, *, packing=None, positions=None):
        calls.append("encoder")
        assert packing is not None or positions is not None
        return encode(inputs, present, packing=packing, positions=positions)

    def custom_pool(memory, present, context, evidence, **geometry):
        calls.append("pool")
        assert geometry["memory_positions"] is not None
        return compute(memory, present, context, evidence, **geometry)

    monkeypatch.setattr(encoder, "compute", custom_encoder)
    monkeypatch.setattr(pool, "compute", custom_pool)
    batch = model.encode(pa.table({"amount": [1.0], "answer": [2.0]}), strata="predict")

    predictions = model(batch, strata="predict")

    assert calls == ["encoder", "pool"]
    assert all(torch.isfinite(value).all() for value in predictions[0].payload.values())


def test_custom_pool_compute_reports_incompatible_dynamic_queries(monkeypatch):
    model = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        events=rf.Branch(reduction=None, amount=rf.Number, target=rf.Number(mask=True)),
    ).eval()
    pool = model.nodes["/events/target"].decoder.pool
    compute = pool.compute

    def custom_pool(memory, present, context, evidence, **geometry):
        return compute(memory, present, None, evidence)

    monkeypatch.setattr(pool, "compute", custom_pool)
    table = pa.Table.from_pylist([{"events": [{"amount": 1.0, "target": 2.0}] * 3}])
    batch = model.encode(table, strata="predict")

    with pytest.raises(RuntimeError, match=r"custom pool compute.*\(1, 3, 8\).*\(1, 1, 8\).*module hooks"):
        model(batch, strata="predict")
