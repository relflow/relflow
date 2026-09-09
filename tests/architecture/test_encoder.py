from unittest.mock import patch

import pyarrow as pa
import torch

import relflow as rf
from relflow.architecture.encoder import BranchEncoder
from relflow.architecture.pool import MeanPool
from relflow.architecture.root import Model
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.experiment import Schema
from relflow.structs.packages import Parcel


def _payload(*, attention: str = "mha", pooling: str = "query") -> dict:
    field: dict = {
        "name": "category",
        "type": "category",
        "mask": True,
        "pooling": pooling,
        "size": 8,
    }
    return {
        "d_model": 16,
        "fields": {
            "name": "root",
            "type": "branch",
            "attention": attention,
            "dropout": 0.0,
            "fields": [
                {
                    "name": "items",
                    "type": "branch",
                    "length": 2,
                    "fields": [field],
                }
            ],
        },
    }


def test_branch_encoder_uses_gqa_kv_head_count():
    schema = Schema.model_validate(_payload(attention="gqa"))
    encoder = BranchEncoder(schema=schema, address="root")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 2


def test_branch_encoder_uses_mqa_kv_head_count():
    schema = Schema.model_validate(_payload(attention="mqa"))
    encoder = BranchEncoder(schema=schema, address="root")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 1


def test_branch_encoder_none_skips_transformer_layers():
    schema = Schema.model_validate(_payload(attention="none"))
    encoder = BranchEncoder(schema=schema, address="root")

    assert len(encoder.encoder) == 0


def test_branch_encoder_skips_coordinate_head_resolution_when_attention_is_disabled():
    schema = rf.Schema.from_tree(
        rf.Number("first"),
        rf.Number("second"),
        d_model=1,
        n_layers=1,
        n_heads=2,
        attention="none",
        reduction=rf.Mean(),
    )

    encoder = BranchEncoder(schema=schema, address="record")

    assert encoder.coordinate_encoder is None


def test_coordinate_encoder_uses_configured_attention_mode():
    for attention, expected_kv_heads in (("gqa", 2), ("mqa", 1)):
        schema = rf.Schema.from_tree(
            rf.Number("first"),
            rf.Number("second"),
            d_model=16,
            n_layers=1,
            n_heads=4,
            attention=attention,
            reduction=None,
        )

        encoder = BranchEncoder(schema=schema, address="record")

        assert encoder.coordinate_encoder is not None
        assert encoder.coordinate_encoder.attention.n_kv_heads == expected_kv_heads


def test_decoder_mean_pooling_repeats_heritage_mean_for_each_target_slot():
    schema = Schema.model_validate(_payload(pooling="mean"))
    model = Model(schema=schema, batch_size=2)
    decoder = model.nodes["root/items/category"].decoder
    parcel = Parcel(
        origin="root",
        destination="",
        payload=torch.randn(2, 3, 16),
        present=torch.ones(2, 3, dtype=torch.bool),
        batch_size=2,
    )

    prediction = decoder([parcel], batch_size=2, device=parcel.payload.device)

    assert isinstance(decoder.pool, MeanPool)
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 8)


def test_branch_encoder_propagates_presence_and_zeros_empty_rows():
    schema = Schema.model_validate(_payload())
    encoder = BranchEncoder(schema=schema, address="root")
    payload = torch.randn(2, 3, 16)
    parcel = Parcel(
        origin="root/items",
        destination="root",
        payload=payload,
        present=torch.tensor([[True, False, True], [False, False, False]]),
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 16)
    assert torch.equal(encoded.present, torch.tensor([True, False]))
    assert torch.equal(encoded.payload[1], torch.zeros_like(encoded.payload[1]))


def test_nested_branch_encoder_preserves_repeated_parent_geometry():
    schema = Schema.model_validate(
        {
            "d_model": 16,
            "fields": {
                "name": "root",
                "type": "branch",
                "fields": [
                    {
                        "name": "parents",
                        "type": "branch",
                        "length": 3,
                        "fields": [
                            {
                                "name": "children",
                                "type": "branch",
                                "length": 2,
                                "fields": [
                                    {
                                        "name": "value",
                                        "type": "category",
                                        "size": 8,
                                    }
                                ],
                            }
                        ],
                    }
                ],
            },
        }
    )
    encoder = BranchEncoder(schema=schema, address="root/parents/children")
    present = torch.tensor(
        [
            [[[True, False], [False, False], [True, True]]],
            [[[False, False], [True, False], [False, True]]],
        ]
    )
    parcel = Parcel(
        origin="root/parents/children/value",
        destination="root/parents/children",
        payload=torch.randn(2, 1, 3, 2, 16),
        present=present,
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 1, 3, 16)
    assert torch.equal(encoded.present, present.any(dim=-1))


def test_branch_encoder_routes_multiple_attention_outputs_to_its_parent():
    schema = Schema.model_validate(
        {
            "d_model": 16,
            "fields": {
                "name": "root",
                "type": "branch",
                "fields": [
                    {
                        "name": "parents",
                        "type": "branch",
                        "length": 3,
                        "fields": [
                            {
                                "name": "children",
                                "type": "branch",
                                "length": 2,
                                "reduction": {"type": "attention", "n_outputs": 2},
                                "fields": [{"name": "value", "type": "number"}],
                            }
                        ],
                    }
                ],
            },
        }
    )
    encoder = BranchEncoder(schema=schema, address="root/parents/children")
    parcel = Parcel(
        origin="root/parents/children/value",
        destination="root/parents/children",
        payload=torch.randn(2, 1, 3, 2, 16),
        present=torch.tensor(
            [
                [[[True, False], [False, False], [True, True]]],
                [[[False, False], [True, False], [False, True]]],
            ]
        ),
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 1, 6, 16)
    assert encoded.present.shape == (2, 1, 6)
    expected = parcel.present.any(dim=-1).unsqueeze(-1).expand(-1, -1, -1, 2).reshape(2, 1, 6)
    assert torch.equal(encoded.present, expected)


def test_branch_encoder_none_reduction_routes_every_encoded_token_and_presence():
    schema = rf.Schema.from_tree(
        rf.Branch(
            rf.Number("first"),
            rf.Number("second"),
            name="items",
            length=2,
            attention="none",
            reduction=None,
        ),
        d_model=4,
        n_layers=1,
        n_heads=2,
        attention="none",
    )
    encoder = BranchEncoder(schema=schema, address="record/items")
    first = Parcel(
        origin="record/items/first",
        destination="record/items",
        payload=torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4),
        present=torch.tensor([[[True, False]], [[True, True]]]),
        batch_size=2,
    )
    second = Parcel(
        origin="record/items/second",
        destination="record/items",
        payload=torch.arange(16, 32, dtype=torch.float32).reshape(2, 1, 2, 4),
        present=torch.tensor([[[False, True]], [[False, False]]]),
        batch_size=2,
    )

    # Runtime leaves and nested branches can arrive in a different order.
    # Encoding must follow the declared schema order, not arrival order.
    encoded = encoder([second, first])

    expected_payload = torch.cat([first.payload, second.payload], dim=-2)
    expected_present = torch.cat([first.present, second.present], dim=-1)
    expected_payload = expected_payload.masked_fill(~expected_present.unsqueeze(-1), 0.0)
    assert encoded.payload.shape == (2, 4, 4)
    assert torch.equal(encoded.payload, expected_payload.reshape(2, 4, 4))
    assert torch.equal(encoded.present, expected_present.reshape(2, 4))


def test_branch_encoder_orders_nested_and_leaf_parcels_by_schema() -> None:
    schema = rf.Schema.from_tree(
        rf.Branch(name="nested", length=2, attention="none", reduction=None, value=rf.Number),
        rf.Number("direct"),
        d_model=4,
        n_layers=1,
        n_heads=2,
        attention="none",
        reduction=None,
    )
    encoder = BranchEncoder(schema=schema, address="record")
    nested = Parcel(
        origin="record/nested",
        destination="record",
        payload=torch.tensor([[[1.0] * 4, [2.0] * 4]]),
        present=torch.ones((1, 2), dtype=torch.bool),
        batch_size=1,
    )
    direct = Parcel(
        origin="record/direct",
        destination="record",
        payload=torch.tensor([[[3.0] * 4]]),
        present=torch.ones((1, 1), dtype=torch.bool),
        batch_size=1,
    )

    encoded = encoder([direct, nested])

    assert torch.equal(encoded.payload, torch.cat((nested.payload, direct.payload), dim=1))


def test_branch_encoder_contextualizes_only_jointly_aligned_field_coordinates():
    schema = rf.Schema.from_tree(
        rf.Branch(
            rf.Number("first"),
            rf.Number("second"),
            name="items",
            length=3,
            reduction=None,
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    encoder = BranchEncoder(schema=schema, address="record/items").eval()
    first = Parcel(
        origin="record/items/first",
        destination="record/items",
        payload=torch.randn(2, 1, 3, 8),
        present=torch.ones(2, 1, 3, dtype=torch.bool),
        batch_size=2,
    )
    second = Parcel(
        origin="record/items/second",
        destination="record/items",
        payload=torch.randn(2, 1, 3, 8),
        present=torch.ones(2, 1, 3, dtype=torch.bool),
        batch_size=2,
    )
    order = torch.tensor([2, 0, 1])

    original = encoder.contextualize([first, second])
    permuted = encoder.contextualize(
        [
            Parcel(
                origin=first.origin,
                destination=first.destination,
                payload=first.payload[:, :, order],
                present=first.present[:, :, order],
                batch_size=2,
            ),
            Parcel(
                origin=second.origin,
                destination=second.destination,
                payload=second.payload[:, :, order],
                present=second.present[:, :, order],
                batch_size=2,
            ),
        ]
    )

    assert torch.allclose(permuted[0].payload, original[0].payload[:, :, order])
    assert torch.allclose(permuted[1].payload, original[1].payload[:, :, order])


def test_branch_encoder_mean_reduction_uses_one_output_and_ignores_padding():
    schema = rf.Schema.from_tree(
        rf.Branch(
            rf.Number("value"),
            name="items",
            length=3,
            attention="none",
            reduction=rf.Mean(),
        ),
        d_model=4,
        n_layers=1,
        n_heads=2,
        attention="none",
    )
    encoder = BranchEncoder(schema=schema, address="record/items")
    parcel = Parcel(
        origin="record/items/value",
        destination="record/items",
        payload=torch.tensor([[[[1.0] * 4, [100.0] * 4, [3.0] * 4]]]),
        present=torch.tensor([[[True, False, True]]]),
        batch_size=1,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (1, 1, 4)
    assert torch.equal(encoded.payload, torch.full((1, 1, 4), 2.0))
    assert torch.equal(encoded.present, torch.ones((1, 1), dtype=torch.bool))


def test_attention_reduction_can_disable_only_its_rotary_position() -> None:
    schema = rf.Schema.from_tree(
        rf.Branch(
            name="items",
            length=3,
            attention="none",
            reduction=rf.Attention(position=False),
            value=rf.Number,
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    encoder = BranchEncoder(schema=schema, address="record/items")

    assert encoder.pool.blocks[0].attention.rotary is None


def test_decoder_receives_canonical_heritage_and_separate_sibling_context():
    model = rf.Model(
        items=rf.Branch(
            length=3,
            reduction=None,
            value=rf.Number,
        ),
        selector=rf.Number,
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
        reduction=rf.Attention(n_outputs=2),
    )
    inputs = model.encode(
        pa.Table.from_pylist(
            [
                {"items": [{"value": 1.0}, {"value": 2.0}], "selector": 1.0, "answer": 2.0},
                {"items": [{"value": 3.0}], "selector": 2.0, "answer": 3.0},
            ]
        ),
        strata=Strata.train,
    )

    decoder = model.nodes["record/answer"].decoder
    with patch.object(decoder, "forward", wraps=decoder.forward) as forward:
        predictions = model(inputs, strata=Strata.train)
    answer = next(prediction for prediction in predictions if prediction.address == "record/answer")

    parcels = forward.call_args.args[0]
    contexts = forward.call_args.kwargs["contexts"]
    assert [parcel.origin for parcel in parcels] == ["record", "record/answer"]
    assert not parcels[1].present.any()
    assert [parcel.origin for parcel in contexts] == [
        "record/items",
        "record/items/value",
        "record/selector",
    ]
    assert answer.payload[TensorKey.content].shape == (2, 1, 1)
    assert all(torch.isfinite(value).all() for value in answer.payload.values())


def test_decoder_supports_zero_context():
    schema = Schema.model_validate(_payload(pooling="mean"))
    model = Model(schema=schema, batch_size=2)
    decoder = model.nodes["root/items/category"].decoder

    prediction = decoder([], batch_size=2, device=torch.device("cpu"))

    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, 8)
    assert all(torch.isfinite(value).all() for value in prediction.payload.values())


def test_decoder_sibling_context_descends_only_through_unreduced_branches():
    preserved = rf.Model(
        items=rf.Branch(length=3, reduction=None, value=rf.Number),
        selector=rf.Number,
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    compressed = rf.Model(
        items=rf.Branch(length=3, reduction=rf.Mean(), value=rf.Number),
        selector=rf.Number,
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    preserved_decoder = preserved.nodes["record/answer"].decoder
    compressed_decoder = compressed.nodes["record/answer"].decoder

    assert "record/items" in preserved_decoder.context_addresses
    assert "record/items/value" in preserved_decoder.context_addresses
    assert "record/selector" in preserved_decoder.context_addresses
    assert "record/items" in compressed_decoder.context_addresses
    assert "record/items/value" not in compressed_decoder.context_addresses


def test_decoder_does_not_build_query_projection_for_only_unaligned_context() -> None:
    model = rf.Model(
        items=rf.Branch(length=3, reduction=None, value=rf.Number),
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    decoder = model.nodes["record/answer"].decoder

    assert decoder.context_addresses == ("record/items", "record/items/value")
    assert decoder.context_projection is None


def test_all_absent_heritage_retains_masked_sibling_context_in_autograd() -> None:
    model = rf.Model(
        hidden=rf.Number(mask=True),
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    batch = model.encode(
        pa.Table.from_pylist(
            [
                {"hidden": 1.0, "answer": 2.0},
                {"hidden": 2.0, "answer": 3.0},
            ]
        ),
        strata=Strata.train,
    )

    output = model.training_step(batch, batch_idx=0)
    assert output is not None
    output["loss"].backward()

    for address in ("record/hidden", "record/answer"):
        decoder = model.nodes[address].decoder
        projection = decoder.context_projection
        assert projection is not None
        assert all(parameter.grad is not None for parameter in projection.parameters())


def test_scalar_decoder_position_defaults_off_and_can_be_enabled() -> None:
    automatic = rf.Model(
        value=rf.Number,
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    ordered = rf.Model(
        value=rf.Number,
        answer=rf.Number(mask=True, decoder_position=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )

    automatic_pool = automatic.nodes["record/answer"].decoder.pool
    ordered_pool = ordered.nodes["record/answer"].decoder.pool
    assert automatic_pool.blocks[0].attention.rotary is None
    assert ordered_pool.blocks[0].attention.rotary is not None


def test_repeated_target_uses_coordinate_conditioned_queries():
    model = rf.Model(
        items=rf.Branch(
            length=3,
            reduction=None,
            entity_id=rf.Hash(n_hashes=2),
            value=rf.Number(mask=True),
        ),
        d_model=8,
        n_layers=1,
        n_heads=2,
        reduction=None,
    )
    decoder = model.nodes["record/items/value"].decoder

    assert decoder.n_context == 3
    assert decoder.context_addresses == ("record/items/entity_id",)
    assert decoder.pool.blocks[0].attention.rotary is not None


def test_decoder_ignores_inactive_sibling_branches() -> None:
    model = rf.Model(
        inactive=rf.Branch(length=3, reduction=None, value=rf.Number(active=False)),
        visible=rf.Number,
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    decoder = model.nodes["record/answer"].decoder

    assert decoder.context_addresses == ("record/visible",)
