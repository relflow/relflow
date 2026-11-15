from unittest.mock import patch

import pyarrow as pa
import pytest
import torch
from tensordict import TensorDict

import relflow as rf
from relflow.architecture.encoder import BranchEncoder, stepwise, timeline
from relflow.architecture.pool import MeanPool
from relflow.architecture.root import Model
from relflow.data.arrow import Geometry
from relflow.structs.enums import Strata, TensorKey, Tokens
from relflow.structs.packages import Parcel


def configuration(*, attention: str | None = "mha", pooling: str = "query") -> rf.Schema:
    return rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        attention=attention,
        dropout=0.0,
        items=rf.Branch(length=2, category=rf.Category(mask=True, pooling=pooling)),
    )


def test_branch_encoder_uses_gqa_kv_head_count():
    schema = configuration(attention="gqa")
    encoder = BranchEncoder(schema=schema, address="/")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 2


def test_branch_encoder_uses_mqa_kv_head_count():
    schema = configuration(attention="mqa")
    encoder = BranchEncoder(schema=schema, address="/")

    assert len(encoder.encoder) == 1
    assert encoder.encoder[0].attention.n_kv_heads == 1


def test_branch_encoder_none_skips_transformer_layers():
    schema = configuration(attention=None)
    encoder = BranchEncoder(schema=schema, address="/")

    assert len(encoder.encoder) == 0
    assert encoder.coordinate_encoder is None


def test_branch_encoder_skips_coordinate_head_resolution_when_attention_is_disabled():
    schema = rf.Schema.from_tree(
        first=rf.Number, second=rf.Number, d_model=1, n_layers=1, n_heads=2, attention=None, reduction=rf.Mean()
    )

    encoder = BranchEncoder(schema=schema, address="/")

    assert encoder.coordinate_encoder is None


def test_coordinate_encoder_uses_configured_attention_mode():
    for attention, expected_kv_heads in (("gqa", 2), ("mqa", 1)):
        schema = rf.Schema.from_tree(
            first=rf.Number,
            second=rf.Number,
            d_model=16,
            n_layers=1,
            n_heads=4,
            attention=attention,
            reduction=None,
        )

        encoder = BranchEncoder(schema=schema, address="/")

        assert encoder.coordinate_encoder is not None
        assert encoder.coordinate_encoder.attention.n_kv_heads == expected_kv_heads


def test_decoder_mean_pooling_repeats_heritage_mean_for_each_target_slot():
    schema = configuration(pooling="mean")
    model = Model(schema=schema, batch_size=2)
    decoder = model.nodes["/items/category"].decoder
    parcel = Parcel(
        origin="/",
        destination="",
        payload=torch.randn(2, 3, 16),
        present=torch.ones(2, 3, dtype=torch.bool),
        batch_size=2,
    )

    prediction = decoder([parcel], batch_size=2, device=parcel.payload.device)

    assert isinstance(decoder.pool, MeanPool)
    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, decoder.linears["content"].out_features)


def test_branch_encoder_propagates_presence_and_zeros_empty_rows():
    schema = configuration()
    encoder = BranchEncoder(schema=schema, address="/")
    payload = torch.randn(2, 3, 16)
    parcel = Parcel(
        origin="/items",
        destination="/",
        payload=payload,
        present=torch.tensor([[True, False, True], [False, False, False]]),
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 16)
    assert torch.equal(encoded.present, torch.tensor([True, False]))
    assert torch.equal(encoded.payload[1], torch.zeros_like(encoded.payload[1]))


def test_nested_branch_encoder_preserves_repeated_parent_geometry():
    schema = rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        parents=rf.Branch(length=3, children=rf.Branch(length=2, value=rf.Category)),
    )
    encoder = BranchEncoder(schema=schema, address="/parents/children")
    present = torch.tensor(
        [
            [[[True, False], [False, False], [True, True]]],
            [[[False, False], [True, False], [False, True]]],
        ]
    )
    parcel = Parcel(
        origin="/parents/children/value",
        destination="/parents/children",
        payload=torch.randn(2, 1, 3, 2, 16),
        present=present,
        batch_size=2,
    )

    encoded = encoder([parcel])

    assert encoded.payload.shape == (2, 1, 3, 16)
    assert torch.equal(encoded.present, present.any(dim=-1))


def test_branch_encoder_routes_multiple_attention_outputs_to_its_parent():
    schema = rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        parents=rf.Branch(
            length=3,
            children=rf.Branch(length=2, reduction=rf.Attention(n_outputs=2), value=rf.Number),
        ),
    )
    encoder = BranchEncoder(schema=schema, address="/parents/children")
    parcel = Parcel(
        origin="/parents/children/value",
        destination="/parents/children",
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
        items=rf.Branch(first=rf.Number, second=rf.Number, length=2, attention=None, reduction=None),
        d_model=4,
        n_layers=1,
        n_heads=2,
        attention=None,
    )
    encoder = BranchEncoder(schema=schema, address="/items")
    first = Parcel(
        origin="/items/first",
        destination="/items",
        payload=torch.arange(16, dtype=torch.float32).reshape(2, 1, 2, 4),
        present=torch.tensor([[[True, False]], [[True, True]]]),
        batch_size=2,
    )
    second = Parcel(
        origin="/items/second",
        destination="/items",
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
        nested=rf.Branch(length=2, attention=None, reduction=None, value=rf.Number),
        direct=rf.Number,
        d_model=4,
        n_layers=1,
        n_heads=2,
        attention=None,
        reduction=None,
    )
    encoder = BranchEncoder(schema=schema, address="/")
    nested = Parcel(
        origin="/nested",
        destination="/",
        payload=torch.tensor([[[1.0] * 4, [2.0] * 4]]),
        present=torch.ones((1, 2), dtype=torch.bool),
        batch_size=1,
    )
    direct = Parcel(
        origin="/direct",
        destination="/",
        payload=torch.tensor([[[3.0] * 4]]),
        present=torch.ones((1, 1), dtype=torch.bool),
        batch_size=1,
    )

    encoded = encoder([direct, nested])

    assert torch.equal(encoded.payload, torch.cat((nested.payload, direct.payload), dim=1))


def test_branch_encoder_contextualizes_only_jointly_aligned_field_coordinates():
    schema = rf.Schema.from_tree(
        items=rf.Branch(first=rf.Number, second=rf.Number, length=3, reduction=None),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    encoder = BranchEncoder(schema=schema, address="/items").eval()
    first = Parcel(
        origin="/items/first",
        destination="/items",
        payload=torch.randn(2, 1, 3, 8),
        present=torch.ones(2, 1, 3, dtype=torch.bool),
        batch_size=2,
    )
    second = Parcel(
        origin="/items/second",
        destination="/items",
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
        items=rf.Branch(value=rf.Number, length=3, attention=None, reduction=rf.Mean()),
        d_model=4,
        n_layers=1,
        n_heads=2,
        attention=None,
    )
    encoder = BranchEncoder(schema=schema, address="/items")
    parcel = Parcel(
        origin="/items/value",
        destination="/items",
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
        items=rf.Branch(length=3, attention=None, reduction=rf.Attention(position=False), value=rf.Number),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    encoder = BranchEncoder(schema=schema, address="/items")

    assert encoder.coordinate_encoder is None
    assert len(encoder.encoder) == 0
    assert len(encoder.pool.blocks) == 1
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

    decoder = model.nodes["/answer"].decoder
    with patch.object(decoder, "forward", wraps=decoder.forward) as forward:
        predictions = model(inputs, strata=Strata.train)
    answer = next(prediction for prediction in predictions if prediction.address == "/answer")

    parcels = forward.call_args.args[0]
    contexts = forward.call_args.kwargs["contexts"]
    # The root arrives as its two-query summary and as its encoded sequence, which stays at the root.
    assert [parcel.origin for parcel in parcels] == ["/", "/", "/answer"]
    assert parcels[0].destination != parcels[0].origin and parcels[1].destination == parcels[1].origin
    assert tuple(parcels[0].payload.shape[1:-1]) == (2,) and tuple(parcels[1].payload.shape[1:-1]) == (4,)
    assert not parcels[2].present.any()
    assert [parcel.origin for parcel in contexts] == [
        "/items",
        "/items/value",
        "/selector",
    ]
    assert answer.payload[TensorKey.content].shape == (2, 1, 1)
    assert all(torch.isfinite(value).all() for value in answer.payload.values())


def test_decoder_supports_zero_context():
    schema = configuration(pooling="mean")
    model = Model(schema=schema, batch_size=2)
    decoder = model.nodes["/items/category"].decoder

    prediction = decoder([], batch_size=2, device=torch.device("cpu"))

    assert prediction.payload[TensorKey.state].shape == (2, 2, len(Tokens))
    assert prediction.payload[TensorKey.content].shape == (2, 2, decoder.linears["content"].out_features)
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

    preserved_decoder = preserved.nodes["/answer"].decoder
    compressed_decoder = compressed.nodes["/answer"].decoder

    assert "/items" in preserved_decoder.context_addresses
    assert "/items/value" in preserved_decoder.context_addresses
    assert "/selector" in preserved_decoder.context_addresses
    assert "/items" in compressed_decoder.context_addresses
    assert "/items/value" not in compressed_decoder.context_addresses


def test_decoder_does_not_build_query_projection_for_only_unaligned_context() -> None:
    model = rf.Model(
        items=rf.Branch(length=3, reduction=None, value=rf.Number),
        answer=rf.Number(mask=True),
        d_model=8,
        n_layers=1,
        n_heads=2,
    )
    decoder = model.nodes["/answer"].decoder

    assert decoder.context_addresses == ("/items", "/items/value")
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

    for address in ("/hidden", "/answer"):
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

    automatic_pool = automatic.nodes["/answer"].decoder.pool
    ordered_pool = ordered.nodes["/answer"].decoder.pool
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
    decoder = model.nodes["/items/value"].decoder

    assert decoder.n_context == 3
    assert decoder.context_addresses == ("/items/entity_id",)
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
    decoder = model.nodes["/answer"].decoder

    assert decoder.context_addresses == ("/visible",)


def test_admit_orders_parcels_by_field_order_and_rejects_strangers():
    encoder = BranchEncoder(schema=configuration(), address=rf.Address("items"))
    category = Parcel(
        payload=torch.zeros(1, 2, 16),
        present=torch.ones(1, 2, dtype=torch.bool),
        origin=rf.Address("items", "category"),
        destination=rf.Address("items"),
        batch_size=1,
    )
    assert encoder.admit([category]) == [category]
    stranger = Parcel(
        payload=torch.zeros(1, 2, 16),
        present=torch.ones(1, 2, dtype=torch.bool),
        origin=rf.Address("other"),
        destination=rf.Address("items"),
        batch_size=1,
    )
    with pytest.raises(ValueError, match="non-child address"):
        encoder.admit([category, stranger])


def test_mix_without_sequence_layers_returns_its_inputs():
    encoder = BranchEncoder(schema=configuration(attention=None), address="/")
    inputs = torch.randn(2, 3, 16, generator=torch.Generator().manual_seed(0))
    present = torch.tensor([[True, True, False], [True, True, True]])
    torch.testing.assert_close(encoder.mix(inputs, present=present, positions=None), inputs)


def test_reduce_without_reduction_returns_the_sequence():
    schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, reduction=None, value=rf.Number)
    encoder = BranchEncoder(schema=schema, address="/")
    selected = torch.randn(2, 1, 16, generator=torch.Generator().manual_seed(0))
    present = torch.ones(2, 1, dtype=torch.bool)
    output, output_present = encoder.reduce(
        selected, present=present, evidence=selected, context=None, positions=None, width=1
    )
    assert output is selected and output_present is present


def test_stepwise_numbers_tokens_by_row_in_sequence_order():
    def token(name, count):
        return Parcel(
            origin=name,
            destination="/",
            payload=torch.zeros(1, count, 4),
            present=torch.ones(1, count, dtype=torch.bool),
            batch_size=1,
        )

    parcels = [token("/a", 3), token("/b", 3), token("/c", 6)]
    assert stepwise(parcels).tolist() == [[0, 1, 2, 0, 1, 2, 0, 0, 1, 1, 2, 2]]


def test_plain_branch_calls_compute_without_steps():
    schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, reduction=None, value=rf.Number)
    encoder = BranchEncoder(schema=schema, address="/")
    calls = []

    def compute(inputs, present, **keywords):
        calls.append(keywords)
        return inputs

    encoder.compute = compute
    payload = torch.randn(2, 3, 16)
    encoder(
        [
            Parcel(
                origin="/value",
                destination="/",
                payload=payload,
                present=torch.ones(2, 3, dtype=torch.bool),
                batch_size=2,
            )
        ]
    )
    assert calls == [{}]


def test_timeline_without_times_returns_structural_positions_unchanged():
    structural = torch.tensor([[0, 1, 2, 3]])
    assert timeline(structural=structural, steps=torch.tensor([[0, 0, 1, 1]]), times=None) is structural


def test_timeline_gives_all_tokens_of_a_row_its_time():
    result = timeline(
        structural=torch.tensor([[0, 1, 2, 3]]),
        steps=torch.tensor([[0, 0, 1, 1]]),
        times=torch.tensor([[0.0, 2.5]], dtype=torch.float64),
    )
    assert result.tolist() == [[0.0, 0.0, 2.5, 2.5]]


def test_timeline_uses_zero_for_padding_steps_outside_time_geometry():
    result = timeline(
        structural=torch.tensor([[0, 1, 2]]),
        steps=torch.tensor([[0, 1, 2]]),
        times=torch.tensor([[4.0, 7.0]], dtype=torch.float64),
    )
    assert result.tolist() == [[4.0, 7.0, 0.0]]


def test_contrastive_scatter_accepts_mixed_precision_mixer_output():
    class BFloatMixer(torch.nn.Module):
        def forward(self, inputs: torch.Tensor, **kwargs: object) -> torch.Tensor:
            return inputs.to(torch.bfloat16)

    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        ett=rf.Branch(length=4, x=rf.Number, contrastive=rf.Contrastive(views=rf.Crops(width=2))),
    )
    encoder = BranchEncoder(schema=schema, address="/ett")
    encoder.encoder = BFloatMixer()
    tokens = torch.randn(2, 4, 8)
    present = torch.ones(2, 4, dtype=torch.bool)
    steps = torch.arange(4).expand(2, -1)
    shares = encoder.contrast(
        tokens,
        present=present,
        positions=None,
        steps=steps,
        views=torch.tensor([[0, 1], [0, 1]]),
        extent=4,
    )
    assert shares is not None and shares.dtype == tokens.dtype and torch.isfinite(shares).all()


def test_contrastive_credit_accepts_mixed_precision_shares():
    schema = rf.Schema.from_tree(
        d_model=8,
        n_layers=1,
        n_heads=2,
        ett=rf.Branch(length=4, x=rf.Number, contrastive=rf.Contrastive(views=rf.Crops(width=2))),
    )
    encoder = BranchEncoder(schema=schema, address="/ett")
    shares = encoder.credit(
        torch.ones(1, dtype=torch.bfloat16),
        indices=torch.tensor([1]),
        rows=2,
        like=torch.zeros(2),
        views=True,
    )
    assert shares is not None and shares.dtype == torch.float32 and shares.tolist() == [0.0, 1.0]


def timed_run(times, *, geometry=True):
    torch.manual_seed(0)
    schema = rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        ett=rf.Branch(length=3, time_index="ts", reduction=None, ts=rf.Number, value=rf.Number),
    )
    encoder = BranchEncoder(schema=schema, address="/ett").eval()
    generator = torch.Generator().manual_seed(1)

    def parcel(name):
        return Parcel(
            origin=f"/ett/{name}",
            destination="/ett",
            payload=torch.randn(1, 1, 3, 16, generator=generator),
            present=torch.ones(1, 1, 3, dtype=torch.bool),
            batch_size=1,
        )

    root = Geometry(
        lengths=torch.ones(1, dtype=torch.int64),
        present=torch.ones(1, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        slots=torch.ones(1, 1, dtype=torch.bool),
        batch_size=[1],
    )
    branch = Geometry(
        lengths=torch.full((1, 1), 3),
        present=torch.ones(1, 1, dtype=torch.bool),
        valid=torch.ones(1, 1, dtype=torch.bool),
        slots=torch.ones(1, 1, 3, dtype=torch.bool),
        times=torch.tensor([[times]], dtype=torch.float64),
        batch_size=[1],
    )
    inputs = TensorDict({"/": root, "/ett": branch}, batch_size=[1]) if geometry else None
    return encoder([parcel("ts"), parcel("value")], geometry=inputs)


def test_time_spacing_changes_encoding_but_not_routed_positions():
    even, uneven = timed_run([0.0, 1.0, 2.0]), timed_run([0.0, 5.0, 6.0])
    assert not torch.allclose(even.payload, uneven.payload)
    assert torch.equal(even.positions, uneven.positions) and even.positions.dtype == torch.int64


def test_timed_branch_without_geometry_times_raises():
    with pytest.raises(ValueError, match="branch '/ett' sets time_index='ts' but the batch geometry has no times"):
        timed_run([0.0, 1.0, 2.0], geometry=False)


def test_plain_branch_builds_no_gap_embedding_and_keeps_its_state_dict():
    schema = rf.Schema.from_tree(d_model=16, n_layers=1, n_heads=4, ett=rf.Branch(length=3, value=rf.Number))
    encoder = BranchEncoder(schema=schema, address="/ett")
    assert encoder.elapsed is None
    assert not any(key.startswith("elapsed") for key in encoder.state_dict())


def mixed_input(times, *, grain, present=None, elapsed=True):
    """Run a timed branch and return what its mixer received, with the gap embedding on or off."""
    torch.manual_seed(0)
    length = len(times)
    schema = rf.Schema.from_tree(
        d_model=16,
        n_layers=1,
        n_heads=4,
        ett=rf.Branch(
            length=length,
            time_index="ts",
            grain=grain,
            mixer=rf.Conv(n_blocks=1) if isinstance(grain, rf.Steps) else rf.Transformer(),
            direction=rf.Direction.causal if isinstance(grain, rf.Steps) else rf.Direction.bidirectional,
            reduction=None,
            ts=rf.Number,
            value=rf.Number,
            other=rf.Number,
        ),
    )
    encoder = BranchEncoder(schema=schema, address="/ett").eval()
    if not elapsed:
        encoder.elapsed = None
    seen = []
    encoder.encoder.register_forward_pre_hook(lambda module, args, kwargs: seen.append(args[0]), with_kwargs=True)
    generator = torch.Generator().manual_seed(1)
    flags = torch.ones(length, dtype=torch.bool) if present is None else present

    def parcel(name):
        return Parcel(
            origin=f"/ett/{name}",
            destination="/ett",
            payload=torch.randn(1, 1, length, 16, generator=generator),
            present=flags.clone().reshape(1, 1, length),
            positions=torch.arange(length).expand(1, 1, length),
            spans=torch.full((1, 1), length),
            slots=torch.ones(1, 1, length, dtype=torch.bool),
            batch_size=1,
        )

    root = Geometry(
        lengths=torch.ones(1, dtype=torch.int64),
        present=torch.ones(1, dtype=torch.bool),
        valid=torch.ones(1, dtype=torch.bool),
        slots=torch.ones(1, 1, dtype=torch.bool),
        batch_size=[1],
    )
    branch = Geometry(
        lengths=torch.full((1, 1), length),
        present=torch.ones(1, 1, dtype=torch.bool),
        valid=torch.ones(1, 1, dtype=torch.bool),
        slots=torch.ones(1, 1, length, dtype=torch.bool),
        times=torch.tensor([[times]], dtype=torch.float64),
        batch_size=[1],
    )
    inputs = TensorDict({"/": root, "/ett": branch}, batch_size=[1])
    encoder([parcel("ts"), parcel("value"), parcel("other")][:3], geometry=inputs)
    return encoder, seen[0]


def test_conv_output_changes_when_only_gaps_change():
    even, uneven = (mixed_input(times, grain=rf.Steps())[1] for times in ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 5.0, 6.0]))
    assert not torch.allclose(even, uneven)
    plain = [
        mixed_input(times, grain=rf.Steps(), elapsed=False)[1] for times in ([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 5.0, 6.0])
    ]
    torch.testing.assert_close(plain[0], plain[1], rtol=0, atol=0)


def test_regular_gaps_add_one_shared_vector_after_the_first_row():
    encoder, with_gaps = mixed_input([0.0, 1.0, 2.0, 3.0], grain=rf.Steps())
    without = mixed_input([0.0, 1.0, 2.0, 3.0], grain=rf.Steps(), elapsed=False)[1]
    offset = with_gaps - without
    first, regular = encoder.elapsed(torch.tensor([0.0, 1.0], dtype=torch.float64))
    torch.testing.assert_close(offset[0, 0], first)
    for row in (1, 2, 3):
        torch.testing.assert_close(offset[0, row], regular)


def test_coordinates_grain_gives_every_token_of_a_row_its_gap():
    times = [0.0, 1.0, 4.0]
    encoder, with_gaps = mixed_input(times, grain=rf.Coordinates())
    without = mixed_input(times, grain=rf.Coordinates(), elapsed=False)[1]
    # Storage is field-major: three fields of three rows each.
    tokens = (with_gaps - without).reshape(3, 3, 16)
    expected = encoder.elapsed(torch.tensor([0.0, 1.0, 3.0], dtype=torch.float64))
    for field in range(3):
        torch.testing.assert_close(tokens[field], expected)


def test_padding_rows_stay_zero_after_the_gap_embedding():
    present = torch.tensor([True, False, True, False])
    _, seen = mixed_input([0.0, 1.0, 2.0, 3.0], grain=rf.Steps(), present=present)
    assert not seen[0, 1].any() and not seen[0, 3].any()
    assert seen[0, 0].any() and seen[0, 2].any()


def test_contrastive_keeps_the_full_record_for_parent_and_decoders():
    def build(contrastive):
        return rf.Model.xs(
            d_model=8,
            dropout=0.0,
            label=rf.Boolean(mask=True),
            ett=rf.Branch.chronological(
                length=8, reduction=rf.Mean(), x=rf.Number, y=rf.Number(mask=0.3), contrastive=contrastive
            ),
        )

    with_views = build(rf.Contrastive(views=rf.Crops(width=4)))
    plain = build(None)
    assert with_views.state_dict().keys() == plain.state_dict().keys()
    plain.load_state_dict(with_views.state_dict())
    table = pa.Table.from_pylist([{"label": True, "ett": [{"x": float(i), "y": float(-i)} for i in range(8)]}])
    left = {
        p.address: p
        for p in with_views(with_views.encode(table, strata="train", seed=3), strata=Strata.train)
        if TensorKey.objective not in p.payload.keys()
    }
    right = {p.address: p for p in plain(plain.encode(table, strata="train", seed=3), strata=Strata.train)}
    assert left.keys() == right.keys()
    for address in left:
        for key, value in right[address].payload.items(include_nested=True, leaves_only=True):
            torch.testing.assert_close(left[address].payload[key], value, rtol=0, atol=0)


@pytest.mark.parametrize(
    "options",
    [{}, {"direction": rf.Direction.causal}, {"grain": rf.Steps(), "inner": rf.Branch(length=3, z=rf.Number)}],
    ids=["default", "causal", "steps"],
)
def test_time_index_survives_variable_width_unreduced_child(options):
    torch.manual_seed(0)
    model = rf.Model.xs(
        d_model=8,
        embed=True,
        events=rf.Branch(
            length=4,
            time_index="t",
            t=rf.Number,
            x=rf.Number,
            **{"inner": rf.Branch(length=3, reduction=None, z=rf.Number)} | options,
        ),
    )
    row = {"events": [{"t": 0.0, "x": 1.0, "inner": [{"z": 1.0}]}, {"t": 1.0, "x": 2.0, "inner": []}]}
    other = {"events": [{"t": 0.0, "x": 1.0, "inner": [{"z": 1.0}, {"z": 2.0}, {"z": 3.0}]}]}
    assert len(model.predict([row, other])) == 2
