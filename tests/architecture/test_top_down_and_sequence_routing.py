import pyarrow as pa
import torch

import relflow as rf
from relflow.structs.packages import Parcel


def model(**options):
    return rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        seg=rf.Category,
        items=rf.Branch(
            length=4, v=rf.Number, c=rf.Number, target=rf.Number(mask=True), sub=rf.Branch(length=3, z=rf.Number)
        ),
        y=rf.Number(mask=True),
        **options,
    ).eval()


TABLE = pa.table(
    {
        "seg": ["a", "b"],
        "items": [
            [{"v": 1.0, "c": 2.0, "sub": [{"z": 1.0}, {"z": 2.0}]}, {"v": 3.0, "c": 4.0, "sub": []}],
            [{"v": 5.0, "c": 6.0, "sub": [{"z": 3.0}]}],
        ],
        "y": [1.0, 2.0],
    }
)


def test_child_branches_condition_on_their_parents_embedded_fields_only():
    configured = model()
    root, items, sub = (configured.nodes[address].encoder for address in ("/", "/items", "/items/sub"))
    assert root.conditioning is None and root.context_origins == frozenset()
    assert items.conditioning is not None and items.context_origins == {rf.Address("/seg")}
    # The hidden target is never embedded, so it cannot leak into its own branch's children.
    assert sub.context_origins == {rf.Address("/items/v"), rf.Address("/items/c")}
    assert items.conditioning.weight.abs().sum() > 0 and items.conditioning.bias is None


def test_conditioning_reaches_every_token_and_the_reduction_queries():
    configured = model()
    encoded = configured.encode(TABLE, strata=rf.Strata.predict)
    items = configured.nodes["/items"].encoder
    outputs = []
    items.register_forward_hook(lambda module, args, output: outputs.append(output))
    with torch.no_grad():
        assert items.conditioning is not None
        items.conditioning.weight.zero_()
        configured(encoded, strata=rf.Strata.predict)
        items.conditioning.weight.fill_(0.5)
        configured(encoded, strata=rf.Strata.predict)
    before, after = outputs
    assert not torch.allclose(before.payload, after.payload)
    assert before.sequence is not None and after.sequence is not None
    assert not torch.allclose(before.sequence.payload, after.sequence.payload)


def test_condition_projects_the_masked_mean_of_the_parent_fields():
    configured = model()
    items = configured.nodes["/items"].encoder
    assert items.conditioning is not None
    with torch.no_grad():
        items.conditioning.weight.copy_(torch.eye(8))
    present = torch.tensor([[[True]], [[False]]])
    parent = Parcel(
        payload=torch.arange(16, dtype=torch.float32).reshape(2, 1, 1, 8),
        present=present,
        origin=rf.Address("/seg"),
        destination=rf.Address(),
        batch_size=2,
    )
    other = Parcel(
        payload=torch.full((2, 1, 1, 8), 10.0),
        present=torch.ones((2, 1, 1), dtype=torch.bool),
        origin=rf.Address("/items/v"),
        destination=rf.Address("/items"),
        batch_size=2,
    )
    # Only the parent's own fields count, averaged over the present ones; an absent field adds nothing.
    conditioned = items.condition([parent, other], rows=2)
    torch.testing.assert_close(conditioned[0], torch.arange(8, dtype=torch.float32))
    torch.testing.assert_close(conditioned[1], torch.zeros(8))


def test_reduced_branches_carry_their_encoded_sequence_and_pass_through_ones_do_not():
    configured = model()
    encoded = configured.encode(TABLE, strata=rf.Strata.predict)
    routed = {}
    for address in ("/items", "/items/sub"):
        configured.nodes[address].encoder.register_forward_hook(
            lambda module, args, output, key=address: routed.__setitem__(key, output)
        )
    with torch.no_grad():
        configured(encoded, strata=rf.Strata.predict)
    items = routed["/items"]
    assert tuple(items.payload.shape) == (2, 1, 8) and items.sequence is not None
    # Two retained items each own four tokens: v, c, the hidden target's slot, and the sub summary.
    assert tuple(items.sequence.payload.shape) == (2, 8, 8)
    # Field-major storage: v, c, the hidden target, then the sub summary, which an empty collection lacks.
    assert items.sequence.present.tolist() == [
        [True, True, True, True, False, False, True, False],
        [True, False, True, False, False, False, True, False],
    ]
    # Each root item owns the branch's whole capacity of 16 positions, whatever the batch width.
    assert items.sequence.widths is not None and items.sequence.widths.tolist() == [[16], [16]]
    assert items.sequence.spans.tolist() == [16, 16] and items.sequence.sequence is None
    passthrough = rf.Model(
        d_model=8, n_layers=1, n_heads=2, events=rf.Branch(reduction=None, amount=rf.Number, y=rf.Number(mask=True))
    ).eval()
    captured = []
    passthrough.nodes["/events"].encoder.register_forward_hook(lambda module, args, output: captured.append(output))
    source = pa.table({"events": [[{"amount": 1.0}, {"amount": 2.0}]]})
    with torch.no_grad():
        passthrough(passthrough.encode(source, strata=rf.Strata.predict), strata=rf.Strata.predict)
    assert captured[0].sequence is None


def test_decoders_read_the_encoded_sequences_of_their_ancestors():
    configured = model()
    encoded = configured.encode(TABLE, strata=rf.Strata.predict)
    memory = {}
    for address in ("/items/target", "/y"):
        configured.nodes[address].decoder.pool.register_forward_pre_hook(
            lambda module, args, kwargs, key=address: memory.__setitem__(
                key, (tuple(args[0].shape), kwargs["memory_positions"].tolist())
            ),
            with_kwargs=True,
        )
    with torch.no_grad():
        configured(encoded, strata=rf.Strata.predict)
    # The root target sees the root summary, the root's three tokens (seg, items, y), and its own leaf.
    assert memory["/y"] == ((2, 5, 8), [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]])
    # The item target adds the items summary, the items sequence item-major, and its own two slots
    # offset by the items branch's capacity of 16 positions.
    shape, positions = memory["/items/target"]
    assert shape == (2, 15, 8)
    assert positions[0] == [0, 1, 2, 3, 4, 5, 9, 6, 10, 7, 11, 8, 12, 21, 22]
    decoder = configured.nodes["/items/target"].decoder
    assert decoder.memory_coordinates[rf.Address("/items")][2] == 1
    assert decoder.sequence_coordinates[rf.Address("/items")][2] == 16
    assert decoder.memory_coordinates[rf.Address()][2] == 1 and decoder.sequence_coordinates[rf.Address()][2] == 3
    assert configured.schema.branch_widths == {rf.Address(): 3, rf.Address("/items"): 16, rf.Address("/items/sub"): 3}


def test_branches_without_a_decoder_beneath_them_skip_their_sequence():
    configured = model()
    captured: dict[str, Parcel] = {}
    for address in ("/items", "/items/sub"):
        configured.nodes[address].encoder.register_forward_hook(
            lambda module, args, output, address=address: captured.__setitem__(address, output)
        )
    with torch.no_grad():
        configured(configured.encode(TABLE, strata=rf.Strata.predict), strata=rf.Strata.predict)
    # The item target reads the items sequence; nothing decodes beneath sub, so its nested sequence,
    # the widest tensor in the tree, is never materialized.
    assert captured["/items"].sequence is not None and captured["/items/sub"].sequence is None

    decoded = rf.Model(
        d_model=8,
        n_layers=1,
        n_heads=2,
        items=rf.Branch(length=4, v=rf.Number, sub=rf.Branch(length=3, z=rf.Number, t=rf.Number(mask=True))),
    ).eval()
    captured.clear()
    for address in ("/items", "/items/sub"):
        decoded.nodes[address].encoder.register_forward_hook(
            lambda module, args, output, address=address: captured.__setitem__(address, output)
        )
    table = pa.table({"items": [[{"v": 1.0, "sub": [{"z": 1.0, "t": 2.0}]}], [{"v": 2.0, "sub": []}]]})
    with torch.no_grad():
        decoded(decoded.encode(table, strata=rf.Strata.predict), strata=rf.Strata.predict)
    assert captured["/items"].sequence is not None and captured["/items/sub"].sequence is not None
