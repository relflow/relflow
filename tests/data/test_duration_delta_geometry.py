"""Duration deltas preserve collection geometry, boundaries, and source order."""

import numpy as np
import pyarrow as pa
import pytest
import torch

import relflow as rf
from relflow.data.ragged import coalesce
from relflow.structs.enums import Tokens

P, N, V = Tokens.padded, Tokens.null, Tokens.valued
ITEM = pa.struct([("at", pa.timestamp("s")), ("index", pa.float64())])


def items(*values):
    return [None if value == "hole" else {"at": value, "index": value} for value in values]


def events(*collections, item=ITEM):
    return pa.table({"events": pa.array(collections, type=pa.list_(item))})


def test_delta_never_crosses_parent_collections():
    model = rf.Model(
        preset="small",
        d_model=8,
        groups=rf.Branch(
            length=2, events=rf.Branch(length=3, elapsed=rf.Duration(query="at", delta=True, granularity="seconds"))
        ),
    )
    group = pa.struct([("events", pa.list_(ITEM))])
    source = pa.table(
        {
            "groups": pa.array(
                [[{"events": items(1, 2, 3)}, {"events": items(4, 5)}], [None, {"events": items(6)}]],
                type=pa.list_(group),
            )
        }
    )
    field = model.encode(source)["/groups/events/elapsed"]
    assert field.content.reshape(2, 2, 3).tolist() == [
        [[0, 1, 1], [0, 1, 0]],
        [[0, 0, 0], [0, 0, 0]],
    ]
    assert field.state.reshape(2, 2, 3).tolist() == [
        [[N, V, V], [N, V, P]],
        [[P, P, P], [N, P, P]],
    ]


@pytest.mark.parametrize("overflow", ["head", "tail", "random"])
def test_delta_stays_aligned_with_other_fields_after_cropping(overflow):
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(
            length=3,
            overflow=overflow,
            index=rf.Number,
            elapsed=rf.Duration(query="at", delta=True, granularity="seconds"),
        ),
    )
    source = events(
        *[[{"at": row * 100 + index * index, "index": float(index)} for index in range(8)] for row in range(4)]
    )
    fields = model.encode(source, strata="train")
    indices = fields["/events/index"].content.reshape(-1).tolist()
    delta = fields["/events/elapsed"]
    for index, state, value in zip(indices, delta.state.reshape(-1).tolist(), delta.content.reshape(-1).tolist()):
        if index == 0:
            assert state == N
        else:
            assert state == V
            assert value == 2 * index - 1


def test_delta_is_independent_of_batch_boundaries_and_arrow_chunks():
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=3, elapsed=rf.Duration(query="at", delta=True, granularity="seconds")),
    )
    item = pa.struct(
        [("at", pa.timestamp("s")), ("index", pa.float64()), ("label", pa.dictionary(pa.int8(), pa.string()))]
    )
    collections = [items(1, 2, 3), [], None, items(4, "hole", 6), items(7)]
    source = events(*collections, item=item)
    chunked = pa.concat_tables([events(*collections[:2], item=item), events(*collections[2:], item=item)])
    whole = model.encode(source)["/events/elapsed"]
    split = model.encode(chunked)["/events/elapsed"]
    torch.testing.assert_close(split.content, whole.content)
    torch.testing.assert_close(split.state, whole.state)
    for row in range(source.num_rows):
        alone = model.encode(source.slice(row, 1))["/events/elapsed"]
        width = alone.state.shape[-1]
        torch.testing.assert_close(alone.state[0], whole.state[row][..., :width])
        torch.testing.assert_close(alone.content[0], whole.content[row][..., :width])


def test_sliced_chunks_preserve_only_their_candidate_ranges():
    rows, width = 24, 16
    records = pa.StructArray.from_arrays(
        [
            pa.array(np.arange(rows * width), type=pa.timestamp("s")),
            pa.array(["a", "b"] * (rows * width // 2)).dictionary_encode(),
        ],
        names=["at", "unrelated"],
    )
    offsets = pa.array(np.arange(0, (rows + 1) * width, width), type=pa.int32())
    collections = pa.ListArray.from_arrays(offsets, records)
    source = pa.table({"events": pa.chunked_array([collections.slice(row, 1) for row in range(rows)])})
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=4, overflow="tail", elapsed=rf.Duration(query="at", delta=True, granularity="seconds")),
    )
    coalesced = coalesce(source, model.schema, rf.Strata.predict)
    assert [len(chunk.records) for chunk in coalesced.layouts["/events"].collections] == [width] * rows
    field = model.encode(source)["/events/elapsed"]
    assert field.state.eq(V).all()
    assert field.content.eq(1.0).all()


def test_large_sliced_offsets_do_not_cross_collection_boundaries():
    count = 1_200_000_000
    empty = pa.DictionaryArray.from_arrays(pa.array([], type=pa.int8()), pa.array([], type=pa.string()))
    zero_lists = pa.Array.from_buffers(pa.list_(empty.type, 0), count, [None], children=[empty])
    records = pa.StructArray.from_arrays([pa.nulls(count), zero_lists], names=["at", "unrelated"])
    first = pa.ListArray.from_arrays(pa.array([0, 2], type=pa.int32()), records)
    second = pa.ListArray.from_arrays(pa.array([count - 2, count], type=pa.int32()), records)
    source = pa.table({"events": pa.chunked_array([first, second])})
    model = rf.Model(
        preset="small",
        d_model=8,
        events=rf.Branch(length=2, elapsed=rf.Duration(query="at", delta=True, granularity="seconds")),
    )
    field = model.encode(source)["/events/elapsed"]
    assert field.state.reshape(-1).tolist() == [N, N, N, N]
    assert [
        len(chunk.records) for chunk in coalesce(source, model.schema, rf.Strata.predict).layouts["/events"].collections
    ] == [2, 2]
