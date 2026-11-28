"""Compact token records preserve both tokenizer matrices and historical cache IDs."""

import numpy as np
import pytest

from relflow.tensorfields.extensions.text_cache.tokens import token_rows, tokens
from relflow.tensorfields.shared.cache import Payloads


@pytest.mark.parametrize("pad", [0, 17])
def test_compact_and_dense_records_roundtrip(pad):
    width = 128
    rows = np.zeros((7, 2, width), dtype=np.int64)
    rows[:, 0] = pad
    for row, start in ((0, 0), (1, width - 2), (2, 5)):
        rows[row, 0, start : start + 2] = [0, 7]
        rows[row, 1, start : start + 2] = 1
    rows[4, :, 2:5] = [[1, 0, 4], [1, 0, 1]]
    rows[5, :, 0] = [2**40, 1]
    rows[6, :, 0] = [19, 3]

    packed = tokens(rows, pad)
    assert np.diff(packed.offsets).tolist() == [20, 20, 20, 12, 2052, 2052, 2052]
    np.testing.assert_array_equal(token_rows(packed, width), rows)
    for row, record in zip(rows, packed, strict=True):
        assert next(iter(tokens(row[None], pad))) == record
    legacy = Payloads.from_values(width.to_bytes(4, "little") + row.astype("<i8").tobytes() for row in rows)
    np.testing.assert_array_equal(token_rows(legacy, width), rows)


@pytest.mark.parametrize("header", [[-4], [-4, -1, 0, 2], [-4, 4, 0, 2], [-3, 0, 0], [4, 0, 0]])
def test_invalid_token_records_raise(header):
    with pytest.raises(ValueError, match="cached text preparation"):
        token_rows(Payloads.from_values([np.array(header, dtype="<i4").tobytes()]), 4)
