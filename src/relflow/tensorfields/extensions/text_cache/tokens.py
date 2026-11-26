"""Lossless token records for the shared artifact store."""

import numpy as np

from relflow.tensorfields.shared.cache import Payloads


def tokens(rows: np.ndarray, pad: int) -> Payloads:
    """Store the attended int32 slice. Unusual tokenizer outputs use the original int64 record."""
    ids, mask = rows[:, 0], rows[:, 1]
    width = ids.shape[1]
    active = mask == 1
    counts = active.sum(axis=1)
    starts = active.argmax(axis=1)
    ends = width - active[:, ::-1].argmax(axis=1)
    bounds = np.iinfo(np.int32)
    compact = (
        ((counts == 0) | (ends - starts == counts))
        & (mask == active).all(axis=1)
        & ((ids == pad) | active).all(axis=1)
        & ((ids >= bounds.min) & (ids <= bounds.max)).all(axis=1)
        & (bounds.min <= pad <= bounds.max)
    )

    # Negative width identifies the compact header (width, start, pad ID).
    # Positive width identifies the original header followed by both int64 matrices.
    offsets = np.zeros(len(rows) + 1, dtype="<u4")
    np.cumsum(np.where(compact, 3 + counts, 1 + 4 * width), out=offsets[1:])
    words = np.empty(int(offsets[-1]), dtype="<i4")
    base = offsets[:-1]
    words[base] = np.where(compact, -width, width)
    if compact.any():
        words[base[compact] + 1] = starts[compact]
        words[base[compact] + 2] = pad
        row, pos = np.nonzero(active & compact[:, None])
        words[base[row] + 3 + pos - starts[row]] = ids[row, pos]
    if not compact.all():
        dense = rows[~compact].astype("<i8", copy=False).reshape(-1, 2 * width).view("<i4")
        words[base[~compact, None] + 1 + np.arange(4 * width)] = dense
    return Payloads(memoryview(words).cast("B"), offsets * 4)


def token_rows(payloads: Payloads, length: int) -> np.ndarray:
    """Rebuild the exact token and mask matrices, including padding positions and IDs."""
    sizes = np.diff(payloads.offsets)
    if np.any(sizes % 4) or np.any(sizes < 4):
        raise ValueError("cached text preparation has an invalid token record size")
    base = payloads.offsets[:-1] // 4
    words = np.frombuffer(payloads.data, dtype="<i4")
    widths = words[base]
    compact = widths == -length
    if np.any(~compact & (widths != length)):
        raise ValueError("cached text preparation has a different token length")
    if np.any(sizes[~compact] != 4 + 16 * length) or np.any(sizes[compact] < 12):
        raise ValueError("cached text preparation has a different token shape")

    rows = np.empty((len(payloads), 2, length), dtype=np.int64)
    if not compact.all():
        dense = words[base[~compact, None] + 1 + np.arange(4 * length)]
        rows[~compact] = dense.view("<i8").reshape(-1, 2, length)
    if compact.any():
        headers = base[compact]
        starts = words[headers + 1].astype(np.int64)
        ends = starts + sizes[compact] // 4 - 3
        if np.any(starts < 0) or np.any(ends > length):
            raise ValueError("cached text preparation has an invalid attended token range")
        positions = np.arange(length)
        active = (positions >= starts[:, None]) & (positions < ends[:, None])
        rows[compact, 0] = words[headers + 2, None]
        rows[compact, 1] = active
        row, pos = np.nonzero(active)
        rows[np.flatnonzero(compact)[row], 0, pos] = words[headers[row] + 3 + pos - starts[row]]
    return rows
