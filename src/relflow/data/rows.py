"""Pure NumPy arithmetic over the retained rows of branch records.

``placement`` is the sorted dense position of each retained record and
``extent`` the row axis size, so ``placement // extent`` is the parent of a
record and the records of one parent are contiguous.
"""

import numpy as np

__all__ = ["last", "ranks", "runs"]


def ranks(placement: np.ndarray, *, extent: int) -> tuple[np.ndarray, np.ndarray]:
    """Number each retained record within its parent.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.

    Returns:
        The rank of each record within its parent and the retained count of that parent.
    """
    if not len(placement):
        return np.empty(0, dtype=np.int64), np.empty(0, dtype=np.int64)
    _, first, counts = np.unique(placement // extent, return_index=True, return_counts=True)
    rank = np.arange(len(placement), dtype=np.int64) - np.repeat(first, counts)
    return rank, np.repeat(counts, counts)


def last(placement: np.ndarray, *, extent: int, count: int) -> np.ndarray:
    """Mark the final ``count`` retained records of each parent.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.
        count: Number of trailing records to mark.

    Returns:
        A Boolean per record.
    """
    rank, total = ranks(placement, extent=extent)
    return rank >= total - count


def runs(placement: np.ndarray, *, extent: int, starts: np.ndarray, length: int) -> np.ndarray:
    """Mark records within ``length`` records after a start of the same parent, start included.

    Args:
        placement: Sorted dense positions of the retained records.
        extent: Size of the row axis.
        starts: Boolean per record that begins a run.
        length: Records one run covers.

    Returns:
        A Boolean per record.
    """
    rank, _ = ranks(placement, extent=extent)
    index = np.arange(len(placement))
    window = np.maximum(index - length + 1, index - rank)
    counted = np.concatenate(([0], np.cumsum(starts)))
    return counted[index + 1] - counted[window] > 0
