"""Seeded crop starts and hidden rows for the contrastive objective.

The offsets and the hidden rows are drawn on the data plane, where the run seed, epoch and batch
ordinal already exist, so the model never sees the seed and prediction draws
nothing. They travel to the model inside the branch geometry.
"""

import hashlib
import math

import numpy as np

from relflow.structs.contrastive import Crops
from relflow.structs.tree import Address

__all__ = ["hidden", "starts"]


def starts(lengths: np.ndarray, *, crops: Crops, seed: int, epoch: int, ordinal: int, address: Address) -> np.ndarray:
    """Draw the first row of two crops for every branch instance.

    The second crop starts ``shift`` rows after the first, where ``shift`` is the
    part of the width the crops do not share. A record shorter than the width
    plus the shift starts at row zero and its crops overlap more; a record
    shorter than the width, empty or null gets ``(0, 0)``.

    Args:
        lengths: Retained rows per branch instance.
        crops: Width and overlap of the two crops.
        seed: Data seed of the run.
        epoch: Epoch of the batch.
        ordinal: Batch ordinal within the epoch.
        address: Branch address, so branches draw independently.

    Returns:
        An int64 array of shape ``(*lengths.shape, 2)`` with the first row of each crop.
    """
    shift = crops.width - math.ceil(crops.overlap * crops.width)
    flat = lengths.reshape(-1)
    key = f"relflow-crops-v1:{seed}:{epoch}:{ordinal}:{address}".encode()
    generator = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    first = generator.integers(0, np.maximum(flat - crops.width - shift, 0) + 1)
    second = first + np.minimum(shift, np.maximum(flat - crops.width, 0))
    return np.stack((first, second), axis=-1).astype(np.int64).reshape(*lengths.shape, 2)


def hidden(
    lengths: np.ndarray, *, extent: int, crops: Crops, seed: int, epoch: int, ordinal: int, address: Address
) -> np.ndarray:
    """Draw which source rows each crop hides, independently per crop.

    The draw covers every row of the branch axis rather than the crop window, so a
    row shared by both crops is hidden in each with its own probability.

    Args:
        lengths: Retained rows per branch instance, which fixes the parent shape.
        extent: Rows of the branch axis.
        crops: Crop settings; ``mask`` is the probability that a row is hidden.
        seed: Data seed of the run.
        epoch: Epoch of the batch.
        ordinal: Batch ordinal within the epoch.
        address: Branch address, so branches draw independently.

    Returns:
        A bool array of shape ``(*lengths.shape, 2, extent)``, true where the crop hides the row.
    """
    key = f"relflow-mask-v1:{seed}:{epoch}:{ordinal}:{address}".encode()
    generator = np.random.default_rng(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
    return generator.random((*lengths.shape, 2, extent)) < crops.mask
