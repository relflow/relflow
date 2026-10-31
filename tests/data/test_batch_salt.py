"""Training batch salts come from the run seed, not from the process-wide random state."""

import random

import pyarrow as pa
import torch

import relflow as rf
from relflow.data.iterables import batch_salt


def test_batch_salt_follows_seed_epoch_and_ordinal():
    assert batch_salt(0, epoch=0, ordinal=0) == batch_salt(0, epoch=0, ordinal=0) != 0
    distinct = {
        batch_salt(0, epoch=0, ordinal=0),
        batch_salt(0, epoch=0, ordinal=1),
        batch_salt(0, epoch=1, ordinal=0),
        batch_salt(1, epoch=0, ordinal=0),
    }
    assert len(distinct) == 4


def test_training_batches_repeat_exactly_whatever_the_process_rng_did():
    model = rf.Model.xs(key=rf.Hash, amount=rf.Number, label=rf.Boolean(mask=True), d_model=8, batch_size=2)
    table = pa.table({"key": ["a", "b", "a", "d"], "amount": [1.0, 2.0, 3.0, 4.0], "label": [True, False, True, False]})

    def leaves(seed: int) -> list[tuple[str, torch.Tensor]]:
        random.seed(seed)
        random.getrandbits(64)
        data = rf.ArrowDataModule(
            model=model, train=table, shuffle=False, num_workers=0, persistent_workers=False, pin_memory=False
        )
        out = []
        for batch in data.train_dataloader():
            out.extend(
                (str(key), value.clone()) for key, value in batch.tensors.items(include_nested=True, leaves_only=True)
            )
        return out

    state = random.getstate()
    try:
        first, second = leaves(1), leaves(2)
    finally:
        random.setstate(state)
    assert [key for key, _ in first] == [key for key, _ in second]
    assert all(torch.equal(a, b) for (_, a), (_, b) in zip(first, second, strict=True))
