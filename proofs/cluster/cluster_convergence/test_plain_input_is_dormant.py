"""Demonstrate the dormant plain-input Cluster footgun.

Avoid expecting ``n_committed`` to learn from a plain input field. Downstream
supervision on ``label`` does not invoke the Cluster reconstruction loss, so
``usage_ema`` never moves and commitment remains at its initialized value.

Example
-------
The ID repeats and predicts a stable label, but the Cluster itself has no
reconstruction objective:

```yaml
input:
  observations:
    - {merchant_id: c2-id7}
    - {merchant_id: c2-id7}
  cluster_config:
    reconstruct: false
expected_output:
  hidden_labels: [L2, L2]
  cluster_state:
    n_committed: unchanged
    adherence: 0.0
```

The expected output of this negative proof is unchanged Cluster state, not a
newly discovered five-group partition.

This negative variant deliberately omits a reconstructing Mask and trains for
five deterministic CPU epochs. The committed-count trajectory must be
constant, and adherence must remain exactly zero. To train adaptive Cluster
state, use the reconstructing setup shown in
``test_reconstructing_category_labels.py``. See ``support.py`` for overall
status, limitations, and promotion criteria.

Run with::

    uv run pytest -n 0 proofs/cluster/cluster_convergence/test_plain_input_is_dormant.py -q
"""

import lightning.pytorch as lit
import pytest
import torch

import relflow as rf
from proofs.cluster.cluster_convergence.support import (
    LOWER,
    TRUE_K,
    UPPER,
    CommittedTrajectory,
    synthetic_records,
)

pytestmark = pytest.mark.proof


def test_cluster_loss_is_dormant_when_field_is_plain_input() -> None:
    torch.manual_seed(0)
    records = synthetic_records()

    model = rf.Model(
        name="event",
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        merchant_id=rf.Cluster(capacity=64, n_clusters=(LOWER, UPPER)),
        label=rf.Category(mask=True, size=TRUE_K, p_unavailable=0.0),
    )

    datamodule = rf.ArrowDataModule(
        model=model,
        train=records,
        validate=records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    address = rf.Address("event", "merchant_id")
    trajectory = CommittedTrajectory(address)

    trainer = lit.Trainer(
        accelerator="cpu",
        max_epochs=5,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=datamodule)

    n_series = [row["n_committed"] for row in trajectory.trajectory]
    adherence = [row["adherence"] for row in trajectory.trajectory]
    assert len(set(n_series)) == 1, f"expected frozen n_committed for dormant cluster, got {n_series}"
    assert all(value == 0.0 for value in adherence), f"expected zero adherence for dormant cluster, got {adherence}"
