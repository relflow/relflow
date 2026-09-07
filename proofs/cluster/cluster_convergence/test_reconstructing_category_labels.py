"""Show adaptive Cluster convergence when reconstruction is engaged.

Use repeated identities and ``Mask(rate=0.5, reconstruct=True)`` when adaptive
cluster commitment is intended. Each identity has a stable categorical regime,
giving the Cluster repeated evidence rather than one unrelated row per ID.

Example
-------
Repeated identities carry a stable hidden label while the reconstructing mask
occasionally hides the identity representation:

```yaml
input:
  observations:
    - {merchant_id: c0-id3}
    - {merchant_id: c0-id3}
    - {merchant_id: c4-id8}
  cluster_config:
    reconstruct_rate: 0.5
expected_output:
  hidden_labels: [L0, L0, L4]
  terminal_cluster_state:
    used_groups: "approximately 5"
```

The expected groups need not use labels 0 through 4 in any particular order.

The model trains for 30 deterministic CPU epochs. Across the final five epochs,
committed count must remain in ``[K-1, K+2]``, usage perplexity must remain
within 1.5 of true ``K=5``, and commitment must not stick to either configured
boundary. This is mechanistic evidence rather than a partition-recovery proof;
see ``support.py`` for status, limitations, and promotion criteria.

Run with::

    uv run pytest -n 0 proofs/cluster/cluster_convergence/test_reconstructing_category_labels.py -q
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
    format_trajectory,
    synthetic_records,
)

pytestmark = pytest.mark.proof


def test_cluster_n_committed_converges_near_generating_regimes() -> None:
    torch.manual_seed(0)
    records = synthetic_records()

    model = rf.Model(
        name="event",
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        merchant_id=rf.Cluster(
            capacity=64,
            n_clusters=(LOWER, UPPER),
            mask=rf.Mask(rate=0.5, reconstruct=True),
        ),
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
        max_epochs=30,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=datamodule)

    tail = trajectory.trajectory[-5:]
    final_n = [row["n_committed"] for row in tail]
    final_ppl = [row["perplexity"] for row in tail]

    assert all(TRUE_K - 1 <= n <= TRUE_K + 2 for n in final_n), (
        f"n_committed did not converge near true K={TRUE_K} over final 5 epochs: "
        f"{final_n}\nFull trajectory:\n{format_trajectory(trajectory.trajectory)}"
    )
    assert all(abs(p - TRUE_K) <= 1.5 for p in final_ppl), (
        f"usage perplexity did not settle near true K={TRUE_K}: {final_ppl}\n"
        f"Full trajectory:\n{format_trajectory(trajectory.trajectory)}"
    )
    assert final_n[-1] != UPPER, (
        f"n_committed glued to upper bound: {final_n}\nFull trajectory:\n{format_trajectory(trajectory.trajectory)}"
    )
    assert final_n[-1] != LOWER or LOWER == TRUE_K, (
        f"n_committed glued to lower bound (mechanism dormant): {final_n}\n"
        f"Full trajectory:\n{format_trajectory(trajectory.trajectory)}"
    )
