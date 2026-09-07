"""Discover repeated-ID clusters from hidden regression regimes.

Each repeated ID belongs to one of five hidden ``y = f_k(x)`` regimes. The
model sees no cluster label, only ``(id, x, y)``. It must discover the partition
through pressure to predict ``y`` from ``x`` plus masked-ID reconstruction.

Example
-------
One repeated identity follows one hidden function across observations. For
representative identities assigned to two of the five generated regimes:

```yaml
input:
  observations:
    - {id: id-0007, x: 0.00}
    - {id: id-0007, x: 1.57}
    - {id: id-0042, x: 1.57}
expected_output:
  hidden_y_targets:
    - {id: id-0007, y: "approximately 10"}
    - {id: id-0007, y: "approximately 11"}
    - {id: id-0042, y: "approximately -2"}
  terminal_cluster_state:
    used_groups: "approximately 5"
```

The proof does not require a particular numeric cluster label because learned
cluster labels may be permuted.

This corrects the unique-ID-per-row benchmark that made Cluster state dormant.
After 30 deterministic CPU epochs, the final five committed counts must remain
in ``[K-1, K+2]``, usage perplexity must remain within 1.5 of true ``K=5``, and
the final count must not equal either configured bound. This remains
mechanistic evidence: train and validation currently reuse observations and no
held-out prediction or ARI gate is present. See ``support.py`` for the complete
remaining-work and promotion criteria.

Run with::

    uv run pytest -n 0 proofs/cluster/cluster_convergence/test_hidden_regression_regimes.py -q
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
    regime_records,
)

pytestmark = pytest.mark.proof


def test_cluster_n_committed_converges_via_regime_regression() -> None:
    torch.manual_seed(0)
    records = regime_records(num_ids=100, obs_per_id=30)

    model = rf.Model(
        name="obs",
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=3e-3),
        id=rf.Cluster(
            capacity=128,
            n_clusters=(LOWER, UPPER),
            mask=rf.Mask(rate=0.5, reconstruct=True),
        ),
        x=rf.Number,
        y=rf.Number(mask=True),
    )

    datamodule = rf.ArrowDataModule(
        model=model,
        train=records,
        validate=records,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )

    address = rf.Address("obs", "id")
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
    assert final_n[-1] != LOWER, (
        f"n_committed glued to lower bound: {final_n}\nFull trajectory:\n{format_trajectory(trajectory.trajectory)}"
    )
