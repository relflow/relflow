"""Shared synthetic data and diagnostics for adaptive Cluster convergence.

Claim
-----
Adaptive Cluster commitment can settle near a repeated identity process's
generating regimes when the Cluster reconstruction objective is engaged.
Assignments are properties of stable identities, so row-varying siblings may
influence downstream tasks through branch memory but do not directly condition
the Cluster reconstruction query.

What to do
----------
Use repeated observations per identity and
``Mask(rate=..., reconstruct=True)`` when the number and assignments of
clusters should learn. Monitor both committed count and usage perplexity.

What not to do
--------------
Do not expect downstream supervision alone to update a plain-input Cluster.
Do not use unique IDs with no repeated behavioral evidence as a convergence
benchmark, and do not treat a near-K count as proof that the partition itself
is correct.

Why
---
Cluster usage state is updated by its reconstruction loss. Repeated identities
let assignment evidence accumulate, while excluding direct sibling query
conditioning prevents unrelated per-row covariates from manufacturing extra
clusters through that shortcut.

Proof status
------------
This proof is a partial mechanistic characterization. Three single-seed gates
cover committed count, usage perplexity, and the dormant plain-input footgun;
they do not yet prove partition recovery or held-out prediction.

Current evidence
----------------
With a reconstructing Mask, the final five epochs remain near the true ``K=5``
in both labeled and hidden-regime processes. A plain-input Cluster leaves
commitment fixed and adherence at zero.

A Cluster needs an engaged reconstruction objective and repeated evidence per
identity. Downstream supervision on another field does not train a dormant
cluster head. Keeping row-varying sibling values out of the direct Cluster
decoder query prevents one identity from fragmenting across its observations.

Protocol and gates
------------------
The positive variants train for 30 deterministic CPU epochs and require the
last five committed counts in ``[K-1, K+2]``, perplexity within 1.5 of ``K``,
and a terminal count at neither configured bound. The negative variant trains
for five epochs and requires commitment and adherence to remain unchanged.

Further work
------------
Add held-out downstream skill versus a marginal baseline and
permutation-invariant assignment ``ARI >= 0.80``. Use independent observations
for train, validation, and test; the current train and validation tables are
identical. Add unique-ID and no-regime controls. Replace the private
``Embedder`` inspection with a public diagnostic, or label it explicitly as
temporary mechanistic instrumentation.

Promotion criteria
------------------
All behavioral, partition, and control gates must pass across three seeds. Use
at least ten seeds before freezing the terminal-window thresholds.
"""

from __future__ import annotations

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import torch

import relflow as rf
from relflow.tensorfields.extensions.cluster import Embedder

TRUE_K: int = 5
IDS_PER_CLUSTER: int = 10
OBS_PER_ID: int = 20
LOWER: int = 3
UPPER: int = 15


def regime_records(
    *,
    num_ids: int = 100,
    obs_per_id: int = 30,
    seed: int = 42,
) -> pa.Table:
    """Create repeated IDs mapped to one of K hidden ``y = f_k(x)`` regimes.

    Unlike the naive unique-ID-per-row setup that makes the mechanism dormant,
    each ID appears ``obs_per_id`` times so the cluster head can pool gradients
    across observations of the same ID.
    """
    rng = np.random.default_rng(seed)
    epsilon = 0.05
    x_low, x_high = 0.0, 10.0

    functions = [
        lambda x: np.sin(x) + 10.0,
        lambda x: np.cos(x) + 4.0,
        lambda x: np.zeros_like(x) - 2.0,
        lambda x: 0.5 * x - 8.0,
        lambda x: -0.3 * x - 14.0,
    ]
    assert len(functions) == TRUE_K

    id_to_cluster = rng.integers(0, TRUE_K, size=num_ids)

    n = num_ids * obs_per_id
    id_indices = np.repeat(np.arange(num_ids), obs_per_id)
    cluster_indices = id_to_cluster[id_indices]
    x = rng.uniform(x_low, x_high, size=n) + rng.normal(0.0, epsilon, size=n)
    y = np.empty(n)
    for k, function in enumerate(functions):
        mask = cluster_indices == k
        y[mask] = function(x[mask]) + rng.normal(0.0, epsilon, size=int(mask.sum()))

    permutation = rng.permutation(n)
    return pa.table(
        {
            "id": [f"id-{i:04d}" for i in id_indices[permutation].tolist()],
            "x": x[permutation].tolist(),
            "y": y[permutation].tolist(),
        }
    )


def synthetic_records(seed: int = 0) -> pa.Table:
    generator = torch.Generator().manual_seed(seed)
    rows: list[dict[str, object]] = []
    for cluster_idx in range(TRUE_K):
        for id_idx in range(IDS_PER_CLUSTER):
            merchant_id = f"c{cluster_idx}-id{id_idx}"
            for _ in range(OBS_PER_ID):
                rows.append(
                    {
                        "merchant_id": merchant_id,
                        "label": f"L{cluster_idx}",
                    }
                )
    indices = torch.randperm(len(rows), generator=generator).tolist()
    shuffled = [rows[i] for i in indices]
    return pa.table(
        {
            "merchant_id": [row["merchant_id"] for row in shuffled],
            "label": [row["label"] for row in shuffled],
        }
    )


class CommittedTrajectory(lit.Callback):
    """Temporary private instrumentation for adaptive Cluster state."""

    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.trajectory: list[dict[str, float]] = []

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module) -> None:  # type: ignore[override]
        embedder = pl_module.nodes[self.address].embedder
        assert isinstance(embedder, Embedder)
        usage = embedder.usage_ema.detach()
        usage_norm = usage / usage.sum().clamp_min(1e-12)
        entropy = -(usage_norm.clamp_min(1e-12) * usage_norm.clamp_min(1e-12).log()).sum()
        self.trajectory.append(
            {
                "epoch": trainer.current_epoch,
                "n_committed": int(embedder.committed.sum().item()),
                "perplexity": float(torch.exp(entropy).item()),
                "adherence": float(embedder.adherence_ema.item()),
            }
        )


def format_trajectory(trajectory: list[dict[str, float]]) -> str:
    header = f"{'epoch':>5} {'n_committed':>12} {'perplexity':>11} {'adherence':>10}"
    rows = [
        f"{row['epoch']:>5.0f} {row['n_committed']:>12.0f} {row['perplexity']:>11.3f} {row['adherence']:>10.4f}"
        for row in trajectory
    ]
    return "\n".join([header, *rows])
