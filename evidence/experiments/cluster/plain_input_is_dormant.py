# %% [markdown]
# ---
# title: Does a Cluster input group itself automatically?
# categories:
# - Cluster
# evidence-id: P019
# description: Use a Cluster field only as an input beside a hidden prediction target.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Groups at the final epoch
#   metric:
#   - committed
#   - -1
#   format: number
# - label: Grouping activity at the final epoch
#   metric:
#   - adherence
#   - -1
#   format: number
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Adherence remains exactly zero: true
#     Committed count remains constant: true
#   duration_seconds: 2.262
#   metrics:
#     adherence:
#     - 0.0
#     - 0.0
#     - 0.0
#     - 0.0
#     - 0.0
#     committed:
#     - 3
#     - 3
#     - 3
#     - 3
#     - 3
#     trajectory:
#     - adherence: 0.0
#       epoch: 0
#       n_committed: 3
#       perplexity: 3.0
#     - adherence: 0.0
#       epoch: 1
#       n_committed: 3
#       perplexity: 3.0
#     - adherence: 0.0
#       epoch: 2
#       n_committed: 3
#       perplexity: 3.0
#     - adherence: 0.0
#       epoch: 3
#       n_committed: 3
#       perplexity: 3.0
#     - adherence: 0.0
#       epoch: 4
#       n_committed: 3
#       perplexity: 3.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 9bd13e83632e3979869446c74c2194261687896006172a10998215e4f1ffc04d
#     cuda: null
#     evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     gpu: null
#     host: 1fd362f58afe
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 7dec5e860dc389c37c3642d8904867249e831773
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 4fea121461e8a252b1319905995a652726d33fe9aa0aa7e98827173163fd740d
#     source_sha256: c243274d899605e59f4be130208912e219015b4c6c91edb1f9948cf10449eade
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   seed: 0
#   started_at: '2026-11-02T04:08:23.519751+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41386396488-1
#   candidate:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   baseline:
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: f38d698e13bd1bfa65f22f7eaf89bab8a17db0bb401d89e19c4112afcc467c37
#   evidence_revision: 7dec5e860dc389c37c3642d8904867249e831773
#   executor_revision: 349b2202e6390a5fe0cbdf09a0c5972e495351ed
#   manifest_sha256: 1f096d9b4dcdf1b2235963715e8b41c325d2a989f2cf15da7c154ddfb9bfb08d
#   passed: true
#   github_run:
#     event: workflow_dispatch
#     head_branch: main
#     head_sha: 7dec5e860dc389c37c3642d8904867249e831773
#     html_url: https://github.com/relflow/relflow/actions/runs/41386396488
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41386396488
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41386396488-attempt-1
#   reports_sha256: fca05e4cdb79410b5090075bab03e4fc6d4f2fbc70e798f208976f4f1d8a7ff9
# ---
#
# ## Example
#
# ```yaml
# merchant_id: c2-id7
# label: L2
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-cluster-plain-input-is-dormant
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with event, merchant_id, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("event", kind: "root", children: (
#   node("merchant_id", type: "Cluster", width: 150pt,),
#   node("label", kind: "target", type: "Category",),
# )))
# ```
#
# ## Comparison
#
# Observe its adaptive grouping state before and after training, without asking the Cluster field to reconstruct itself.
#
# ## Result
#
# {{< evidence P019 status >}}
#
# Its grouping state stays dormant in this setup. Add a reconstructing mask when adaptive grouping is intended; then test whether the groups are useful.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P019 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P019 script >}}
#
# ### How it works
#
# Cluster's reconstruction loss updates its adaptive usage and commitment state.
# Training only the sibling label objective does not invoke that loss. The
# absence of a Cluster reconstruction objective therefore leaves the measured
# commitment mechanism inactive even though the identity recurs.
#
# This model uses 1,000 observations: five label groups, ten identities per group,
# and 20 observations per identity. It trains for five deterministic epochs,
# reusing the same table for validation. A callback inspects the committed count
# and adherence at each epoch.
#
# To engage the mechanism, the paired
# [reconstructing-label experiment](reconstructing-category-labels.html) adds
# `rf.Mask(rate=0.5, reconstruct=True)` to the Cluster field.
#
# ### Remaining work
#
# Repeat this negative control alongside the positive variants across three
# core seeds. The broader family still needs independent held-out observations,
# partition recovery, and unique-identity and no-regime controls. Its count
# thresholds also need at least ten calibration seeds.
#
# ### Complete experiment code
#

# %%
"""P019: show that a plain-input Cluster keeps its adaptive state dormant.

The supervised label objective does not invoke the Cluster reconstruction
loss. Repeated identities alone should leave commitment and adherence frozen.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P019"


def records(seed: int) -> Iterator[dict]:
    """Repeat each of 50 identities twenty times with one stable regime label."""
    rows = [
        {"merchant_id": f"c{cluster}-id{identity}", "label": f"L{cluster}"}
        for cluster in range(5)
        for identity in range(10)
        for _ in range(20)
    ]
    order = torch.randperm(len(rows), generator=torch.Generator().manual_seed(seed)).tolist()
    for index in order:
        yield rows[index]


class Trajectory(lit.Callback):
    """Inspect Cluster internals; these diagnostics do not establish partition accuracy."""

    def __init__(self, address: rf.Address) -> None:
        self.address = address
        self.rows: list[dict] = []

    def on_train_epoch_end(self, trainer: lit.Trainer, pl_module: lit.LightningModule) -> None:
        embedder = pl_module.nodes[self.address].embedder
        if not isinstance(embedder, rf.TENSORFIELDS["cluster"].Embedder):
            raise TypeError(f"{self.address} requires a Cluster embedder, got {type(embedder).__name__}")
        usage = embedder.usage_ema.detach()
        probabilities = usage / usage.sum().clamp_min(1e-12)
        bounded = probabilities.clamp_min(1e-12)
        entropy = -(bounded * bounded.log()).sum()
        self.rows.append(
            {
                "epoch": trainer.current_epoch,
                "n_committed": int(embedder.committed.sum().item()),
                "perplexity": float(torch.exp(entropy).item()),
                "adherence": float(embedder.adherence_ema.item()),
            }
        )


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        merchant_id=rf.Cluster(bounds=(3, 15)),
        label=rf.Category(mask=True, p_unavailable=0.0),
    )
    source = partial(records, seed=seed)
    data = rf.SyntheticDataModule(train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("merchant_id"))
    model.fit(
        data,
        batch_size=64,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.003),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=5,
        max_steps=steps if steps is not None else -1,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )

    committed = [row["n_committed"] for row in trajectory.rows]
    adherence = [row["adherence"] for row in trajectory.rows]
    return {"trajectory": trajectory.rows, "committed": committed, "adherence": adherence}, {
        "Committed count remains constant": len(set(committed)) == 1,
        "Adherence remains exactly zero": all(value == 0.0 for value in adherence),
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)

# %% [markdown]
# </details>
