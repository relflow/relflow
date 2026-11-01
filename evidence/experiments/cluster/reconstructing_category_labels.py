# %% [markdown]
# ---
# title: Do stable labels produce the right groups?
# categories:
# - Cluster
# evidence-id: P020
# description: Repeated identities carry stable labels. Ask the Cluster field to reconstruct identities while predicting
#   their labels.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Groups at the final epoch
#   metric:
#   - final_committed
#   - -1
#   format: number
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Final committed counts remain between 4 and 7: true
#     Final perplexity remains within 1.5 of five: true
#     Terminal commitment is above the lower bound: true
#     Terminal commitment is below the upper bound: true
#   duration_seconds: 16.886
#   metrics:
#     final_committed:
#     - 5
#     - 5
#     - 5
#     - 5
#     - 5
#     final_perplexity:
#     - 5.114455699920654
#     - 5.104739189147949
#     - 5.096655368804932
#     - 5.089267253875732
#     - 5.082054138183594
#     trajectory:
#     - adherence: 0.04619096964597702
#       epoch: 0
#       n_committed: 4
#       perplexity: 4.081431865692139
#     - adherence: 0.007059536408632994
#       epoch: 1
#       n_committed: 5
#       perplexity: 5.285671234130859
#     - adherence: 0.012704098597168922
#       epoch: 2
#       n_committed: 4
#       perplexity: 4.450099945068359
#     - adherence: 0.014835817739367485
#       epoch: 3
#       n_committed: 5
#       perplexity: 4.599492073059082
#     - adherence: 0.019742105156183243
#       epoch: 4
#       n_committed: 5
#       perplexity: 5.057950019836426
#     - adherence: 0.01945986971259117
#       epoch: 5
#       n_committed: 5
#       perplexity: 5.258633613586426
#     - adherence: 0.0013292209478095174
#       epoch: 6
#       n_committed: 6
#       perplexity: 6.37700891494751
#     - adherence: 0.004482896067202091
#       epoch: 7
#       n_committed: 5
#       perplexity: 5.396448135375977
#     - adherence: 0.005243333522230387
#       epoch: 8
#       n_committed: 5
#       perplexity: 5.4288716316223145
#     - adherence: 0.005397984758019447
#       epoch: 9
#       n_committed: 5
#       perplexity: 5.433709144592285
#     - adherence: 0.005319373682141304
#       epoch: 10
#       n_committed: 5
#       perplexity: 5.410321235656738
#     - adherence: 0.005141713656485081
#       epoch: 11
#       n_committed: 5
#       perplexity: 5.392104625701904
#     - adherence: 0.004907333757728338
#       epoch: 12
#       n_committed: 5
#       perplexity: 5.3671135902404785
#     - adherence: 0.004642982967197895
#       epoch: 13
#       n_committed: 5
#       perplexity: 5.345626354217529
#     - adherence: 0.004362401086837053
#       epoch: 14
#       n_committed: 5
#       perplexity: 5.319127082824707
#     - adherence: 0.0040809898637235165
#       epoch: 15
#       n_committed: 5
#       perplexity: 5.292667865753174
#     - adherence: 0.00380383781157434
#       epoch: 16
#       n_committed: 5
#       perplexity: 5.265294075012207
#     - adherence: 0.0035371228586882353
#       epoch: 17
#       n_committed: 5
#       perplexity: 5.240423202514648
#     - adherence: 0.0032878248021006584
#       epoch: 18
#       n_committed: 5
#       perplexity: 5.220227241516113
#     - adherence: 0.0030514413956552744
#       epoch: 19
#       n_committed: 5
#       perplexity: 5.200494289398193
#     - adherence: 0.0028315214440226555
#       epoch: 20
#       n_committed: 5
#       perplexity: 5.18286657333374
#     - adherence: 0.0026295857969671488
#       epoch: 21
#       n_committed: 5
#       perplexity: 5.166749477386475
#     - adherence: 0.0024415047373622656
#       epoch: 22
#       n_committed: 5
#       perplexity: 5.152550220489502
#     - adherence: 0.0022703739814460278
#       epoch: 23
#       n_committed: 5
#       perplexity: 5.1390557289123535
#     - adherence: 0.0021123678889125586
#       epoch: 24
#       n_committed: 5
#       perplexity: 5.126739978790283
#     - adherence: 0.0019665879663079977
#       epoch: 25
#       n_committed: 5
#       perplexity: 5.114455699920654
#     - adherence: 0.0018323969561606646
#       epoch: 26
#       n_committed: 5
#       perplexity: 5.104739189147949
#     - adherence: 0.0017118208343163133
#       epoch: 27
#       n_committed: 5
#       perplexity: 5.096655368804932
#     - adherence: 0.0016020586481317878
#       epoch: 28
#       n_committed: 5
#       perplexity: 5.089267253875732
#     - adherence: 0.0015011352952569723
#       epoch: 29
#       n_committed: 5
#       perplexity: 5.082054138183594
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 457383d3163f0f2bdb642b078b0e0fd27ecaaf7b192cc5f352710fdbfd73abd4
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 1d174cdeb1f6
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: 62cddb487df95a8d458459a4a586f111f060609ec531a02851410cf97a72373e
#     source_sha256: 8b1c0c9bea462ffc9707e837705cf928e077d8d52f3bf1505099b4c1f67b01d2
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 0
#   started_at: '2026-11-01T01:36:03.993116+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
#
# ## Example
#
# ```yaml
# merchant_id: c0-id3
# label: L0
# ```
#
# The label is hidden; identities are sometimes hidden for reconstruction during training.
#
# ```{typst}
# //| label: fig-evidence-cluster-reconstructing-category-labels
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
# Compare learned group count and usage with the five hidden label groups.
#
# ## Result
#
# {{< evidence P020 status >}}
#
# The model reaches roughly five groups, but count and usage do not measure whether grouping is correct. Reused validation records also leave independent predictive quality untested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P020 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P020 script >}}
#
# ### How it works
#
# The merchant field uses
# `rf.Cluster(mask=rf.Mask(rate=0.5, reconstruct=True), ...)`. Its reconstruction
# loss updates adaptive usage and commitment state. Repetition lets evidence
# accumulate for an identity. `label=rf.Category(mask=True, ...)` supplies the
# supervised task without exposing its answer as an input.
#
# Assignments belong to stable identities. Row-varying sibling context does not
# directly condition the Cluster reconstruction query. The
# [plain-input control](plain-input-is-dormant.html) shows what happens when the
# Cluster reconstruction objective is absent.
#
# The model trains for 30 deterministic epochs. A callback inspects internal
# Cluster state at each epoch; the gates examine its final five entries.
#
# ### Remaining work
#
# Use independent observations for each split, add held-out downstream skill
# against a marginal baseline, and require adjusted Rand index (ARI) ≥ 0.80
# for partition recovery. Add unique-identity and no-regime controls. Replace
# internal-state inspection with a public diagnostic when available.
#
# The family requires three passing core seeds and at least ten calibration
# seeds before its terminal-window thresholds can be promoted.
#
# ### Complete experiment code
#

# %%
"""P020: track adaptive clusters across five repeated-identity labels.

Training and validation deliberately reuse observations. Count and usage
perplexity diagnose the mechanism; they do not prove partition recovery or
held-out prediction quality.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P020"


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
        merchant_id=rf.Cluster(bounds=(3, 15), mask=rf.Mask(rate=0.5, reconstruct=True)),
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
        max_epochs=30,
        max_steps=steps if steps is not None else -1,
        callbacks=[trajectory],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )

    tail = trajectory.rows[-5:]
    committed = [row["n_committed"] for row in tail]
    perplexity = [row["perplexity"] for row in tail]
    return {
        "trajectory": trajectory.rows,
        "final_committed": committed,
        "final_perplexity": perplexity,
    }, {
        "Final committed counts remain between 4 and 7": all(4 <= value <= 7 for value in committed),
        "Final perplexity remains within 1.5 of five": all(abs(value - 5) <= 1.5 for value in perplexity),
        "Terminal commitment is above the lower bound": committed[-1] != 3,
        "Terminal commitment is below the upper bound": committed[-1] != 15,
    }


if __name__ == "__main__":
    report(PROOF_ID, run, seed=0)

# %% [markdown]
# </details>
