# %% [markdown]
# ---
# title: Do five clusters mean five behaviors were found?
# categories:
# - Cluster
# evidence-id: P018
# description: Repeated identities follow five hidden numerical rules. Watch how the Cluster field groups them during
#   training.
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
#     Final committed counts remain between 4 and 7: false
#     Final perplexity remains within 1.5 of five: false
#     Terminal commitment is above the lower bound: true
#     Terminal commitment is below the upper bound: true
#   duration_seconds: 31.157
#   metrics:
#     final_committed:
#     - 8
#     - 8
#     - 8
#     - 8
#     - 8
#     final_perplexity:
#     - 7.921018123626709
#     - 7.864084720611572
#     - 7.864062309265137
#     - 7.8577070236206055
#     - 7.895829677581787
#     trajectory:
#     - adherence: 0.0596262626349926
#       epoch: 0
#       n_committed: 4
#       perplexity: 4.4743194580078125
#     - adherence: 0.06287430971860886
#       epoch: 1
#       n_committed: 5
#       perplexity: 5.211883068084717
#     - adherence: 0.004480892792344093
#       epoch: 2
#       n_committed: 6
#       perplexity: 6.449525833129883
#     - adherence: 0.0044525605626404285
#       epoch: 3
#       n_committed: 8
#       perplexity: 7.820277214050293
#     - adherence: 0.010754283517599106
#       epoch: 4
#       n_committed: 7
#       perplexity: 6.627710342407227
#     - adherence: 0.011299673467874527
#       epoch: 5
#       n_committed: 7
#       perplexity: 6.919882297515869
#     - adherence: 0.010402935557067394
#       epoch: 6
#       n_committed: 7
#       perplexity: 6.963478088378906
#     - adherence: 0.009252095595002174
#       epoch: 7
#       n_committed: 7
#       perplexity: 6.974049091339111
#     - adherence: 0.008121979422867298
#       epoch: 8
#       n_committed: 7
#       perplexity: 7.066467761993408
#     - adherence: 0.007775044068694115
#       epoch: 9
#       n_committed: 7
#       perplexity: 7.114893436431885
#     - adherence: 0.006880196277052164
#       epoch: 10
#       n_committed: 7
#       perplexity: 7.067854404449463
#     - adherence: 0.005913118366152048
#       epoch: 11
#       n_committed: 7
#       perplexity: 6.948287010192871
#     - adherence: 0.0052897571586072445
#       epoch: 12
#       n_committed: 7
#       perplexity: 6.861160755157471
#     - adherence: 0.004672950133681297
#       epoch: 13
#       n_committed: 7
#       perplexity: 6.929600238800049
#     - adherence: 0.004071409814059734
#       epoch: 14
#       n_committed: 7
#       perplexity: 6.949246406555176
#     - adherence: 0.000262480549281463
#       epoch: 15
#       n_committed: 8
#       perplexity: 8.127616882324219
#     - adherence: 0.0003884865727741271
#       epoch: 16
#       n_committed: 8
#       perplexity: 8.008110046386719
#     - adherence: 0.0004629143513739109
#       epoch: 17
#       n_committed: 8
#       perplexity: 7.916283130645752
#     - adherence: 0.0006673262105323374
#       epoch: 18
#       n_committed: 8
#       perplexity: 8.012361526489258
#     - adherence: 0.0007330988300964236
#       epoch: 19
#       n_committed: 8
#       perplexity: 8.081974983215332
#     - adherence: 0.0007852783310227096
#       epoch: 20
#       n_committed: 8
#       perplexity: 8.076993942260742
#     - adherence: 0.0008025173447094858
#       epoch: 21
#       n_committed: 8
#       perplexity: 7.995616436004639
#     - adherence: 0.0008230764651671052
#       epoch: 22
#       n_committed: 8
#       perplexity: 7.957146644592285
#     - adherence: 0.0008152077789418399
#       epoch: 23
#       n_committed: 8
#       perplexity: 7.9559550285339355
#     - adherence: 0.0007473612786270678
#       epoch: 24
#       n_committed: 8
#       perplexity: 7.930139541625977
#     - adherence: 0.0006774786161258817
#       epoch: 25
#       n_committed: 8
#       perplexity: 7.921018123626709
#     - adherence: 0.0006148697575554252
#       epoch: 26
#       n_committed: 8
#       perplexity: 7.864084720611572
#     - adherence: 0.0005607149796560407
#       epoch: 27
#       n_committed: 8
#       perplexity: 7.864062309265137
#     - adherence: 0.0005023656412959099
#       epoch: 28
#       n_committed: 8
#       perplexity: 7.8577070236206055
#     - adherence: 0.00046136201126500964
#       epoch: 29
#       n_committed: 8
#       perplexity: 7.895829677581787
#   mode: full
#   outcome: not_met
#   provenance:
#     code_sha256: 42a32d908802db5bb01ae0abe406eb8d6dc531c69cbeb244807dd983f2015637
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 716d0ced68ef
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
#     script_sha256: fd5af2e6577fcfd26f8930162e049049658423f4010a4133590efc639eec18db
#     source_sha256: becc096deed409b8fe3af2521f4101d3e076a1a1865157c1add0de606744c481
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 0
#   started_at: '2026-11-01T01:35:47.947823+00:00'
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
# id: id-0007
# x: 0.0
# y: 10.0
# ```
#
# The numeric answer is hidden. Identity reconstruction activates grouping; the generating rule is not a model input.
#
# ```{typst}
# //| label: fig-evidence-cluster-hidden-regression-regimes
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with obs, id, x, y. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("obs", kind: "root", children: (
#   node("id", type: "Cluster", width: 150pt,),
#   node("x", type: "Number"),
#   node("y", kind: "target", type: "Number",),
# )))
# ```
#
# ## Comparison
#
# Compare the number and usage of learned groups with the five generating rules.
#
# ## Result
#
# {{< evidence P018 status >}}
#
# The number settles near five, but that does not show the correct identities share a group. Training and validation reuse the same records; independent prediction and grouping quality remain untested.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P018 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P018 script >}}
#
# ### How it works
#
# The generator supplies 100 identities with 30 observations each. Its five
# regimes are `sin(x) + 10`, `cos(x) + 4`, `-2`, `0.5x - 8`, and `-0.3x - 14`,
# with small additive noise. The repeated identity lets the model associate
# behavior across changing `x` values.
#
# The ID field uses `rf.Mask(rate=0.5, reconstruct=True)` to engage the Cluster
# reconstruction loss. The numeric objective adds pressure to explain `y`.
# Cluster assignments belong to identities; row-varying `x` does not directly
# condition the query that reconstructs the identity itself.
#
# The model trains for 30 deterministic epochs. Training and validation
# currently use the same 3,000 observations. An internal-state callback measures
# the final five epochs.
#
# ### Remaining work
#
# Create independent train, validation, and test observations. Measure held-out
# regression against a marginal baseline and require permutation-invariant
# partition recovery with ARI ≥ 0.80. Add unique-identity and no-regime controls,
# and replace internal instrumentation with a public diagnostic when available.
#
# Repeat all behavioral and partition gates across three core seeds and at least
# ten calibration seeds before promoting this evidence.
#
# ### Complete experiment code
#

# %%
"""P018: track adaptive clusters across five hidden regression regimes.

Training and validation deliberately reuse observations. Count and usage
perplexity diagnose the mechanism; they do not prove partition recovery or
held-out prediction quality.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P018"


def records(seed: int) -> Iterator[dict]:
    """Repeat 100 identities thirty times across five hidden regression regimes."""
    rng = np.random.default_rng(seed)
    functions = (
        lambda x: np.sin(x) + 10.0,
        lambda x: np.cos(x) + 4.0,
        lambda x: np.zeros_like(x) - 2.0,
        lambda x: 0.5 * x - 8.0,
        lambda x: -0.3 * x - 14.0,
    )
    id_to_cluster = rng.integers(0, 5, size=100)
    identities = np.repeat(np.arange(100), 30)
    clusters = id_to_cluster[identities]
    x = rng.uniform(0.0, 10.0, size=3000) + rng.normal(0.0, 0.05, size=3000)
    y = np.empty(3000)
    for cluster, function in enumerate(functions):
        selected = clusters == cluster
        y[selected] = function(x[selected]) + rng.normal(0.0, 0.05, size=int(selected.sum()))
    for index in rng.permutation(3000):
        yield {"id": f"id-{identities[index]:04d}", "x": float(x[index]), "y": float(y[index])}


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
        id=rf.Cluster(bounds=(3, 15), mask=rf.Mask(rate=0.5, reconstruct=True)),
        x=rf.Number,
        y=rf.Number(mask=True),
    )
    source = partial(records, seed=seed + 42)
    data = rf.SyntheticDataModule(train=source, validate=source, seed=seed)
    trajectory = Trajectory(rf.Address("id"))
    model.fit(
        data,
        batch_size=128,
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
