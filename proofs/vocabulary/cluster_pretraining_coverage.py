# %% [markdown]
# ---
# title: Cluster Pretraining Coverage
# categories: [Vocabulary]
# proof-id: P065
# description: Unknown reconstruction targets must affect coverage without inventing uniform-label supervision or hiding denominator changes.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P065 status >}}
#
# ## Insights
#
# A pretraining validation score combines two questions: could this vocabulary
# express the target, and did context recover it? Separating those questions
# prevents 90% unavailable targets from masquerading as either good accuracy
# or a demand for uniform predictions. Unknown answers remain impossible to
# name; excluding their undefined categorical loss is not a modeling success.
#
# Evaluation uses fixed query-selected masks. A second independent random
# mask would otherwise confound vocabulary coverage with mask sampling noise.

# %%
"""P065: report stable reconstruction quality alongside pretraining coverage."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
from reporting import report

import relflow as rf

PROOF_ID = "P065"
BUDGET = 800
LABELS = tuple(f"class-{index}" for index in range(8))

# %% [markdown]
# ## Examples
#
# ```yaml
# x: cue-2
# code: class-2
# hide: true
# ```
#
# The familiar categorical cue identifies this selected known target.
#
# ```yaml
# x: cue-2
# code: novel-2
# hide: true
# ```
#
# Replacing only the hidden answer leaves identical visible evidence. The
# closed vocabulary cannot name that answer or infer that its spelling changed.
#
# ```yaml
# x: cue-2
# code: novel-2
# hide: false
# ```
#
# Unselected inputs do not enter reconstruction metrics. In particular,
# coverage means coverage of selected valued targets, not every source value.
#
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-pretraining-coverage
# //| fig-cap: "A fixed source query selects reconstruction coordinates, while a familiar categorical cue identifies the answer."
# //| fig-alt: "A root contains Category x and Cluster code reconstructed where the source hide flag is true."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Category", detail: "Familiar class cue"),
#   node("code", type: "Cluster", detail: "Reconstruct where hide is true"),
# )))
# ```


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for index, label in enumerate(rng.integers(0, len(LABELS), rows)):
        yield {"x": f"cue-{label}", "code": LABELS[label], "hide": index % 2 == 0}


def build() -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        dropout=0.0,
        x=rf.Category(p_unavailable=0.0),
        code=rf.Cluster(n_clusters=8, p_unavailable=1.0, mask=rf.Mask(query="hide", reconstruct=True)),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def evaluate(trainer: lit.Trainer, model: rf.Model, rows: list[dict]) -> dict[str, float]:
    data = rf.SyntheticDataModule(model=model, validate=lambda: iter(rows), seed=0)
    result = trainer.validate(model, datamodule=data, verbose=False)[0]
    prefix = ".code/validate."
    return {name.removeprefix(prefix): float(value) for name, value in result.items() if name.startswith(prefix)}


# %% [markdown]
# ## Training and controls
#
# Train for 800 updates on 8,192 rows; storage grows to admit all eight labels.
# Evaluation selects exactly 1,024 of 2,048 independent records. Compare the
# original set with copies retaining 50%, 10%, or 0% of selected target names.
# All copies preserve the exact visible inputs and query-selected masks.
#
# Repeat the 10%-coverage case with unknown targets grouped into separate
# batches. Count-weighted epoch metrics must agree despite all-OOV batches.
# Scores with no known targets must be undefined, not zero reconstruction
# error. Naive accuracy over all selected valued targets counts OOV as wrong.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=8192, seed=seed + 1),
        validate=partial(records, rows=1024, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    original = list(records(rows=2048, seed=seed + 3))
    vocabulary = rf.Cluster.vocabulary(model, "/code")
    counts = tuple(model.nodes["/code"].embedder.counters["content"].counts.tolist())
    metrics, checks = {}, {}
    for keep in (1024, 512, 102, 0):
        selected = 0
        rows = []
        for row in original:
            unavailable = row["hide"] and selected >= keep
            rows.append({**row, "code": f"novel-{row['code']}" if unavailable else row["code"]})
            selected += int(row["hide"])
        measured = evaluate(trainer, model, rows)
        # Keep undefined quantities explicit without nonstandard JSON NaNs.
        metrics[f"known_{keep}"] = {name: value if np.isfinite(value) else None for name, value in measured.items()}
        checks[f"{keep}: exact known and unavailable target counts"] = (
            measured.get("targets.known") == keep and measured.get("targets.unavailable") == 1024 - keep
        )
        checks[f"{keep}: coverage matches selected targets"] = (
            abs(measured.get("coverage.content", -1) - keep / 1024) < 1e-7
        )
        if keep:
            checks[f"{keep}: known accuracy exceeds 0.95"] = measured.get("accuracy.content", -1) > 0.95
            checks[f"{keep}: all-target accuracy includes unavailable errors"] = (
                abs(measured.get("accuracy.all", -1) - measured.get("accuracy.content", 0) * keep / 1024) < 1e-7
            )
        else:
            checks["All-OOV conditional scores are undefined"] = all(
                name in measured and np.isnan(measured[name])
                for name in ("accuracy.content", "nll.content", "loss.content")
            )
            checks["All-OOV all-target accuracy is zero"] = measured.get("accuracy.all") == 0
        if keep == 102:
            rng = np.random.default_rng(seed + 4)
            permuted = [rows[index] for index in rng.permutation(len(rows))]
            regrouped = evaluate(trainer, model, permuted)
            names = (
                "targets.known",
                "targets.unavailable",
                "coverage.content",
                "accuracy.content",
                "accuracy.all",
                "nll.content",
                "loss.content",
            )
            drift = (
                max(abs(measured[name] - regrouped[name]) for name in names)
                if all(name in measured and name in regrouped for name in names)
                else None
            )
            metrics["batch_partition_drift"] = drift
            metrics["reported_content_loss_drift"] = abs(measured["loss.content"] - regrouped["loss.content"])
            checks["Rebatching preserves counts and scores"] = drift is not None and drift < 1e-5
    checks["Validation leaves mapping and exposures frozen"] = (
        rf.Cluster.vocabulary(model, "/code") == vocabulary
        and tuple(model.nodes["/code"].embedder.counters["content"].counts.tolist()) == counts
    )
    return metrics, checks


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P065 evidence >}}
#
# Development runs passed the coverage and rebatching checks but failed the
# 0.95 known-label accuracy gate. An earlier numerical-cue version and a
# categorical-cue run extended to 4,000 updates also failed that gate. These
# failures remain evidence, not a reason to lower the acceptance threshold.
# Stable bookkeeping and log-space balancing do not establish reliable
# eight-way label recovery through the latent cluster factorization.
# The final panel (CPU default and GPU default/7701/7702) passed coverage
# checks throughout, but only GPU seed 7702 met every learning gate.
#
# This fixed-vocabulary experiment isolates coverage and aggregation. Latent
# cluster confidence is not label confidence. The auxiliary balance objective
# still constrains cluster use; this proof does not establish calibration or
# recovery of a latent partition. Input unavailability is deliberately 1.0:
# it must not corrupt the query-selected reconstruction answers.
#
# All-OOV validation cannot supply label accuracy or label loss. Known targets
# must remain learnable even though 1,016 allocated label rows are unused. A growing
# training vocabulary still changes the covered validation population. Delayed
# worker admission, evolving class difficulty, and stochastic evaluation masks
# need separate diagnostics; these scores do not make all validation loss
# trajectories comparable or identify unknown labels from hidden answers.
#
# ## Reproduce
#
# {{< proof P065 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7651)
