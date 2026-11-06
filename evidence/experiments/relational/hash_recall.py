# %% [markdown]
# ---
# title: Can it look up a value using a new ID?
# categories:
# - Associative recall
# description: Mix source values and questions in different positions. Use unfamiliar IDs to connect each question
#   to its source, and judge the result over a panel of three seeds.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Matching IDs, median of three seeds
#   metric:
#   - median
#   - intact_nrmse
#   format: error
# - label: Changed IDs, median of three seeds
#   metric:
#   - median
#   - broken_nrmse
#   format: error
# - label: Matching IDs, worst seed
#   metric:
#   - worst
#   - intact_nrmse
#   format: error
# evidence-group: Relational
# ---
#
# ## Example
#
# ```yaml
# memory:
#   - {role: source, is_query: false, entity_id: K1, value: 0.4}
#   - {role: query, is_query: true, entity_id: K2, value: -0.8}
#   - {role: source, is_query: false, entity_id: K2, value: -0.8}
#   - {role: query, is_query: true, entity_id: K1, value: 0.4}
# ```
#
# Source values are visible. Query values are hidden wherever is_query is true.
#
# ```{typst}
# //| label: fig-evidence-hash-recall
# //| fig-cap: "Selector notes identify hidden values. Source values are visible; query values are hidden."
# //| fig-alt: "Model tree with association, memory, entity_id, role, is_query, value. Selector notes identify hidden values. Source values are visible; query values are hidden."
# #tree(node("association", kind: "root", width: 120pt, children: (
#   node("memory", kind: "branch", repeated: true, width: 120pt, children: (
#     node("entity_id", type: "Hash"),
#     node("role", type: "Category"),
#     node("is_query", type: "Boolean"),
#     node("value", type: "Number", width: 155pt, detail: "Source visible; query hidden"),
#   )),
# )))
# ```
#
# ## Comparison
#
# Change question IDs while retaining their original answers. This should break an ID-based lookup.
#
# ## Result
#
# {{< evidence P025 status >}}
#
# Three paired seeds train and score the same task, and the gates read their medians, so one lucky or unlucky draw cannot decide the outcome. Only two source values are tested; missing IDs and larger memories need separate checks.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P025 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P025 script >}}
#
# ### How it works
#
# Coordinate-local encoding binds a source identity to its value. Visible fields
# at each query coordinate condition its decoder query over retained memory.
# Source and query orders are independently shuffled. Rotating only query keys
# while retaining target values should destroy the original association.
# The [aligned control](aligned-position-control.html) shows what happens when
# position alone supplies an alternative answer path.
#
# Each seed of the panel draws its own records and initialization. The gates
# require the median seed to recall matching IDs within 0.25 normalized RMSE,
# to lose that recall when IDs change, and every seed to beat the constant
# baseline and to be hurt by the change; the worst seed is reported beside
# the medians. Training uses AdamW at learning rate 0.001 for 800 updates:
# at 0.003 roughly one seed in six finds the lookup and the rest settle on the
# constant predictor, while at 0.001 every seed of a twelve-seed calibration
# recalls.
#
# ### Remaining work
#
# Extend the panel to ten lightweight seeds before freezing the thresholds.
# Test complete record permutations, simultaneous key renaming, larger
# memories, absent or duplicate keys, padding, and Hash collisions. Use an
# explicit application lookup when the mapping must be exact.
#
# ### Complete experiment code
#

# %%
"""Retrieve unseen keys from source/query records shuffled into one branch.

Only query values are masked. Independent order and unseen identities prevent
positional copying and vocabulary memorization. Rotating query keys while
retaining targets tests whether successful recall actually uses identity. A
panel of three seeds decides, through its medians.

Run this file with --help for seed, training-budget, and reporting options.
"""

from __future__ import annotations

from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from itertools import repeat
from multiprocessing import get_context

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import experiment, report

PAIR_COUNT = 2
PANEL = (0, 100, 200)


def records(*, rows: int, seed: int, namespace: str, broken_identity: bool = False) -> Iterator[dict]:
    """Generate unseen, observation-local key/value associations."""
    rng = np.random.default_rng(seed)
    for row in range(rows):
        keys = [f"{namespace}-{row:06d}-{pair:02d}" for pair in range(PAIR_COUNT)]
        values = rng.uniform(-1.0, 1.0, size=PAIR_COUNT)
        query_keys = list(keys)
        if broken_identity:
            query_keys = query_keys[1:] + query_keys[:1]
        source_order = rng.permutation(PAIR_COUNT)
        query_order = rng.permutation(PAIR_COUNT)
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in source_order]
        target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in query_order]
        memory = [{**item, "role": "source", "is_query": False} for item in source]
        memory.extend({**item, "role": "query", "is_query": True} for item in target)
        rng.shuffle(memory)
        yield {"memory": memory}


def normalized_rmse(model: rf.Model, records: list[dict]) -> float:
    """Score only query coordinates; source values remain visible to the model."""
    output = model.predict(records).to_pylist()
    actual, predicted = [], []
    for row, result in zip(records, output, strict=True):
        for item, prediction in zip(row["memory"], result["predictions"]["memory"], strict=True):
            if item["is_query"]:
                actual.append(item["value"])
                predicted.append(prediction["value"]["content"])
    actual = np.asarray(actual)
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - actual)) / np.mean(np.square(actual))))


def trial(seed: int, steps: int | None, accelerator: str) -> dict:
    """Train and score one seed of the panel."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        d_model=64,
        n_layers=2,
        n_heads=4,
        reduction=None,
        memory=rf.Branch(
            length=2 * PAIR_COUNT,
            n_layers=2,
            reduction=None,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            role=rf.Category(p_unavailable=0.0),
            is_query=rf.Boolean,
            value=rf.Number(mask=rf.Mask(query="is_query", dropout=False, reconstruct=True), objective="mse"),
        ),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=seed + 1, namespace="train"),
        validate=partial(records, rows=512, seed=seed + 2, namespace="validate"),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.001),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_steps=800 if steps is None else min(steps, 800),
        max_epochs=-1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        check_val_every_n_epoch=5,
    )
    test = list(records(rows=1024, seed=seed + 3, namespace="test"))
    broken = list(records(rows=1024, seed=seed + 3, namespace="test", broken_identity=True))
    intact_nrmse = normalized_rmse(model, test)
    broken_nrmse = normalized_rmse(model, broken)
    return {
        "seed": seed,
        "steps": model.trainer.global_step,
        "intact_nrmse": intact_nrmse,
        "broken_nrmse": broken_nrmse,
        "corruption_gap": broken_nrmse - intact_nrmse,
    }


@experiment("P025", seed=17)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    if accelerator == "cpu":
        with ProcessPoolExecutor(
            max_workers=2,
            mp_context=get_context("spawn"),
            initializer=torch.set_num_threads,
            initargs=(torch.get_num_threads(),),
        ) as pool:
            trials = list(pool.map(trial, [seed + offset for offset in PANEL], repeat(steps), repeat(accelerator)))
    else:
        trials = [trial(seed + offset, steps, accelerator) for offset in PANEL]
    keys = ("intact_nrmse", "broken_nrmse", "corruption_gap")
    median = {key: float(np.median([result[key] for result in trials])) for key in keys}
    worst = {
        "intact_nrmse": max(result["intact_nrmse"] for result in trials),
        "corruption_gap": min(result["corruption_gap"] for result in trials),
    }
    metrics = {"trials": trials, "median": median, "worst": worst}
    checks = {
        "Finite errors on every seed": all(np.isfinite([result[key] for result in trials for key in keys])),
        "Every seed beats the constant baseline (intact nRMSE <= 0.90)": worst["intact_nrmse"] <= 0.90,
        "Median identity recall nRMSE <= 0.25": median["intact_nrmse"] <= 0.25,
        "Median broken identities nRMSE >= 0.80": median["broken_nrmse"] >= 0.80,
        "Median corruption gap >= 0.50": median["corruption_gap"] >= 0.50,
        "Every seed is hurt by changed IDs": worst["corruption_gap"] > 0.0,
    }
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
