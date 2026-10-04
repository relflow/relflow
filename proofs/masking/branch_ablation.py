# %% [markdown]
# ---
# title: Can it recover when a whole branch is hidden?
# categories:
# - Dynamic masking
# proof-id: P053
# description: Provide two redundant measurements of the same signal and sometimes hide an entire branch.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# proof-readout:
# - label: First view hidden
#   metric:
#   - patterns
#   - hiddenvisible
#   - nrmse
#   format: error
# - label: Second view hidden
#   metric:
#   - patterns
#   - visiblehidden
#   - nrmse
#   format: error
# - label: Both views hidden
#   metric:
#   - patterns
#   - hiddenhidden
#   - nrmse
#   format: error
# ---
#
# ## Example
#
# ```yaml
# views:
#   - {reading: 0.4, backup: 0.9, hidden: true, hide_backup: false}
#   - {reading: 0.4, backup: 0.9, hidden: false, hide_backup: true}
# target: 0.4
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-proof-branch-ablation
# //| fig-cap: "Amber cards are hidden prediction targets. hidden skips a whole view and its descendants."
# //| fig-alt: "Model tree with record, views, reading, backup, target. Amber cards are hidden prediction targets. hidden skips a whole view and its descendants."
# #tree(node("record", kind: "root", children: (
#   node("views", kind: "branch", repeated: true, body: [Skip when `hidden`], children: (
#     node("reading", type: "Number"),
#     node("backup", type: "Number", body: [Also skip when `hide_backup`]),
#   )),
#   node("target", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Test each branch alone, both visible, and both hidden. Change hidden descendant values to check they cannot affect the answer.
#
# ## Result
#
# {{< proof P053 status >}}
#
# Either view supports prediction. When both are hidden, the model falls back near the constant baseline; the test shows redundancy, rather than causal attribution.
#
# <details class="proof-details">
# <summary>Experiment details and code</summary>
#
# {{< proof P053 evidence >}}
#
# ### Run this experiment
#
# {{< proof P053 script >}}
#
# ### Data and model
#
# Each independent record draws a latent `z ~ Uniform(-1, 1)`. Both views expose
# `reading = z` and `backup = 2*z + 0.1`; the hidden target is `1.5*z - 0.2`.
# The latent variable is never a separate input. Independent branch selectors
# produce all four visibility patterns, independently of `z`. Leaf selectors
# also hide each backup with probability 0.5. Train/validation/test use
# 4,096/512/2,048 independent records.
#
# ### Training and controls
#
# The model uses the [xs preset](../../core-concepts/model-tree.qmd#choose-a-size).
#
# Fit for 500 AdamW updates at learning rate 0.002. On the same test observations,
# set each of the four branch-mask patterns deterministically. At least one
# visible view must give nRMSE below 0.30; no visible views must give nRMSE above
# 0.85. Poison all skipped descendants and the supervised target: prediction
# drift must stay below 1e-5. Permuting visible views between records must remove
# accuracy. The encoded presence bits must equal the union of ancestor and
# local skip selectors. Unselected backups are explicitly left visible when
# their branch is available.
#
# ### Complete experiment code
#

# %%
"""P053: redundant-view learning under composed branch and leaf skip policies."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P053"
BUDGET = 500


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        value = float(rng.uniform(-1.0, 1.0))
        yield {
            "views": [
                {
                    "reading": value,
                    "backup": 2 * value + 0.1,
                    "hidden": bool(rng.random() < 0.5),
                    "hide_backup": bool(rng.random() < 0.5),
                }
                for _ in range(2)
            ],
            "target": 1.5 * value - 0.2,
        }


def build() -> rf.Model:
    return rf.Model.xs(
        batch_size=64,
        views=rf.Branch(
            length=2,
            mask=rf.Mask(query="hidden", skip=True, dropout=False),
            reading=rf.Number,
            backup=rf.Number(mask=rf.Mask(query="hide_backup", skip=True, dropout=False)),
        ),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["/target"]["content"] for row in output], dtype=np.float64)


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    model.optimizer = rf.adamw(learning_rate=0.002)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, datamodule=data)
    model.eval()
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    metrics = {"steps": trainer.global_step, "baseline_rmse": baseline, "patterns": {}}
    checks = {}
    for pattern in ((False, False), (True, False), (False, True), (True, True)):
        name = "".join("hidden" if hidden else "visible" for hidden in pattern)
        selected = [
            {
                **row,
                "views": [
                    {**view, "hidden": hidden, "hide_backup": False}
                    for view, hidden in zip(row["views"], pattern, strict=True)
                ],
            }
            for row in test
        ]
        predicted = prediction(model, selected)
        poisoned = [
            {
                "target": 1000.0,
                "views": [
                    {**view, "reading": 1000.0, "backup": -1000.0} if view["hidden"] else view for view in row["views"]
                ],
            }
            for row in selected
        ]
        measured = {
            "nrmse": float(np.sqrt(np.mean((predicted - actual) ** 2))) / baseline,
            "hidden_value_drift": float(np.max(np.abs(predicted - prediction(model, poisoned)))),
        }
        metrics["patterns"][name] = measured
        checks[f"{name}: expected information boundary"] = (
            measured["nrmse"] > 0.85 if all(pattern) else measured["nrmse"] < 0.30
        )
        checks[f"{name}: skipped values cannot affect predictions"] = measured["hidden_value_drift"] < 1e-5
    order = np.random.default_rng(seed + 4).permutation(len(test))
    shuffled = [
        {
            "target": row["target"],
            "views": [{**view, "hidden": False, "hide_backup": False} for view in test[index]["views"]],
        }
        for row, index in zip(test, order, strict=True)
    ]
    metrics["shuffled_nrmse"] = float(np.sqrt(np.mean((prediction(model, shuffled) - actual) ** 2))) / baseline
    checks["Shuffling available evidence removes accuracy"] = metrics["shuffled_nrmse"] > 0.85
    fields = model.encode(pa.Table.from_pylist(test), strata="predict")
    for name in ("reading", "backup"):
        expected = np.asarray(
            [
                [not (view["hidden"] or (name == "backup" and view["hide_backup"])) for view in row["views"]]
                for row in test
            ]
        )
        present = fields[f"/views/{name}"].present.cpu().numpy().reshape(expected.shape)
        checks[f"{name}: ancestor and local skips compose"] = bool(np.array_equal(present, expected))
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5301)

# %% [markdown]
# </details>
