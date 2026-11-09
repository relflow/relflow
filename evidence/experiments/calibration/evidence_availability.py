# %% [markdown]
# ---
# title: "Does confidence stay appropriate when useful readings disappear?"
# categories: ["Calibration"]
# description: "Measure written probabilities against an exact visible-information oracle while missingness rates and subgroup frequencies change."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: "Matched probability error"
#   metric: ["conditions", "matched", "probability_rmse"]
#   format: number
# - label: "Missing-reading error"
#   metric: ["conditions", "missing", "probability_rmse"]
#   format: number
# - label: "Selection-shift error"
#   metric: ["conditions", "selection_shift", "probability_rmse"]
#   format: number
# evidence-group: Calibration
# ---
#
# Measure written probabilities against an exact visible-information oracle while missingness rates and
# subgroup frequencies change.
#
# ## Example
#
# ```yaml
# group: rare
# signal: 1.0
# hide_signal: true
# target: true
# ```
#
# The rare group has three possible latent signal values. Its outcome probabilities are 0.03, 0.40, and
# 0.90. When the reading is hidden at random, the appropriate probability is their average, about 0.443,
# rather than the hidden value's 0.90.
#
# A different selection policy hides low readings more often. Its missing-value probability is about
# 0.224. That is a different statistical problem, and a model is not expected to infer a new observation
# policy without evidence about it.
#
# ```{typst}
# //| label: fig-evidence-p089
# //| fig-cap: "The target is hidden. The diagram shows the informative representation; controls change only the declared condition."
# //| fig-alt: "Does confidence stay appropriate when useful readings disappear? Visible inputs lead to one hidden prediction target."
# #tree(node("record", kind: "root", children: (node("group", type: "Enum"), node("signal", type: "Number", detail: "Hidden by selector"), node("target", kind: "target", type: "Boolean"),)))
# ```
#
# ## Comparison
#
# Train with 20% random hiding, then score matched visibility, 80% random hiding, and a changed subgroup
# mix. A separate value-dependent selection shift is diagnostic. The exact finite-table oracle
# integrates only over values hidden from the model; a constant prior is the low-information reference.
#
# ## Result
#
# {{< evidence P089 status >}}
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P089 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P089 script >}}
#
# ### Population, controls, and scoring
#
# Generate 8,192 training, 1,024 validation, and 4,096 independent test rows with a 10% rare subgroup.
# All six group/signal cells have declared Bernoulli probabilities. One model receives 500 updates; no
# threshold or checkpoint is selected on test. The cohort-shift condition changes rare-group frequency
# to 40% while retaining conditional probabilities. Report proper expected loss and probability error
# separately for each group and visibility. Selection-shift quality is recorded without demanding
# universal robustness to an unannounced change in the true conditional law. P061 already studies
# unknown-ID calibration; this experiment changes observation availability instead.
#
# ### Complete experiment code

# %%
"""P089: probability quality under changed visibility and subgroup frequency."""

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 500
TRAIN_ROWS = 8192
TEST_ROWS = 4096
GROUPS = ("common", "rare")
PROBABILITY = np.asarray([[0.01, 0.08, 0.35], [0.03, 0.40, 0.90]])
POLICIES = {
    "training": np.asarray([0.2, 0.2, 0.2]),
    "missing": np.asarray([0.8, 0.8, 0.8]),
    "selection": np.asarray([0.8, 0.5, 0.1]),
}


def records(*, rows: int, seed: int, policy: str = "training", rare_fraction: float = 0.1) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    groups = (rng.random(rows) < rare_fraction).astype(int)
    signals = rng.integers(0, 3, rows)
    hidden = rng.random(rows) < POLICIES[policy][signals]
    labels = rng.random(rows) < PROBABILITY[groups, signals]
    for group, signal, missing, label in zip(groups, signals, hidden, labels, strict=True):
        yield {
            "group": GROUPS[int(group)],
            "signal": float(signal - 1),
            "hide_signal": bool(missing),
            "target": bool(label),
        }


def oracle(rows: list[dict], policy: str) -> np.ndarray:
    """Integrate hidden signal values under the actual observation policy."""
    missing = POLICIES[policy]
    hidden_probability = PROBABILITY @ (missing / missing.sum())
    return np.asarray(
        [
            hidden_probability[GROUPS.index(row["group"])]
            if row["hide_signal"]
            else PROBABILITY[GROUPS.index(row["group"]), int(row["signal"]) + 1]
            for row in rows
        ]
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "target"} for row in rows]
    output = pa.table(model.predict(pa.Table.from_pylist(visible), batch_size=256))["predictions"].to_pylist()
    return np.asarray([row["target"]["content"]["probability"] for row in output], dtype=np.float64)


def score(predicted: np.ndarray, expected: np.ndarray, rows: list[dict], baseline: float) -> dict:
    groups = np.asarray([row["group"] for row in rows])
    hidden = np.asarray([row["hide_signal"] for row in rows])
    clipped = np.clip(predicted, 1e-7, 1 - 1e-7)
    result = {
        "probability_rmse": float(np.sqrt(np.mean((predicted - expected) ** 2))),
        "constant_rmse": float(np.sqrt(np.mean((baseline - expected) ** 2))),
        "expected_nll": float(np.mean(-expected * np.log(clipped) - (1 - expected) * np.log1p(-clipped))),
        "oracle_expected_nll": float(np.mean(-expected * np.log(expected) - (1 - expected) * np.log1p(-expected))),
        "excess_expected_brier": float(np.mean((predicted - expected) ** 2)),
        "finite": bool(np.isfinite(predicted).all()),
        "strata": {},
    }
    result["excess_expected_nll"] = result["expected_nll"] - result["oracle_expected_nll"]
    for group in GROUPS:
        for visibility, mask in (("visible", ~hidden), ("missing", hidden)):
            selected = (groups == group) & mask
            result["strata"][f"{group}_{visibility}"] = {
                "rows": int(selected.sum()),
                "probability_rmse": float(np.sqrt(np.mean((predicted[selected] - expected[selected]) ** 2))),
            }
    return result


def fit(model: rf.Model, train: pa.Table, validate: pa.Table, seed: int, steps: int | None, accelerator: str) -> int:
    """Fit the declared fixed-data protocol without checkpoint selection."""
    data = rf.ArrowDataModule(train=train, validate=validate, num_workers=0, persistent_workers=False, pin_memory=False)
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.003,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
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
    model.eval()
    return int(model.trainer.global_step)


@experiment("P089", seed=8901)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        group=rf.Enum(values=GROUPS, p_unavailable=0.0),
        signal=rf.Number(mask=rf.Mask(query="hide_signal", skip=True, dropout=False)),
        target=rf.Boolean(mask=True),
    )
    train = list(records(rows=TRAIN_ROWS, seed=seed + 1))
    updates = fit(
        model,
        pa.Table.from_pylist(train),
        pa.Table.from_pylist(list(records(rows=1024, seed=seed + 2))),
        seed,
        steps,
        accelerator,
    )
    baseline = float(PROBABILITY[0].mean() * 0.9 + PROBABILITY[1].mean() * 0.1)
    metrics = {"steps": updates, "conditions": {}}
    checks = {"Full training budget retained": updates == BUDGET}
    for name, policy, rare_fraction in (
        ("matched", "training", 0.1),
        ("missing", "missing", 0.1),
        ("cohort_shift", "missing", 0.4),
        ("selection_shift", "selection", 0.1),
    ):
        rows = list(records(rows=TEST_ROWS, seed=seed + 3, policy=policy, rare_fraction=rare_fraction))
        expected = oracle(rows, policy)
        predicted = prediction(model, rows)
        measured = score(predicted, expected, rows, baseline)
        metrics["conditions"][name] = measured
        checks[f"{name}: finite probabilities"] = measured["finite"]
        checks[f"{name}: every visibility/group stratum represented"] = all(
            item["rows"] >= 10 for item in measured["strata"].values()
        )
        if name != "selection_shift":
            checks[f"{name}: probability RMSE below 0.15"] = measured["probability_rmse"] < 0.15
            checks[f"{name}: probabilities improve on fixed prior"] = (
                measured["probability_rmse"] < measured["constant_rmse"]
            )
            for stratum, item in measured["strata"].items():
                checks[f"{name}, {stratum}: probability RMSE below 0.20"] = item["probability_rmse"] < 0.20
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
