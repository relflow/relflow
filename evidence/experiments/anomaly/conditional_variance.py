# %% [markdown]
# ---
# title: Do legitimate noisy groups trigger too many anomaly warnings?
# categories:
# - Anomaly
# description: Predict an expected response in quiet and noisy groups, then compare
#   global and group-specific residual thresholds.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Conditional anomaly ranking
#   metric:
#   - conditional
#   - auc
#   format: auc
# - label: Precision at 1% prevalence
#   metric:
#   - conditional
#   - precision_at_one_percent
#   format: percent
# evidence-group: Anomaly
# ---
#
# The same-sized prediction error can be extraordinary in a quiet group and ordinary in a noisy one. A
# useful anomaly warning should account for that difference instead of systematically over-alerting on
# legitimate noisy records.
#
# ## Example
#
# ```yaml
# group: noisy
# x: 0.5
# answer: 3.4
# ```
#
# Predict an expected response in quiet and noisy groups, then compare global and group-specific residual
# thresholds. The example is illustrative; its answer is hidden during scoring.
#
# ```{typst}
# //| label: fig-evidence-p096
# //| fig-cap: "Visible fields supply evidence; prediction targets are hidden."
# //| fig-alt: "Predict an expected response in quiet and noisy groups, then compare global and group-specific residual thresholds."
# #tree(node("record",kind:"root",children:(node("group",type:"Enum"),node("x",type:"Number"),node("answer",kind:"target",type:"Number"))))
# ```
#
# ## Comparison
#
# Train one context-aware predictor, then compare a global residual threshold and a validation-only
# threshold for each group. A marginal-only baseline ignores context. Anomalies permute complete response
# values across test contexts, preserving the response distribution exactly.
#
# ## Result
#
# {{< evidence P096 status >}}
#
# No result is claimed before a full run is recorded. The checks below are provisional hypotheses, not
# established capabilities. Lower error and proper loss are better; larger accuracy is better. Controls and
# each evaluation condition remain separate.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P096 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P096 script >}}
#
# ### Process, splits, and interpretation
#
# Balanced groups have response 2x-2 or 2x+2 with Gaussian noise standard deviation 0.1 or 0.8. In the
# illustration, the noisy group expects3.0 at x=0.5, so3.4 is a plausible legitimate value. Independent
# training/validation/test streams contain4096/2048/8192 rows. A500-update predictor is shared by both
# residual scoring policies. Global and group thresholds are their validation 95th percentiles; no test
# labels choose thresholds. Report legitimate false positives separately for each group, detection recall,
# AUROC, and expected precision at 1% anomaly prevalence. Conditional scores divide by their own validation
# threshold. The marginal control has identical clean/anomaly score distributions by construction. High
# recall without useful precision is not sufficient, and heterogeneous Gaussian noise does not cover
# arbitrary uncertainty or new unseen groups.
#
# ### Complete experiment code
# %%
"""P096: compare anomaly thresholds when legitimate groups have unequal noise."""

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 500
GROUPS = ("quiet", "noisy")
SIGMA = {"quiet": 0.1, "noisy": 0.8}


def records(rows: int, seed: int):
    rng = np.random.default_rng(seed)
    groups = np.resize(np.arange(2), rows)
    rng.shuffle(groups)
    for group, x, z in zip(groups, rng.uniform(-1.0, 1.0, rows), rng.normal(size=rows), strict=True):
        name = GROUPS[group]
        yield {"group": name, "x": float(x), "answer": float(2 * x + (-2.0, 2.0)[group] + SIGMA[name] * z)}


def build() -> rf.Model:
    return rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        group=rf.Enum(values=GROUPS),
        x=rf.Number,
        answer=rf.Number(mask=True, objective="mse"),
    )


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = pl.DataFrame(
        model.predict(pa.Table.from_pylist([{"group": row["group"], "x": row["x"]} for row in rows]))
    )["predictions"].to_list()
    return np.asarray([row["answer"]["content"] for row in output])


def auc(normal: np.ndarray, anomalous: np.ndarray) -> float:
    ordered = np.sort(normal)
    return float(
        np.mean(
            (np.searchsorted(ordered, anomalous, "left") + np.searchsorted(ordered, anomalous, "right"))
            / (2 * len(ordered))
        )
    )


def operating(normal: np.ndarray, anomalous: np.ndarray, groups: np.ndarray) -> dict:
    false_positive = float(np.mean(normal > 1.0))
    recall = float(np.mean(anomalous > 1.0))
    prevalence = 0.01
    denominator = prevalence * recall + (1 - prevalence) * false_positive
    return {
        "auc": auc(normal, anomalous),
        "false_positive_rate": false_positive,
        "recall": recall,
        "precision_at_one_percent": float(prevalence * recall / denominator) if denominator else 0.0,
        "false_positive_by_group": {name: float(np.mean(normal[groups == name] > 1.0)) for name in GROUPS},
    }


def fit(model: rf.Model, train: list[dict], validation: list[dict], seed: int, budget: int, accelerator: str) -> int:
    """Fit only the supplied training rows with a fixed optimizer/update budget."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validation),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_steps=budget,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        limit_val_batches=1,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P096", seed=9601)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    train = list(records(4096, seed + 1))
    validation = list(records(2048, seed + 2))
    test = list(records(8192, seed + 3))
    budget = BUDGET if steps is None else min(steps, BUDGET)
    used = fit(model, train, validation, seed, budget, accelerator)
    validation_residual = np.abs(predict(model, validation) - np.asarray([row["answer"] for row in validation]))
    valid_groups = np.asarray([row["group"] for row in validation])
    groups = np.asarray([row["group"] for row in test])
    global_threshold = max(float(np.quantile(validation_residual, 0.95)), 1e-8)
    thresholds = {
        name: max(float(np.quantile(validation_residual[valid_groups == name], 0.95)), 1e-8) for name in GROUPS
    }
    prediction = predict(model, test)
    actual = np.asarray([row["answer"] for row in test])
    broken = actual[np.random.default_rng(seed + 4).permutation(len(test))]
    clean = np.abs(prediction - actual)
    anomalous = np.abs(prediction - broken)
    marginal_mean = float(np.mean([row["answer"] for row in train]))
    marginal_threshold = max(
        float(np.quantile(np.abs(np.asarray([row["answer"] for row in validation]) - marginal_mean), 0.95)), 1e-8
    )
    scale = np.asarray([thresholds[name] for name in groups])
    global_score = operating(clean / global_threshold, anomalous / global_threshold, groups)
    conditional = operating(clean / scale, anomalous / scale, groups)
    marginal = operating(
        np.abs(actual - marginal_mean) / marginal_threshold, np.abs(broken - marginal_mean) / marginal_threshold, groups
    )
    metrics = {
        "updates": used,
        "global_threshold": global_threshold,
        "conditional_thresholds": thresholds,
        "global": global_score,
        "conditional": conditional,
        "marginal": marginal,
        "prediction_nrmse": float(np.sqrt(np.mean(clean**2)) / np.std(actual)),
        "anomaly_prevalence": 0.01,
        "target_marginals_exactly_preserved": bool(np.array_equal(np.sort(actual), np.sort(broken))),
    }
    checks = {
        "Permutation preserves the target marginal exactly": metrics["target_marginals_exactly_preserved"],
        "Conditional predictor beats marginal standard deviation": metrics["prediction_nrmse"] < 0.45,
        "Conditional anomaly AUROC reaches 0.75": conditional["auc"] >= 0.75,
        "Marginal-only anomaly AUROC stays at 0.50": abs(marginal["auc"] - 0.5) < 1e-6,
    }
    for name in GROUPS:
        checks[f"Conditional threshold keeps {name} false positives below 0.10"] = (
            conditional["false_positive_by_group"][name] <= 0.10
        )
    checks["All global and conditional operating points are finite"] = all(
        np.isfinite(item[key])
        for item in (global_score, conditional, marginal)
        for key in ("auc", "false_positive_rate", "recall", "precision_at_one_percent")
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
