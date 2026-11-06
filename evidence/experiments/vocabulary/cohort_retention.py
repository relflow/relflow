# %% [markdown]
# ---
# title: What is forgotten when old examples stop arriving?
# categories:
# - Vocabulary
# description: Predict a cohort-specific response and retain earlier cohorts through
#   sequential adaptation.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Joint reference error
#   metric:
#   - arms
#   - joint
#   - final_mean_nrmse
#   format: number
# - label: No-replay error
#   metric:
#   - arms
#   - no_replay
#   - final_mean_nrmse
#   format: number
# evidence-group: Vocabulary
# ---
#
# Adapting to new observations can improve the current cohort while damaging earlier predictions. Retention
# measured while every old example is rehearsed does not reveal this tradeoff.
#
# ## Example
#
# ```yaml
# cohort: A
# x: 0.25
# answer: -0.5
# ```
#
# Predict a cohort-specific response and retain earlier cohorts through sequential adaptation. The example
# is illustrative; its answer is hidden during scoring.
#
# ```{typst}
# //| label: fig-evidence-p086
# //| fig-cap: "Visible fields supply evidence; prediction targets are hidden."
# //| fig-alt: "Predict a cohort-specific response and retain earlier cohorts through sequential adaptation."
# #tree(node("record",kind:"root",children:(node("cohort",type:"Enum"),node("x",type:"Number"),node("answer",kind:"target",type:"Number"))))
# ```
#
# ## Comparison
#
# Compare no replay, 32 retained training rows per earlier cohort, full joint rehearsal, and an A-only
# continued-training control. Every phase receives the same update budget; every cohort has an independent
# frozen test set.
#
# ## Result
#
# {{< evidence P086 status >}}
#
# No result is claimed before a full run is recorded. The checks below are provisional hypotheses, not
# established capabilities. Lower error and proper loss are better; larger accuracy is better. Controls and
# each evaluation condition remain separate.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P086 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P086 script >}}
#
# ### Process, splits, and interpretation
#
# Cohorts A, B, and C share x in [-1,1] but have responses 2x-1, -2x+1, and 1.5x. Thus A at x=0.25 has
# answer -0.5, while B at the same x has answer 0.5. The visible cohort changes which rule applies. Cohort
# identity is a visible declared Enum. Each cohort has 2048 training, 512 validation, and 1024 held-out rows
# from separate seeds. Four arms begin from the same initialization and receive 400 updates in each of three
# phases. The replay sample uses only old training rows, never tests. Score all cohorts after every phase,
# including not-yet-trained cohorts as diagnostics. Positive controls must learn and retain known tasks; no
# perfect no-replay retention or guaranteed replay superiority is assumed. Statistics continue to follow the
# actual training stream, so reported forgetting includes the complete fitted model state.
#
# ### Complete experiment code
# %%
"""P086: score every prior cohort after each sequential adaptation phase."""

from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 400
COHORTS = ("A", "B", "C")


def equal(left, right) -> bool:
    """Compare complete model state, including non-tensor vocabulary metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and (
                bool(torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
                if left.is_floating_point() or left.is_complex()
                else torch.equal(left, right)
            )
        )
    if isinstance(left, np.ndarray):
        return (
            bool(np.array_equal(left, right, equal_nan=True))
            if left.dtype.kind in "fc"
            else bool(np.array_equal(left, right))
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and np.isnan(left):
        return bool(np.isnan(right))
    return bool(left == right)


def records(rows: int, seed: int, cohort: str):
    rng = np.random.default_rng(seed)
    index = COHORTS.index(cohort)
    for x in rng.uniform(-1.0, 1.0, rows):
        yield {"cohort": cohort, "x": float(x), "answer": float((2.0, -2.0, 1.5)[index] * x + (-1.0, 1.0, 0.0)[index])}


def build() -> rf.Model:
    return rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        cohort=rf.Enum(values=COHORTS),
        x=rf.Number,
        answer=rf.Number(mask=True, objective="mse"),
    )


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{"cohort": row["cohort"], "x": row["x"]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


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


@experiment("P086", seed=8601)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    budget = BUDGET if steps is None else min(steps, BUDGET)
    train = {name: list(records(2048, seed + 10 + i, name)) for i, name in enumerate(COHORTS)}
    validation = {name: list(records(512, seed + 20 + i, name)) for i, name in enumerate(COHORTS)}
    test = {name: list(records(1024, seed + 30 + i, name)) for i, name in enumerate(COHORTS)}
    scales = {name: float(np.std([row["answer"] for row in rows])) for name, rows in train.items()}
    metrics = {"arms": {}, "replay_rows_per_old_cohort": 32}
    checks = {}
    for arm in ("no_replay", "small_replay", "joint", "continued_A"):
        lit.seed_everything(seed, workers=True)
        model = build()
        history = []
        for phase, current in enumerate(COHORTS):
            seen = COHORTS[: phase + 1]
            if arm == "continued_A":
                fitting = train["A"]
            elif arm == "joint":
                fitting = [row for name in seen for row in train[name]]
            elif arm == "small_replay":
                fitting = train[current] + [row for name in seen[:-1] for row in train[name][:32]]
            else:
                fitting = train[current]
            used = fit(model, fitting, validation[current], seed + phase, budget, accelerator)
            before = deepcopy(model.state_dict())
            scores = {}
            for name in COHORTS:
                output = predict(model, test[name])
                actual = np.asarray([row["answer"] for row in test[name]])
                scores[name] = {"nrmse": float(np.sqrt(np.mean((output - actual) ** 2)) / scales[name])}
            unchanged = equal(before, model.state_dict())
            history.append({"phase": current, "updates": used, "training_rows": len(fitting), "scores": scores})
            checks[f"{arm} phase {current}: finite scores"] = bool(
                np.isfinite([row["nrmse"] for row in scores.values()]).all()
            )
            checks[f"{arm} phase {current}: evaluation preserves parameters"] = unchanged
            if arm != "continued_A":
                checks[f"{arm} phase {current}: current cohort learns below 0.50 nRMSE"] = (
                    scores[current]["nrmse"] < 0.50
                )
        forgetting = {
            name: float(
                history[-1]["scores"][name]["nrmse"] - min(item["scores"][name]["nrmse"] for item in history[i:])
            )
            for i, name in enumerate(COHORTS)
        }
        metrics["arms"][arm] = {
            "history": history,
            "forgetting": forgetting,
            "final_mean_nrmse": float(np.mean([row["nrmse"] for row in history[-1]["scores"].values()])),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    checks["Joint-data reference retains all cohorts below 0.35 nRMSE"] = all(
        score["nrmse"] < 0.35 for score in metrics["arms"]["joint"]["history"][-1]["scores"].values()
    )
    checks["Continued-A control retains A below 0.30 nRMSE"] = (
        metrics["arms"]["continued_A"]["history"][-1]["scores"]["A"]["nrmse"] < 0.30
    )
    metrics["replay_minus_no_replay_final_error"] = (
        metrics["arms"]["small_replay"]["final_mean_nrmse"] - metrics["arms"]["no_replay"]["final_mean_nrmse"]
    )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
