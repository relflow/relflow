# %% [markdown]
# ---
# title: Do additional reconstruction objectives interfere with the primary task?
# categories:
# - Representation
# description: Fit a small shared encoder to a reliable primary target and a noisy auxiliary
#   target, comparing summed and Jacobian objectives.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Summed primary error
#   metric:
#   - conditions
#   - '8.0'
#   - summed
#   - nrmse
#   - primary
#   format: number
# - label: Jacobian primary error
#   metric:
#   - conditions
#   - '8.0'
#   - jacobian
#   - nrmse
#   - primary
#   format: number
# evidence-group: Representation
# evidence-requires:
# - relflow[torchjd] with the TorchJD solver extra
# ---
#
# A shared encoder can receive objectives that compete for limited capacity. This research experiment
# measures the tradeoff against single-task references, rather than assuming that adding a target or
# enabling Jacobian descent must improve every task.
#
# ## Example
#
# ```yaml
# observations: [0.5, -0.5, 0.25, -0.25, 0.1, -0.1, 0.2, -0.2]
# primary: [0.5, -0.5, 0.25, -0.25]
# auxiliary: [0.1, -0.1, 0.2, -0.2]
# ```
#
# Fit a small shared encoder to a reliable primary target and a noisy auxiliary target, comparing summed and
# Jacobian objectives. The example is illustrative; its answer is hidden during scoring.
#
# ```{typst}
# //| label: fig-evidence-p091
# //| fig-cap: "Visible fields supply evidence; prediction targets are hidden."
# //| fig-alt: "Fit a small shared encoder to a reliable primary target and a noisy auxiliary target, comparing summed and Jacobian objectives."
# #tree(node("record",kind:"root",body:[Shared width-six encoder],children:(node("observations",type:"Vector(8)"),node("primary",kind:"target",type:"Vector(4)"),node("auxiliary",kind:"target",type:"Vector(4)"))))
# ```
#
# ## Comparison
#
# Compare primary-only, auxiliary-only, ordinary summed objectives, and jacobian=True. Repeat auxiliary
# weights one and eight without scaling raw target values, to change the actual objective mixture. Report
# each target separately.
#
# ## Result
#
# {{< evidence P091 status >}}
#
# No result is claimed before a full run is recorded. The checks below are provisional hypotheses, not
# established capabilities. Lower error and proper loss are better; larger accuracy is better. Controls and
# each evaluation condition remain separate.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P091 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P091 script >}}
#
# ### Process, splits, and interpretation
#
# Eight independent observed coordinates are uniform on[-1,1]. The primary target reconstructs the first
# four; the auxiliary target reconstructs the last four with Gaussian noise of standard deviation 0.3. The
# illustration shows zero auxiliary noise; both target vectors are hidden. A width-six projection cannot
# continuously preserve all eight independent coordinates, giving a real shared information bottleneck. All
# arms use a width-six, one-layer, two-head, ffnx-one encoder, 4,096/512/2,048 independent
# training/validation/test records, and 500 updates. Arms share initialization and observations but
# single-task heads intentionally omit the other objective. Field weights alter actual objectives rather
# than normalized source units. Gate only interpretable single-task learning and finite outcomes;
# regret/Pareto comparisons and runtime are diagnostics until independent calibration justifies a
# superiority claim. This separately provisioned research lane requires relflow[torchjd] and does not belong
# in the dependency-free core evidence catalog.
#
# ### Complete experiment code
# %%
"""P091: measure held-out objective interference and optional Jacobian aggregation."""

import time
from importlib.metadata import version

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 500


def records(rows: int, seed: int):
    rng = np.random.default_rng(seed)
    for observed, noise in zip(rng.uniform(-1.0, 1.0, (rows, 8)), rng.normal(0.0, 0.3, (rows, 4)), strict=True):
        yield {
            "observations": observed.tolist(),
            "primary": observed[:4].tolist(),
            "auxiliary": (observed[4:] + noise).tolist(),
        }


def build(arm: str, weight: float) -> rf.Model:
    fields = {"observations": rf.Vector(n_dim=8, n_heads=2)}
    if arm != "auxiliary_only":
        fields["primary"] = rf.Vector(n_dim=4, n_heads=2, mask=True, objective="l2")
    if arm != "primary_only":
        fields["auxiliary"] = rf.Vector(n_dim=4, n_heads=2, mask=True, objective="l2", weight=weight)
    return rf.Model(d_model=6, n_layers=1, n_heads=2, ffnx=1, dropout=0.0, jacobian=arm == "jacobian", fields=fields)


def score(model: rf.Model, rows: list[dict], arm: str) -> dict:
    output = pl.DataFrame(model.predict(pa.Table.from_pylist([{"observations": row["observations"]} for row in rows])))[
        "predictions"
    ].to_list()
    names = (
        ("auxiliary",)
        if arm == "auxiliary_only"
        else ("primary",)
        if arm == "primary_only"
        else ("primary", "auxiliary")
    )
    return {
        name: float(
            np.sqrt(
                np.mean(
                    (np.asarray([row[name]["content"] for row in output]) - np.asarray([row[name] for row in rows]))
                    ** 2
                )
            )
            / np.std([row[name] for row in rows])
        )
        for name in names
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


@experiment("P091", seed=9101)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    try:
        import torchjd
    except ImportError as error:
        raise RuntimeError(
            "P091 requires the supplied relflow wheel with the torchjd extra; provision it before selecting the research lane."
        ) from error
    train = list(records(4096, seed + 1))
    validation = list(records(512, seed + 2))
    test = list(records(2048, seed + 3))
    budget = BUDGET if steps is None else min(steps, BUDGET)
    metrics = {"conditions": {}, "torchjd_module": torchjd.__name__, "torchjd_version": version("torchjd")}
    checks = {}
    for weight in (1.0, 8.0):
        arms = {}
        for arm in ("primary_only", "auxiliary_only", "summed", "jacobian"):
            lit.seed_everything(seed, workers=True)
            model = build(arm, weight)
            start = time.monotonic()
            used = fit(model, train, validation, seed, budget, accelerator)
            result = score(model, test, arm)
            arms[arm] = {
                "nrmse": result,
                "updates": used,
                "duration_seconds": time.monotonic() - start,
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
            }
            checks[f"weight {weight} {arm}: finite held-out errors"] = bool(np.isfinite(list(result.values())).all())
        reference = arms["primary_only"]["nrmse"]["primary"]
        for arm in ("summed", "jacobian"):
            arms[arm]["primary_regret"] = arms[arm]["nrmse"]["primary"] - reference
        arms["jacobian_minus_summed_primary_error"] = (
            arms["jacobian"]["nrmse"]["primary"] - arms["summed"]["nrmse"]["primary"]
        )
        metrics["conditions"][str(weight)] = arms
        checks[f"weight {weight}: primary-only reference learns below 0.40 nRMSE"] = reference < 0.40
        checks[f"weight {weight}: auxiliary-only reference beats constant prediction"] = (
            arms["auxiliary_only"]["nrmse"]["auxiliary"] < 0.85
        )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
