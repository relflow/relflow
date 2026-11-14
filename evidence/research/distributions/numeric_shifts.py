# %% [markdown]
# ---
# title: "Which numeric population shifts trigger warnings?"
# categories: ["Input distributions"]
# description: "Compare large location, spread, and shape changes with independent draws from the same Gaussian population."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-requires:
# - "An unreleased relflow wheel exporting Drift and model.distributions"
# evidence-group: Input distributions
# evidence-readout:
# - label: "Shift detection rate"
#   metric: [detection_rate]
#   format: number
# - label: "Same-population warnings"
#   metric: [control_warnings]
#   format: number
# ---
#
# Compare large location, spread, and shape changes with independent draws from the same Gaussian population.
#
# These are monitoring diagnostics, not a predictive-accuracy claim. The three
# predeclared seed trials run a short real training loop before evaluation.
#
# ## Example
#
# ```yaml
# amount: 0.4
# partner: 0.42
# target: true
# ```
#
# The displayed observation illustrates the generator, rather than a measured
# prediction. The Boolean target is hidden during prediction; it exercises a real
# training loop but is not scored as evidence of learning.
#
# ```{typst}
# //| label: fig-evidence-p101
# //| fig-cap: "Visible inputs supply profiles; the target is hidden during prediction."
# //| fig-alt: "A record contains visible typed inputs and a hidden Boolean target."
# #tree(node("record", kind: "root", children: (
#   node("amount", type: "Number"),
#   node("partner", type: "Number"),
#   node("target", kind: "target", type: "Boolean"),
# )))
# ```
#
# ## Comparison
#
# All arms share the same held-out Gaussian draws and nuisance target labels. Moving `amount` by +4, multiplying it by eight, or replacing it with a tight symmetric mixture near -1 and +1 should warn. Independent same-population draws should stay quiet. Flipping only the sign of `partner` changes correlation from nearly +1 to nearly -1 while preserving its marginal population; that boundary case should also stay quiet.
#
# ## Result
#
# {{< evidence P101 status >}}
#
# Detection rates approach one when the declared shifts warn. False-positive
# counts should remain zero for the independent same-population controls.
# Missing records are not successful measurements. A smoke run establishes
# execution only, and these new research studies do not replace paired core CI.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P101 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P101 script >}}
#
# ### Population, controls, and limits
#
# Each seed uses 4,096 training, 1,024 validation, and 2,048 held-out observations. Sixteen AdamW updates consume 2,048 training rows at batch size 128, learning rate 0.001, and weight decay 0.01. Each condition is checked independently in validation, test, and prediction with a cleared comparison window and the same training reference. Default Drift thresholds remain unchanged. The Boolean target is independent fair noise, so no predictive performance is claimed. Tensor parameters, normalizers, and counters must remain unchanged during evaluation. Three fixed seeds make this an initial sensitivity study rather than a general false-positive-rate guarantee.
#
# ### Complete experiment code

# %%
"""P101: numeric drift sensitivity, quiet controls, and a joint-distribution blind spot."""

import logging

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 16
TRAIN_ROWS = 4096
VALIDATE_ROWS = 1024
TEST_ROWS = 2048
SEED_OFFSETS = (0, 100, 200)


class Capture(logging.Handler):
    """Retain actual structured warnings without replacing logging behavior."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        context = getattr(record, "relflow_context", {})
        if context.get("component") == "drift":
            self.records.append(dict(context))


CONDITIONS = ("same_population", "location", "spread", "shape", "correlation")


def records(*, rows, seed, condition="same_population"):
    rng = np.random.default_rng(seed)
    values = rng.normal(0, 1, rows)
    noise = rng.normal(0, 0.05, rows)
    target = rng.random(rows) < 0.5
    for index, (value, error, answer) in enumerate(zip(values, noise, target, strict=True)):
        amount = value
        partner = value + error
        if condition == "location":
            amount += 4
        elif condition == "spread":
            amount *= 8
        elif condition == "shape":
            amount = (-1 if index % 2 == 0 else 1) + 0.02 * error
        elif condition == "correlation":
            partner = -value + error
        yield {"amount": float(amount), "partner": float(partner), "target": bool(answer)}


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        drift=True,
        d_model=16,
        n_layers=1,
        n_heads=2,
        attention=None,
        amount=rf.Number,
        partner=rf.Number,
        target=rf.Boolean(mask=True),
    )

    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(list(records(rows=TRAIN_ROWS, seed=seed + 1))),
        validate=pa.Table.from_pylist(list(records(rows=VALIDATE_ROWS, seed=seed + 2))),
        seed=seed,
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=0.001,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=BUDGET if steps is None else min(steps, BUDGET),
        limit_train_batches=16,
        limit_val_batches=1,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()

    frozen = {name: value.clone() for name, value in model.state_dict().items() if isinstance(value, torch.Tensor)}
    capture = Capture()
    core = logging.getLogger("relflow")
    core.addHandler(capture)
    arms = {}
    checks = {}
    try:
        for split in ("validate", "test", "predict"):
            for condition in CONDITIONS:
                model.distributions.clear(strata=split)
                before = len(capture.records)
                source = pa.Table.from_pylist(list(records(rows=TEST_ROWS, seed=seed + 3, condition=condition)))
                model.encode(source.drop(["target"]) if split == "predict" else source, strata=split)
                warnings = capture.records[before:]
                reference = model.distributions.profiles["/amount"]["train"]["value"]
                current = model.distributions.profiles["/amount"][split]["value"]
                distance, allowance = reference.compare(current)
                correlation = float(np.corrcoef(source["amount"].to_numpy(), source["partner"].to_numpy())[0, 1])
                detected = any(w["address"] == "/amount" and w["feature"] == "value" for w in warnings)
                arms[f"{split}/{condition}"] = {
                    "distance": distance,
                    "allowance": allowance,
                    "correlation": correlation,
                    "warnings": warnings,
                    "detected": detected,
                    "reference_count": reference.count,
                    "count": current.count,
                    "reference_quantiles": reference.summary()["quantiles"],
                    "observed_quantiles": current.summary()["quantiles"],
                }
                if condition in ("location", "spread", "shape"):
                    checks[f"{split}/{condition}: amount shift detected"] = detected
                else:
                    checks[f"{split}/{condition}: marginal profiles remain quiet"] = len(warnings) == 0
                    if condition == "correlation":
                        checks[f"{split}: changed joint relationship is explicit"] = correlation < -0.98
        unchanged = all(torch.equal(value, model.state_dict()[name]) for name, value in frozen.items())
        checks["evaluation leaves learned tensor state unchanged"] = unchanged
    finally:
        core.removeHandler(capture)
    return {
        "seed": seed,
        "steps": model.trainer.global_step,
        "arms": arms,
        "learned_state_unchanged": unchanged,
    }, checks


@experiment("P101", seed=10101)
def run(seed, steps, accelerator):
    trials, checks = [], {}
    for offset in SEED_OFFSETS:
        metrics, measured = trial(seed + offset, steps, accelerator)
        trials.append(metrics)
        checks.update({f"seed={seed + offset}/{name}": value for name, value in measured.items()})
    changed = [
        a for t in trials for name, a in t["arms"].items() if name.split("/")[1] in ("location", "spread", "shape")
    ]
    controls = [a for t in trials for name, a in t["arms"].items() if name.endswith("same_population")]
    return {
        "trials": trials,
        "seed_panel": [seed + offset for offset in SEED_OFFSETS],
        "budget": BUDGET,
        "train_rows": TRAIN_ROWS,
        "validate_rows": VALIDATE_ROWS,
        "test_rows": TEST_ROWS,
        "detection_rate": sum(a["detected"] for a in changed) / len(changed),
        "control_warnings": sum(len(a["warnings"]) for a in controls),
    }, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
