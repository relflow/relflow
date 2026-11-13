# %% [markdown]
# ---
# title: "Do typed profiles distinguish population changes from matched controls?"
# categories: ["Input distributions"]
# description: "Probe label, Boolean, set, vector, calendar, and nested null-rate changes while preserving a matched unshifted population."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-requires:
# - "An unreleased relflow wheel exporting Drift and model.distributions"
# evidence-group: Input distributions
# evidence-readout:
# - label: "Typed shift detection rate"
#   metric: [detection_rate]
#   format: number
# - label: "Same-population warnings"
#   metric: [control_warnings]
#   format: number
# ---
#
# Probe label, Boolean, set, vector, calendar, and nested null-rate changes while preserving a matched unshifted population.
#
# These are monitoring diagnostics, not a predictive-accuracy claim. The three
# predeclared seed trials run a short real training loop before evaluation.
#
# ## Example
#
# ```yaml
# merchant: merchant-2
# flag: false
# tags: [tag-1, tag-4]
# vector: [0.3, -0.8]
# happened: "2026-03-15"
# events: [{amount: 0.4}, {amount: null}]
# target: true
# ```
#
# The displayed observation illustrates the generator, rather than a measured
# prediction. The Boolean target is hidden during prediction; it exercises a real
# training loop but is not scored as evidence of learning.
#
# ```{typst}
# //| label: fig-evidence-p102
# //| fig-cap: "Visible inputs supply profiles; the target is hidden during prediction."
# //| fig-alt: "A record contains visible typed inputs and a hidden Boolean target."
# #tree(node("record", kind: "root", children: (
#   node("merchant", type: "Category"),
#   node("flag", type: "Boolean"),
#   node("tags", type: "Set"),
#   node("vector", type: "Vector", detail: "2 components"),
#   node("happened", type: "Period", detail: "month_of_year"),
#   node("events", kind: "branch", detail: "length <= 8", children: (node("amount", type: "Number"),)),
#   node("target", kind: "target", type: "Boolean"),
# )))
# ```
#
# ## Comparison
#
# One arm changes one declared typed feature: novel merchants, all-true flags, novel set members, eight-member sets instead of two, eight-times-larger vectors, January instead of a uniform calendar, or all-null nested amounts. A same-population control uses independent draws. The original random draws, collection lengths, and hidden Boolean labels remain matched across interventions.
#
# ## Result
#
# {{< evidence P102 status >}}
#
# Detection rates approach one when the declared shifts warn. False-positive
# counts should remain zero for the independent same-population controls.
# Missing records are not successful measurements. A smoke run establishes
# execution only, and these new research studies do not replace paired core CI.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P102 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P102 script >}}
#
# ### Population, controls, and limits
#
# Three fixed seeds each use 4,096 training, 1,024 validation, and 2,048 held-out records, with 16 AdamW steps at batch size 128 and learning rate 0.001. Events have one to eight observations and a 5% null rate before intervention. Four merchants and eight set labels form the familiar vocabulary. Each probe starts a fresh prediction window, preserving the training reference. Evaluation must not change model tensors, merchants, or set members. Text semantics, vector rotations preserving mean and norm, unseen joint relationships, and calibrated population-wide false-positive rates are outside this study.
#
# ### Complete experiment code

# %%
"""P102: typed drift features and matched negative controls on independent nested observations."""

import logging
from datetime import datetime

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


CONDITIONS = {
    "same_population": None,
    "labels": ("/merchant", "label"),
    "boolean": ("/flag", "value"),
    "members": ("/tags", "member"),
    "set_size": ("/tags", "size"),
    "vector": ("/vector", "norm"),
    "calendar": ("/happened", "month_of_year.cos"),
    "nulls": ("/events/amount", "state"),
}


def records(*, rows, seed, condition="same_population"):
    rng = np.random.default_rng(seed)
    for index in range(rows):
        merchant = int(rng.integers(0, 4))
        members = rng.choice(8, 2, replace=False)
        vector = rng.normal(size=2)
        month = int(rng.integers(1, 13))
        flag = bool(rng.random() < 0.5)
        events = [
            {"amount": None if rng.random() < 0.05 else float(rng.normal())} for _ in range(int(rng.integers(1, 9)))
        ]
        target = bool(rng.random() < 0.5)
        yield {
            "merchant": f"merchant-{merchant + (20 if condition == 'labels' else 0)}",
            "flag": True if condition == "boolean" else flag,
            "tags": [f"tag-{int(member) + (20 if condition == 'members' else 0)}" for member in members]
            if condition != "set_size"
            else [f"tag-{member}" for member in range(8)],
            "vector": (vector * (8 if condition == "vector" else 1)).tolist(),
            "happened": datetime(2026, 1 if condition == "calendar" else month, 15),
            "events": [{"amount": None} for _ in events] if condition == "nulls" else events,
            "target": target,
        }


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        drift=True,
        d_model=16,
        n_layers=1,
        n_heads=2,
        attention=None,
        merchant=rf.Category,
        flag=rf.Boolean,
        tags=rf.Set,
        vector=rf.Vector(n_dim=2),
        happened=rf.Period("month_of_year"),
        events=rf.Branch(length=8, attention=None, amount=rf.Number),
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
    vocabulary = rf.Category.vocabulary(model, "/merchant")
    members = rf.Set.vocabulary(model, "/tags")
    capture = Capture()
    core = logging.getLogger("relflow")
    core.addHandler(capture)
    arms, checks = {}, {}
    try:
        for condition, expected in CONDITIONS.items():
            model.distributions.clear(strata="predict")
            before = len(capture.records)
            source = pa.Table.from_pylist(list(records(rows=TEST_ROWS, seed=seed + 3, condition=condition)))
            model.encode(source.drop(["target"]))
            warnings = capture.records[before:]
            detected = expected is not None and any((w["address"], w["feature"]) == expected for w in warnings)
            arms[condition] = {"warnings": warnings, "detected": detected, "profiles": model.distributions.summary()}
            checks[
                f"{condition}: expected typed feature detected" if expected else "same population: no drift warnings"
            ] = detected if expected else not warnings
        checks["evaluation cannot admit merchants or set members"] = (
            rf.Category.vocabulary(model, "/merchant") == vocabulary and rf.Set.vocabulary(model, "/tags") == members
        )
        checks["evaluation leaves learned tensor state unchanged"] = all(
            torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()
        )
        checks["missing prediction answers are expected"] = "predict" not in model.distributions.summary()["/target"]
    finally:
        core.removeHandler(capture)
    return {"seed": seed, "steps": model.trainer.global_step, "arms": arms}, checks


@experiment("P102", seed=10201)
def run(seed, steps, accelerator):
    trials, checks = [], {}
    for offset in SEED_OFFSETS:
        measured, criteria = trial(seed + offset, steps, accelerator)
        trials.append(measured)
        checks.update({f"seed={seed + offset}/{name}": value for name, value in criteria.items()})
    changed = [arm for trial in trials for condition, arm in trial["arms"].items() if condition != "same_population"]
    return {
        "trials": trials,
        "seed_panel": [seed + offset for offset in SEED_OFFSETS],
        "budget": BUDGET,
        "train_rows": TRAIN_ROWS,
        "validate_rows": VALIDATE_ROWS,
        "test_rows": TEST_ROWS,
        "detection_rate": sum(arm["detected"] for arm in changed) / len(changed),
        "control_warnings": sum(len(trial["arms"]["same_population"]["warnings"]) for trial in trials),
    }, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
