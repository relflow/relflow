# %% [markdown]
# ---
# title: "Do warning policies and checkpoints preserve the monitoring contract?"
# categories: ["Input distributions"]
# description: "Apply the same numeric shift to monitored, tolerant, and disabled fields, then check sample guards, suppression, clearing, and checkpoint persistence."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-requires:
# - "An unreleased relflow wheel exporting Drift and model.distributions"
# evidence-group: Input distributions
# evidence-readout:
# - label: "Monitoring contracts met"
#   metric: [contracts_met]
#   format: number
# - label: "Monitoring contracts tested"
#   metric: [contracts_total]
#   format: number
# ---
#
# Apply the same numeric shift to monitored, tolerant, and disabled fields, then check sample guards, suppression, clearing, and checkpoint persistence.
#
# These are monitoring diagnostics, not a predictive-accuracy claim. The three
# predeclared seed trials run a short real training loop before evaluation.
#
# ## Example
#
# ```yaml
# monitored: 0.4
# tolerant: 0.4
# disabled: 0.4
# target: true
# ```
#
# The displayed observation illustrates the generator, rather than a measured
# prediction. The Boolean target is hidden during prediction; it exercises a real
# training loop but is not scored as evidence of learning.
#
# ```{typst}
# //| label: fig-evidence-p103
# //| fig-cap: "Visible inputs supply profiles; the target is hidden during prediction."
# //| fig-alt: "A record contains visible typed inputs and a hidden Boolean target."
# #tree(node("record", kind: "root", children: (
#   node("monitored", type: "Number", detail: "default drift policy"),
#   node("tolerant", type: "Number", detail: "threshold = 0.99"),
#   node("disabled", type: "Number", detail: "drift = false"),
#   node("target", kind: "target", type: "Boolean"),
# )))
# ```
#
# ## Comparison
#
# All three numeric inputs are identical. Each held-out value shifts by +8. Thirty-two observations should be quiet; the complete cohort should warn once only for the default field. Repeating the cohort should suppress duplicates, including after a checkpoint round trip. Clearing the comparison window should rearm exactly one warning without deleting the training reference.
#
# ## Result
#
# {{< evidence P103 status >}}
#
# Detection rates approach one when the declared shifts warn. False-positive
# counts should remain zero for the independent same-population controls.
# Missing records are not successful measurements. A smoke run establishes
# execution only, and these new research studies do not replace paired core CI.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P103 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P103 script >}}
#
# ### Population, controls, and limits
#
# Three fixed seeds each train for 16 AdamW steps, batch size 128, learning rate 0.001, from independent 4,096/1,024 training/validation records. The held-out cohort contains 2,048 rows. Threshold 0.99 is a declared tolerance for this finite cohort; it is not an unconditional disable switch. A fresh prediction window after clearing contains exactly 2,048 rows. Checkpoints must support Torch weights-only inspection. Monitoring is rank-local and cumulative; this study makes no distributed aggregation, predictive-accuracy, or runtime-overhead claim.
#
# ### Complete experiment code

# %%
"""P103: warning suppression, opt-out, sample guards, checkpoint persistence, and immutable learned state."""

import logging
from pathlib import Path
from tempfile import TemporaryDirectory

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


def records(*, rows, seed, shift=0):
    rng = np.random.default_rng(seed)
    for value, target in zip(rng.normal(size=rows), rng.random(rows) < 0.5, strict=True):
        value = float(value + shift)
        yield {"monitored": value, "tolerant": value, "disabled": value, "target": bool(target)}


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    model = rf.Model(
        drift=True,
        d_model=16,
        n_layers=1,
        n_heads=2,
        attention=None,
        monitored=rf.Number,
        tolerant=rf.Number(drift=rf.Drift(threshold=0.99)),
        disabled=rf.Number(drift=False),
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
    reference = model.distributions.summary()["/monitored"]["train"]
    capture = Capture()
    core = logging.getLogger("relflow")
    core.addHandler(capture)
    checks, windows = {}, {}
    changed = pa.Table.from_pylist(list(records(rows=TEST_ROWS, seed=seed + 3, shift=8))).drop(["target"])
    try:
        for name, source in (("too_small", changed.slice(0, 32)), ("full", changed), ("suppressed", changed)):
            before = len(capture.records)
            model.encode(source)
            windows[name] = capture.records[before:]
        checks["small cohorts remain quiet"] = not windows["too_small"]
        checks["full shifted cohort warns exactly once"] = (
            len(windows["full"]) == 1
            and windows["full"][0]["address"] == "/monitored"
            and windows["full"][0]["feature"] == "value"
        )
        checks["repeat observations suppress duplicate warnings"] = not windows["suppressed"]
        checks["disabled field has no profiles"] = "/disabled" not in model.distributions.profiles
        checks["tolerant field records metadata without warning"] = (
            "/tolerant" in model.distributions.profiles
            and not any(w["address"] == "/tolerant" for w in capture.records)
        )
        checks["evaluation preserves the training reference"] = (
            model.distributions.summary()["/monitored"]["train"] == reference
        )
        checks["evaluation leaves learned tensor state unchanged"] = all(
            torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()
        )
        with TemporaryDirectory(prefix="relflow-drift-evidence-") as temporary:
            path = Path(temporary) / "model.ckpt"
            model.save(path)
            checkpoint = torch.load(path, weights_only=True, map_location="cpu")
            loaded = rf.Model.load(path)
            checks["checkpoint metadata permits weights-only loading"] = "distributions" in checkpoint
            checks["checkpoint retains exact profiles and warning suppression"] = (
                loaded.distributions.state_dict() == model.distributions.state_dict()
            )
            before = len(capture.records)
            loaded.encode(changed)
            windows["loaded_suppressed"] = capture.records[before:]
            checks["loaded warning suppression persists"] = not windows["loaded_suppressed"]
            loaded.distributions.clear(strata="predict")
            before = len(capture.records)
            loaded.encode(changed)
            windows["cleared"] = capture.records[before:]
            checks["clearing comparison rearms exactly one warning"] = (
                len(windows["cleared"]) == 1 and windows["cleared"][0]["address"] == "/monitored"
            )
            checks["clearing comparison retains the training reference"] = (
                loaded.distributions.summary()["/monitored"]["train"] == reference
            )
        checks["missing supervised prediction target is quiet"] = not any(
            w["address"] == "/target" for w in capture.records
        )
    finally:
        core.removeHandler(capture)
    return {"seed": seed, "steps": model.trainer.global_step, "windows": windows, "reference": reference}, checks


@experiment("P103", seed=10301)
def run(seed, steps, accelerator):
    trials, checks = [], {}
    for offset in SEED_OFFSETS:
        measured, criteria = trial(seed + offset, steps, accelerator)
        trials.append(measured)
        checks.update({f"seed={seed + offset}/{name}": value for name, value in criteria.items()})
    return {
        "trials": trials,
        "seed_panel": [seed + offset for offset in SEED_OFFSETS],
        "budget": BUDGET,
        "train_rows": TRAIN_ROWS,
        "validate_rows": VALIDATE_ROWS,
        "test_rows": TEST_ROWS,
        "contracts_met": sum(checks.values()),
        "contracts_total": len(checks),
    }, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
