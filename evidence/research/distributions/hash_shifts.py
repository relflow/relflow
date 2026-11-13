# %% [markdown]
# ---
# title: "Does Hash detect identity drift independently of training salts?"
# categories: ["Input distributions"]
# description: "Measure Hash identity-frequency and null-rate drift across validation, test, and prediction, independently of per-batch training salts."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-requires:
# - "An unreleased relflow wheel exporting Drift, Enum, Hash, Cluster, and model.distributions"
# evidence-group: Input distributions
# evidence-readout:
# - label: "Declared shifts detected"
#   metric: [detection_rate]
#   format: number
# - label: "Control warnings"
#   metric: [control_warnings]
#   format: number
# ---
#
# Measure Hash identity-frequency and null-rate drift across validation, test, and prediction, independently of per-batch training salts.
#
# ## Example
#
# ```yaml
# value: 1152921504606846978
# peer: 1152921504606846978
# target: true
# ```
#
# This is an illustrative generator observation, not a model prediction. The hidden
# Boolean target is independent fair noise. It exercises a real optimizer loop;
# these experiments claim monitoring behavior rather than predictive accuracy.
#
# ```{typst}
# //| label: fig-evidence-p105
# //| fig-cap: "Typed input frequencies are monitored; the Boolean target is hidden during prediction."
# //| fig-alt: "A record has typed identity inputs and a hidden Boolean prediction target."
# #tree(node("record", kind: "root", children: (
#   node("value", type: "Hash", detail: "2 hash lanes; integer IDs near 2^60"),
#   node("peer", type: "Hash", detail: "same IDs and salt"),
#   node("target", kind: "target", type: "Boolean"),
# )))
# ```
#
# ## Comparison
#
# Compare an independent same-population control with an all-one-identity frequency shift, a disjoint population offset by 64, and an all-null population. Two worker encodings of identical source rows use different training salts: tensor hashes must change, equal IDs across fields must stay equal, and profiling histograms must remain identical.
#
# Training builds the reference through 16 real AdamW updates. Validation, test,
# and prediction each receive their own independent held-out stream. Within a
# stratum, controls and interventions share latent draws and target labels.
# Each arm starts a fresh comparison window and must preserve the training profile
# and learned tensor state. A shifted arm must emit a real warning identifying
# the feature and stratum, with sample counts and distance exceeding its cutoff.
# An independent same-population control must emit no drift warning. Five
# successive 64-row frequency-shift batches additionally verify per-batch
# accumulation, the minimum-count guard, and one warning per feature/window.
#
# ## Result
#
# {{< evidence P105 status >}}
#
# The declared shift detection rate should be one and control warnings zero.
# These predeclared checks are provisional until a full-budget record exists.
# Smoke runs are retained separately and cannot establish capability.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P105 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P105 script >}}
#
# ### Population, budgets, and limits
#
# Every seed trial draws 4,096 source training rows, 1,024 independent validation
# rows, and 2,048 held-out rows per stratum and condition. Sixteen updates at batch
# size 128 consume 2,048 training exposures. The model uses width 16, one layer,
# two heads, disabled root attention, AdamW learning rate 0.001, weight decay 0.01,
# and float32 precision. There is no checkpoint selection or early stopping.
# Three fixed seeds use offsets 0, 100, and 200 from the declared seed. Monitoring
# keeps its default threshold 0.2 and minimum count 128. Canonical input labels are
# uniform among four values, independently missing in 5% of rows.
#
# The four IDs start at 2^60, where adjacent integers collapse if naively cast to float64. Novel IDs add 64. Both identity fields receive the same value in each row. Profiling uses 128 fixed identity buckets and can miss collisions for other identifiers; this is a selected synthetic population, not a collision-free guarantee. No vocabulary is learned by Hash.
#
# These are rank-local marginal diagnostics. They do not estimate a production
# false-positive rate or prove joint-distribution detection. Actual warning
# records and reference/observed histograms are retained for every arm.
#
# ### Complete experiment code

# %%
"""P105: Hash drift across strata with large integer IDs and salt-independent profiling."""

import logging

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from relflow.data.iterables import encode
from reporting import experiment, report

BUDGET = 16
TRAIN_ROWS = 4096
VALIDATE_ROWS = 1024
TEST_ROWS = 2048
SEED_OFFSETS = (0, 100, 200)
STRATA = ("validate", "test", "predict")


class Capture(logging.Handler):
    """Retain real structured warnings without changing profile computation."""

    def __init__(self):
        super().__init__(logging.WARNING)
        self.records = []

    def emit(self, record):
        context = getattr(record, "relflow_context", {})
        if context.get("component") == "drift":
            self.records.append(dict(context))


IDENTITY_BASE = 2**60
CONDITIONS = {"same_population": None, "frequencies": "label", "novel": "label", "nulls": "state"}


def records(*, rows, seed, condition="same_population"):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 4, rows)
    missing = rng.random(rows) < 0.05
    targets = rng.random(rows) < 0.5
    for index, absent, target in zip(labels, missing, targets, strict=True):
        value = IDENTITY_BASE + (0 if condition == "frequencies" else int(index)) + (64 if condition == "novel" else 0)
        value = None if absent or condition == "nulls" else value
        yield {"value": value, "peer": value, "target": bool(target)}


def build():
    return rf.Model(
        drift=True,
        d_model=16,
        n_layers=1,
        n_heads=2,
        attention=None,
        value=rf.Hash(n_hashes=2),
        peer=rf.Hash(n_hashes=2),
        target=rf.Boolean(mask=True),
    )


def trial(seed, steps, accelerator):
    lit.seed_everything(seed, workers=True)
    model = build()

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
    reference = model.distributions.summary()["/value"]["train"]
    capture = Capture()
    core = logging.getLogger("relflow")
    core.addHandler(capture)
    arms, checks = {}, {}
    checks["training reference counts every observed row"] = (
        reference["state"]["count"] == model.trainer.global_step * 128
    )

    source = pa.Table.from_pylist(list(records(rows=128, seed=seed + 6)))
    first = encode(
        source,
        model.schema,
        rf.Strata.train,
        model.interprocess_encoding_context,
        seed=seed,
        ordinal=0,
        drift=model.drift,
    )
    second = encode(
        source,
        model.schema,
        rf.Strata.train,
        model.interprocess_encoding_context,
        seed=seed,
        ordinal=1,
        drift=model.drift,
    )
    valued = first.tensors["/value"].state.eq(rf.Tokens.valued)
    checks["different training salts change encoded IDs"] = not torch.equal(
        first.tensors["/value"].content[valued], second.tensors["/value"].content[valued]
    )
    checks["equal IDs retain equality across fields in each salted batch"] = all(
        torch.equal(batch.tensors["/value"].content, batch.tensors["/peer"].content) for batch in (first, second)
    )
    checks["profile identity frequencies ignore training salts"] = np.array_equal(
        first.profiles["/value"]["label"].counts, second.profiles["/value"]["label"].counts
    )
    checks["worker profiling cannot update model-owned reference"] = (
        model.distributions.summary()["/value"]["train"] == reference
    )

    try:
        for ordinal, split in enumerate(STRATA):
            for condition, expected in CONDITIONS.items():
                model.distributions.clear(strata=split)
                before = len(capture.records)
                source = pa.Table.from_pylist(
                    list(records(rows=TEST_ROWS, seed=seed + 3 + ordinal, condition=condition))
                )
                model.encode(source.drop(["target"]) if split == "predict" else source, strata=split)
                warnings = capture.records[before:]
                observed = model.distributions.summary()["/value"][split]
                detected = expected is not None and any(
                    w["address"] == "/value" and w["feature"] == expected for w in warnings
                )
                arms[f"{split}/{condition}"] = {
                    "split": split,
                    "condition": condition,
                    "expected_feature": expected,
                    "warnings": warnings,
                    "detected": detected,
                    "reference": reference,
                    "observed": observed,
                }
                if expected is None:
                    checks[f"{split}/{condition}: no drift warning"] = not warnings
                else:
                    checks[f"{split}/{condition}: expected feature warns"] = detected
                    checks[f"{split}/{condition}: warning records stratum and sample evidence"] = all(
                        w["strata"] == split
                        and w["reference"] == "train"
                        and w["reference_count"] >= 128
                        and w["count"] >= 128
                        and w["distance"] > w["threshold"] + w["allowance"]
                        for w in warnings
                    )
                checks[f"{split}/{condition}: all held-out rows profiled"] = observed["state"]["count"] == TEST_ROWS

        for ordinal, split in enumerate(STRATA):
            model.distributions.clear(strata=split)
            source = pa.Table.from_pylist(
                list(records(rows=TEST_ROWS, seed=seed + 3 + ordinal, condition="frequencies"))
            )
            before = len(capture.records)
            trace = []
            for batch in range(5):
                start = len(capture.records)
                chunk = source.slice(batch * 64, 64)
                model.encode(chunk.drop(["target"]) if split == "predict" else chunk, strata=split)
                observed = model.distributions.summary()["/value"][split]
                warnings = capture.records[start:]
                incoming = [w for w in warnings if w["address"] == "/value" and w["feature"] == "label"]
                eligible = min(reference["label"]["count"], observed["label"]["count"]) >= 128
                checks[f"{split}/batch={batch}: no warning before sample minimum"] = eligible or not incoming
                checks[f"{split}/batch={batch}: profile counts accumulate every batch"] = (
                    observed["state"]["count"] == (batch + 1) * 64
                )
                trace.append(
                    {
                        "batch": batch,
                        "profiled_rows": observed["state"]["count"],
                        "label_count": observed["label"]["count"],
                        "eligible": eligible,
                        "warnings": warnings,
                    }
                )
            warnings = capture.records[before:]
            detected = sum(w["address"] == "/value" and w["feature"] == "label" for w in warnings) == 1
            checks[f"{split}: incremental shift warns exactly once"] = detected
            arms[f"{split}/batched_frequencies"] = {
                "split": split,
                "condition": "batched_frequencies",
                "expected_feature": "label",
                "warnings": warnings,
                "detected": detected,
                "reference": reference,
                "observed": observed,
                "batch_trace": trace,
            }

        checks["evaluation leaves learned tensor state unchanged"] = all(
            torch.equal(value, model.state_dict()[name]) for name, value in frozen.items()
        )
        checks["evaluation preserves exact training profile"] = (
            model.distributions.summary()["/value"]["train"] == reference
        )
        checks["missing prediction targets are expected"] = "predict" not in model.distributions.summary()["/target"]
    finally:
        core.removeHandler(capture)
    return {"seed": seed, "steps": model.trainer.global_step, "arms": arms, "training_reference": reference}, checks


@experiment("P105", seed=10501)
def run(seed, steps, accelerator):
    trials, checks = [], {}
    for offset in SEED_OFFSETS:
        measured, criteria = trial(seed + offset, steps, accelerator)
        trials.append(measured)
        checks.update({f"seed={seed + offset}/{name}": value for name, value in criteria.items()})
    changed = [arm for trial in trials for arm in trial["arms"].values() if arm["expected_feature"] is not None]
    controls = [
        arm
        for trial in trials
        for arm in trial["arms"].values()
        if arm["expected_feature"] is None and "observed" in arm
    ]
    return {
        "trials": trials,
        "seed_panel": [seed + offset for offset in SEED_OFFSETS],
        "budget": BUDGET,
        "train_rows": TRAIN_ROWS,
        "validate_rows": VALIDATE_ROWS,
        "test_rows": TEST_ROWS,
        "strata": STRATA,
        "detection_rate": sum(arm["detected"] for arm in changed) / len(changed),
        "shift_cases": len(changed),
        "control_cases": len(controls),
        "control_warnings": sum(len(arm["warnings"]) for arm in controls),
    }, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
