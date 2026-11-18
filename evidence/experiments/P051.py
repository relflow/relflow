# %% [markdown]
# ---
# title: Can hiding values teach a relationship in both directions?
# categories:
# - Masking
# description: Sometimes hide x or y during training, then ask the model to recover the missing value from the other.
# evidence-readout:
# - label: Recover y from x
#   metric:
#   - directions
#   - y
#   - nrmse
#   format: error
# - label: Recover x from y
#   metric:
#   - directions
#   - x
#   - nrmse
#   format: error
# evidence-group: Masking
# evidence-diagram:
#   caption: Selector notes identify hidden values. Each field is sometimes hidden during training.
#   alt: Model tree with record, x, y. Selector notes identify hidden values. Each field is sometimes hidden during
#     training.
# evidence-review:
#   question: Can hiding values teach a relationship in both directions?
#   modeling: Generate independent x values and set y = 2x + 0.3. Randomly hide either field during training so the
#     model learns to reconstruct it from the other. At test time, request each direction and compare intact context
#     with shuffled context.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Recover x · shuffled y
#       path:
#       - directions
#       - x
#       - shuffled_nrmse
#       format: number
#     - label: Recover x · correct y
#       path:
#       - directions
#       - x
#       - nrmse
#       format: number
#     - label: Recover y · shuffled x
#       path:
#       - directions
#       - y
#       - shuffled_nrmse
#       format: number
#     - label: Recover y · correct x
#       path:
#       - directions
#       - y
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: The four bars use normalized error so both directions can be shown on comparable reference scales. Hidden-value
#     poisoning is checked separately.
# evidence-plot:
#   mode: regression
#   target: y
#   focus: Can either field be reconstructed from its visible partner?
#   comparison: Inspect both directions, hidden-value poisoning, and shuffled visible context.
#   limit: Only selected hidden targets contribute to each scored reconstruction direction.
#   additional_targets:
#   - x
# evidence-cost:
#   baseline: 19
#   candidate: 19
# ---
#
# ## Example
#
# The two numbers obey `y = 2x + 0.3`. In the illustration, visible `x: 0.4` identifies hidden `y` as `1.1`;
# reversing the selection would ask for `x = (1.1 - 0.3) / 2 = 0.4`. The `hide_x` and `hide_y` flags choose
# which stored values are unavailable at prediction time. During training, each field is also randomly
# selected for reconstruction at a 35% rate, exposing the model to different information patterns.
#
# ```yaml
# x: 0.4
# y: 1.1
# hide_x: false
# hide_y: true
# ```
#
# Values marked for reconstruction are hidden. During training, the selection is sampled separately for each field.
#
# ```typst
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", body: [35% sampled; query `hide_x`]),
#   node("y", type: "Number", body: [35% sampled; query `hide_y`]),
# )))
# ```
#
# ## Comparison
#
# Evaluation requests each direction separately on fresh records. Shuffling the other, visible field breaks
# the relationship while keeping the original target fixed. Hiding both fields removes the evidence
# entirely. Replacing a hidden target with a large, unrelated number checks that its stored value cannot
# leak into its prediction. Separate mask audits verify partial sampling, repeatability within an epoch, and
# changes across epochs without changing the fitted model with held-out data.
#
# Score both directions on new records. Shuffle the visible value, or hide both, to remove the needed information.
#
# ## Result
#
# Each normalized error compares one direction with a constant guess from that target's training values. Low
# intact error and high shuffled or both-hidden error support reconstruction from the other field, rather
# than successful guessing without context. The two directions share this simple invertible relation and
# bounded synthetic inputs. They do not demonstrate recovery from arbitrary missingness patterns, causal
# inference, or reliable imputation when real variables contain noise or do not determine one another.
#
# The model learns both directions. When both values are hidden, the original answer cannot be identified; this is a small relationship test, not general imputation evidence.
#
# ## Method
#
# ### Data and model
#
# Independent records draw `x ~ Uniform(-1, 1)` and set `y = 2*x + 0.3`.
# Train, validation, and test contain 4,096, 512, and 2,048 independent rows.
# Each field has a 35% sampled reconstruction policy and a Boolean query policy.
# Query flags are false during fitting; inference turns on exactly one flag.
# Neither flag is an embedded field. No separate supervised target is added.
#
# ### Training and controls
#
# The model uses the frozen Evidence control architecture; its settings are shown in the code.
#
# Fit for 400 AdamW updates at learning rate 0.002. Evaluate the final model;
# no checkpoint or gate is selected using the test set. Each direction must
# achieve nRMSE below 0.25 relative to its training-mean baseline. Shuffling the
# remaining input, or hiding both fields, must give nRMSE above 0.85. Poisoning
# only a hidden source value must leave predictions unchanged within 1e-5.
# Mask diagnostics also check reproducibility, epoch variation, and the absence
# of sampled reconstruction requests during ordinary prediction.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "lightning",
#     "numpy",
#     "polars",
#     "pyarrow",
#     "pyyaml>=6",
#     "relflow>=0.5.0",
#     "torch",
# ]
#
# [tool.relflow.evidence]
# capability = "training"
# types = []
# conditions = ["missing-inputs", "changed-associations", "hidden-values", "objectives"]
# ///

"""P051: sampled cross-field reconstruction with deterministic inference controls."""

import sys
from collections.abc import Iterator
from functools import partial
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from policies import CONTROL
from reporting import experiment, report

BUDGET = 400


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for value in rng.uniform(-1.0, 1.0, rows):
        yield {"x": float(value), "y": float(2 * value + 0.3), "hide_x": False, "hide_y": False}


class Model(rf.Model):
    x = rf.Number(
        objective="mse",
        mask=(rf.Mask(rate=0.35, reconstruct=True), rf.Mask(query="hide_x", reconstruct=True)),
    )
    y = rf.Number(
        objective="mse",
        mask=(rf.Mask(rate=0.35, reconstruct=True), rf.Mask(query="hide_y", reconstruct=True)),
    )


def build() -> rf.Model:
    return Model(preset=CONTROL.name)


def prediction(model: rf.Model, rows: list[dict], target: str) -> np.ndarray:
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(rows)))["predictions"].to_list()
    return np.asarray([row[target]["content"] for row in output], dtype=np.float64)


def error(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


@experiment("P051", seed=5101)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
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
    trainer = model.trainer
    model.eval()
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    source = pa.Table.from_pylist(test)
    # Training-mode audits must not update the fitted model with held-out moments.
    probe = build()
    encoded = [probe.encode(source, strata="train", seed=seed, epoch=epoch) for epoch in (0, 0, 1)]
    ordinary = probe.encode(source, strata="predict")
    metrics = {"steps": trainer.global_step, "directions": {}}
    checks = {}
    order = np.random.default_rng(seed + 4).permutation(len(test))
    for target, context in (("y", "x"), ("x", "y")):
        selected = [{**row, f"hide_{target}": True} for row in test]
        actual = np.asarray([row[target] for row in test])
        baseline = error(actual, float(np.mean([row[target] for row in train])))
        predicted = prediction(model, selected, target)
        shuffled = [{**row, context: test[index][context]} for row, index in zip(selected, order, strict=True)]
        hidden = [{**row, f"hide_{context}": True} for row in selected]
        poisoned = [{**row, target: 1000.0 + index} for index, row in enumerate(selected)]
        measured = {
            "rmse": error(actual, predicted),
            "baseline_rmse": baseline,
            "nrmse": error(actual, predicted) / baseline,
            "shuffled_nrmse": error(actual, prediction(model, shuffled, target)) / baseline,
            "both_hidden_nrmse": error(actual, prediction(model, hidden, target)) / baseline,
            "hidden_value_drift": float(np.max(np.abs(predicted - prediction(model, poisoned, target)))),
        }
        masks = [batch[f"/{target}"].trainable for batch in encoded]
        measured["sampled_fraction"] = float(masks[0].float().mean())
        metrics["directions"][target] = measured
        checks[f"{target}: held-out nRMSE below 0.25"] = measured["nrmse"] < 0.25
        checks[f"{target}: shuffled context nRMSE above 0.85"] = measured["shuffled_nrmse"] > 0.85
        checks[f"{target}: no context nRMSE above 0.85"] = measured["both_hidden_nrmse"] > 0.85
        checks[f"{target}: hidden values cannot affect predictions"] = measured["hidden_value_drift"] < 1e-5
        checks[f"{target}: sampling is partial"] = 0.25 < measured["sampled_fraction"] < 0.45
        checks[f"{target}: sampling repeats within an epoch"] = bool((masks[0] == masks[1]).all())
        checks[f"{target}: sampling changes across epochs"] = bool((masks[0] != masks[2]).any())
        checks[f"{target}: ordinary prediction requests no reconstruction"] = not bool(
            ordinary[f"/{target}"].inferred.any()
        )
    return metrics, checks


if __name__ == "__main__":
    report(run)
