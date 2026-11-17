# %% [markdown]
# ---
# title: Can learning preserve small differences beside enormous offsets?
# categories:
# - Representation
# description: Compare the same smooth regression around zero and around one trillion, keeping source values and residual
#   scoring in float64.
# evidence-readout:
# - label: Wide-offset error
#   metric:
#   - arms
#   - wide
#   - nrmse
#   format: number
# - label: Centered error
#   metric:
#   - arms
#   - centered
#   - nrmse
#   format: number
# - label: Float32 input error
#   metric:
#   - arms
#   - float32
#   - nrmse
#   format: number
# evidence-group: Representation
# evidence-diagram:
#   caption: The target is hidden. The diagram shows the informative representation; controls change only the declared
#     condition.
#   alt: Can learning preserve small differences beside enormous offsets? Visible inputs lead to one hidden prediction
#     target.
# evidence-review:
#   question: Can learning preserve small differences beside enormous offsets?
#   modeling: Generate the same small regression differences near zero and near a trillion. Fit matched models using
#     float64 source values, explicit centering, or deliberate float32 input conversion. Compare residual error after
#     removing the target offset.
#   comparison:
#     metric: Normalized RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Input rounded to float32
#       path:
#       - arms
#       - float32
#       - nrmse
#       format: number
#     - label: Large offset retained in float64
#       path:
#       - arms
#       - wide
#       - nrmse
#       format: number
#     - label: Explicitly centered input
#       path:
#       - arms
#       - centered
#       - nrmse
#       format: number
#     definition: Root mean squared error divided by the reference scale defined in this experiment. Zero is exact;
#       one equals that reference scale.
#     direction: Lower error is better
#   note: The float32 control loses the source differences before the model receives them. This tests preservation
#     and learning of numerical detail, not recovery of information already rounded away.
# evidence-plot:
#   target: target
#   focus: Can small numeric differences survive a very large offset?
#   comparison: Compare numeric arms on matched source values, including the shuffled-input control.
#   limit: Plots center large-offset values explicitly so the offset cannot visually erase their spread.
#   renderer: plot
# ---
#
# Compare the same smooth regression around zero and around one trillion, keeping source values and
# residual scoring in float64.
#
# ## Example
#
# ```yaml
# x: 1000000000000.25
# target: 1000000000000.75
# ```
#
# The source differs from one trillion by only 0.25. The hidden target triples that residual, so its
# residual is 0.75. Both quantities remain ordinary finite numbers.
#
# The comparison asks whether that small spread survives encoding, learning, and numeric readout. It
# does not ask one model trained at zero to understand a new, unseen offset.
#
# ```typst
# #tree(node("record", kind: "root", children: (node("x", type: "Number"), node("target", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Fit separate, identically initialized models at zero, at large offsets, and after explicit centering.
# A deliberate float32-ingress arm destroys the input spread before relflow sees it. Shuffling held-out
# inputs checks that low error requires their relationship to the answer.
#
# ## Result
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# ## Method
#
# ### Population, controls, and scoring
#
# Each arm uses 4,096 training, 512 validation, and 2,048 test rows, 400 updates, batch 64, learning
# rate 0.003, and weight decay 0.01. There is no validation-based model selection. The wide and centered
# arms share quantized float64 draws. Residual RMSE is computed after subtracting the target offset in
# float64 and normalized by a training-mean forecast. All checks are provisional pending independent
# calibration. The float32 arm is expected to lose information; low error there would demand a leakage
# investigation.
#

# %%
# /// script
# requires-python = ">=3.12"
# dependencies = ["matplotlib>=3.10",
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
# capability = "representation"
# types = ["Number"]
# conditions = ["changed-associations", "representations", "numeric-extremes"]
# ///

"""P087: test learned residual quality through large source offsets."""

import sys
from collections.abc import Iterator
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
TRAIN_ROWS = 4096
TEST_ROWS = 2048
OFFSET = 1e12
ARMS = ("zero", "wide", "centered", "float32")


def records(*, rows: int, seed: int, arm: str) -> Iterator[dict]:
    """Keep latent draws matched while changing numeric ingress/readout offsets."""
    rng = np.random.default_rng(seed)
    residual = rng.uniform(-1.0, 1.0, rows)
    for value in residual:
        source = float(OFFSET + value)
        target = float(OFFSET + 3.0 * (source - OFFSET))
        if arm == "zero":
            source, target = float(value), float(3.0 * value)
        elif arm == "centered":
            source, target = source - OFFSET, target - OFFSET
        elif arm == "float32":
            source = float(np.float32(source))
        yield {"x": source, "target": target}


def table(rows: list[dict], *, targets: bool = True) -> pa.Table:
    columns = {"x": pa.array([row["x"] for row in rows], type=pa.float64())}
    if targets:
        columns["target"] = pa.array([row["target"] for row in rows], type=pa.float64())
    return pa.table(columns)


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = pl.DataFrame(model.predict(table(rows, targets=False), batch_size=256))["predictions"].to_list()
    return np.asarray([row["target"]["content"] for row in output], dtype=np.float64)


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


class Model(rf.Model):
    x = rf.Number
    target = rf.Number(mask=True, objective="mse")


@experiment("P087", seed=8701)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics = {"budget": BUDGET, "arms": {}}
    checks = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)

        model = Model(preset=CONTROL.name)
        train = list(records(rows=TRAIN_ROWS, seed=seed + 1, arm=arm))
        test = list(records(rows=TEST_ROWS, seed=seed + 3, arm=arm))
        updates = fit(
            model, table(train), table(list(records(rows=512, seed=seed + 2, arm=arm))), seed, steps, accelerator
        )
        center = OFFSET if arm in ("wide", "float32") else 0.0
        actual = np.asarray([row["target"] - center for row in test], dtype=np.float64)
        mean = float(np.mean([row["target"] - center for row in train]))
        baseline = float(np.sqrt(np.mean((actual - mean) ** 2)))
        predicted = prediction(model, test) - center
        shuffled = [dict(row) for row in test]
        for index, other in enumerate(np.random.default_rng(seed + 4).permutation(len(test))):
            shuffled[index]["x"] = test[int(other)]["x"]
        corrupted = prediction(model, shuffled) - center
        error = float(np.sqrt(np.mean((predicted - actual) ** 2)))
        metrics["arms"][arm] = {
            "steps": updates,
            "residual_rmse": error,
            "nrmse": error / baseline,
            "shuffled_nrmse": float(np.sqrt(np.mean((corrupted - actual) ** 2))) / baseline,
            "input_unique_values": len(set(row["x"] for row in train)),
            "finite": bool(np.isfinite(predicted).all() and np.isfinite(corrupted).all()),
        }
        checks[f"{arm}: finite held-out predictions"] = metrics["arms"][arm]["finite"]
        checks[f"{arm}: full budget retained"] = updates == BUDGET
        if arm != "float32":
            checks[f"{arm}: residual nRMSE below 0.25"] = error / baseline < 0.25
            checks[f"{arm}: shuffled inputs remove most signal"] = metrics["arms"][arm]["shuffled_nrmse"] > 0.8
    arms = metrics["arms"]
    checks["Wide offsets stay within 0.15 nRMSE of precentered inputs"] = (
        arms["wide"]["nrmse"] <= arms["centered"]["nrmse"] + 0.15
    )
    checks["Float32 ingress collapses the source to one value"] = arms["float32"]["input_unique_values"] == 1
    checks["Collapsed ingress retains at least 0.80 nRMSE"] = arms["float32"]["nrmse"] >= 0.80
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    import numpy as np

    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import COLORS, axes_pair, dots

    m = bundle.record["metrics"]["arms"]
    arms = ["zero", "wide", "centered", "float32"]
    fig, axes = axes_pair(width=10, height=4.9)
    dots(
        axes[0],
        ["Zero-centered source", "Float64 source near 10¹²", "Explicitly precentered", "Float32 ingress near 10¹²"],
        {"Intact inputs": [m[a]["nrmse"] for a in arms], "Shuffled inputs": [m[a]["shuffled_nrmse"] for a in arms]},
        xlabel="Residual RMSE / constant-guess RMSE",
        reference=(1, "Constant-guess error"),
    )
    pairs = bundle.pairs("target")
    for index, arm in enumerate(["wide", "float32"]):
        pair = next((p for p in pairs if p["context"].get("arm") == arm and p["call"] == "prediction(model, test)"))
        actual, predicted = (pair["actual"] - 1000000000000.0, pair["predicted"] - 1000000000000.0)
        if not np.isclose(np.sqrt(np.mean((actual - predicted) ** 2)), m[arm]["residual_rmse"]):
            raise ValueError("Centered precision scatter disagrees with the proof's residual score")
        axes[1].scatter(actual, predicted, s=7, alpha=0.23, color=COLORS[index], label=human(arm), rasterized=True)
    axes[1].plot([-3.2, 3.2], [-3.2, 3.2], ls="--", color="gray")
    axes[1].set(
        xlabel="Actual target minus 10¹²",
        ylabel="Model prediction minus 10¹²",
        title="Predictions after subtracting 10¹²",
        xlim=(-3.3, 3.3),
        ylim=(-3.3, 3.3),
    )
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.18)
    return (
        fig,
        f"All four arms use matched latent draws and the same fixed budget. The wide scatter subtracts 10¹² in float64 on both axes. Float32 ingress leaves {m['float32']['input_unique_values']:,} distinct training input, versus {m['wide']['input_unique_values']:,} in the wide float64 arm; later normalization cannot recover discarded differences.",
        {},
    )


if __name__ == "__main__":
    report(run)
