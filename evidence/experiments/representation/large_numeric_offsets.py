# %% [markdown]
# ---
# title: "Can learning preserve small differences beside enormous offsets?"
# categories: ["Representation"]
# description: "Compare the same smooth regression around zero and around one trillion, keeping source values and residual scoring in float64."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: "Wide-offset error"
#   metric: ["arms", "wide", "nrmse"]
#   format: number
# - label: "Centered error"
#   metric: ["arms", "centered", "nrmse"]
#   format: number
# - label: "Float32 input error"
#   metric: ["arms", "float32", "nrmse"]
#   format: number
# evidence-group: Representation
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
# ```{typst}
# //| label: fig-evidence-p087
# //| fig-cap: "The target is hidden. The diagram shows the informative representation; controls change only the declared condition."
# //| fig-alt: "Can learning preserve small differences beside enormous offsets? Visible inputs lead to one hidden prediction target."
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
# {{< evidence P087 status >}}
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P087 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P087 script >}}
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
# ### Complete experiment code

# %%
"""P087: test learned residual quality through large source offsets."""

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
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
    output = pa.table(model.predict(table(rows, targets=False), batch_size=256))["predictions"].to_pylist()
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


@experiment("P087", seed=8701)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics = {"budget": BUDGET, "arms": {}}
    checks = {}
    for arm in ARMS:
        lit.seed_everything(seed, workers=True)
        model = rf.Model.xs(x=rf.Number, target=rf.Number(mask=True, objective="mse"))
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


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
