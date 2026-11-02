# %% [markdown]
# ---
# title: Is an unfamiliar category the same as a null?
# categories:
# - Vocabulary
# evidence-id: P055
# description: Compare familiar labels, new labels, and missing values in a task where identity and numeric context
#   both matter.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Error on paired new identities
#   metric:
#   - unknown
#   - paired_rmse
#   format: number
# - label: Unavoidable error without identity
#   metric:
#   - unknown
#   - information_limit_rmse
#   format: number
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     Evaluation does not admit labels or change counts: true
#     Null remains distinct from unknown content: true
#     Paired novel identities stay near their information limit: true
#     Unknown codes retain valued state: true
#     Unknown content uses the allocation-independent sentinel: true
#     Unknown fallback nRMSE below 0.25: true
#     Unknown spelling cannot alter predictions: true
#     'known: held-out nRMSE below 0.25': true
#     'null: held-out nRMSE below 0.25': true
#   duration_seconds: 13.502
#   metrics:
#     known:
#       baseline_rmse: 1.9253678498400628
#       nrmse: 0.07660095583111826
#       rmse: 0.14748501762425378
#     'null':
#       baseline_rmse: 1.1205872792139533
#       nrmse: 0.14862581090559648
#       rmse: 0.16654819306366986
#     steps: 600
#     unknown:
#       fallback_nrmse: 0.23963495599963372
#       information_limit_rmse: 1.5
#       null_prediction_gap: 3.382460893936809
#       paired_rmse: 1.525272913261471
#       renaming_drift: 0.0
#     vocabulary:
#     - left
#     - right
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 60e583a1467a9138b29f742d72a34db51fd88f491f5c11c51e2ebcbb946f392d
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 9438c9bd8132
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: 5fb8a21f9bdbf9ebdbccf05a90317302c2922e1f247388430210dbf226b61ffb
#     source_sha256: 9a2c1d9d1968667ef04a7f6be74b0655a27f1413adf00ae5d07056a698bd372f
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 5501
#   started_at: '2026-11-01T01:47:35.555448+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
#
# ## Example
#
# ```yaml
# x: 0.4
# code: right
# target: 2.3
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-unknown-input-fallback
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, x, code, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number"),
#   node("code", type: "Category", body: [30% unavailable content in training]),
#   node("target", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Rename new labels without changing other inputs. Train with some input labels made unavailable so the model can learn a fallback.
#
# ## Result
#
# {{< evidence P055 status >}}
#
# Different new labels share unavailable identity content, while null remains distinct. Other context can support a fallback, but the model cannot infer the new label’s individual effect.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P055 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P055 script >}}
#
# ### Data and model
#
# Independent rows draw `x ~ Uniform(-1, 1)` and a code with probabilities
# 0.4 left, 0.4 right, and 0.2 null. The target is `2*x + effect(code)`.
# Training/validation/test use 4,096/512/2,048 rows. At test time, two never-seen
# codes have opposite effects of -1.5 and +1.5. The model cannot distinguish
# their identities, so `2*x` is the paired unknown-code conditional mean.
#
# ### Training and controls
#
# The model uses the [xs preset](../../concepts/model-tree.qmd#choose-a-size).
#
# Use 600 AdamW updates at learning rate 0.002. Known-code and null predictions
# must each achieve nRMSE below 0.25 against their training-mean baseline.
# Unknown predictions must approximate `2*x` below 0.25 nRMSE and stay within
# 15% of the 1.5 RMSE information limit on paired novel identities. Renaming
# an unknown code cannot change predictions; replacing it with null must.
# All evaluation uses frozen vocabularies and normalization statistics.
#
# ### Complete experiment code
#

# %%
"""P055: known category effects, unknown-content fallback, and distinct null semantics."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P055"
BUDGET = 600
EFFECTS = {"left": -1.5, "right": 1.5, None: -3.0}


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    codes = np.asarray(["left", "right", None], dtype=object)
    for x, index in zip(rng.uniform(-1, 1, rows), rng.choice(3, rows, p=[0.4, 0.4, 0.2]), strict=True):
        code = codes[index]
        yield {"x": float(x), "code": code, "target": float(2 * x + EFFECTS[code])}


def build() -> rf.Model:
    return rf.Model.xs(
        x=rf.Number,
        code=rf.Category(p_unavailable=0.3),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    result = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["target"]["content"] for row in result], dtype=np.float64)


def rmse(actual: np.ndarray, predicted: np.ndarray | float) -> float:
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


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
    vocabulary = rf.Category.vocabulary(model, "/code")
    counts = rf.Category.counts(model, "/code")
    predicted = prediction(model, test)
    metrics = {"steps": trainer.global_step, "vocabulary": vocabulary}
    checks = {}
    for name, is_null in (("known", False), ("null", True)):
        selected = np.asarray([(row["code"] is None) == is_null for row in test])
        actual = np.asarray([row["target"] for row in test])[selected]
        mean = float(np.mean([row["target"] for row in train if (row["code"] is None) == is_null]))
        baseline = rmse(actual, mean)
        metrics[name] = {
            "rmse": rmse(actual, predicted[selected]),
            "baseline_rmse": baseline,
            "nrmse": rmse(actual, predicted[selected]) / baseline,
        }
        checks[f"{name}: held-out nRMSE below 0.25"] = metrics[name]["nrmse"] < 0.25
    novel = [{**row, "code": "novel-left"} for row in test]
    renamed = [{**row, "code": "novel-right"} for row in test]
    nulls = [{**row, "code": None} for row in test]
    first = prediction(model, novel)
    second = prediction(model, renamed)
    expected = 2 * np.asarray([row["x"] for row in test])
    baseline = rmse(expected, float(np.mean([2 * row["x"] for row in train])))
    metrics["unknown"] = {
        "fallback_nrmse": rmse(expected, first) / baseline,
        "paired_rmse": rmse(np.concatenate((expected - 1.5, expected + 1.5)), np.concatenate((first, second))),
        "information_limit_rmse": 1.5,
        "renaming_drift": float(np.max(np.abs(first - second))),
        "null_prediction_gap": float(np.mean(np.abs(first - prediction(model, nulls)))),
    }
    field = model.encode(pa.Table.from_pylist(novel), strata="test")["/code"]
    checks.update(
        {
            "Unknown codes retain valued state": bool(field.state.eq(rf.Tokens.valued).all()),
            "Unknown content uses the allocation-independent sentinel": bool(field.content.eq(-1).all()),
            "Unknown fallback nRMSE below 0.25": metrics["unknown"]["fallback_nrmse"] < 0.25,
            "Paired novel identities stay near their information limit": 1.5 - 1e-6
            <= metrics["unknown"]["paired_rmse"]
            < 1.5 * 1.15,
            "Unknown spelling cannot alter predictions": metrics["unknown"]["renaming_drift"] < 1e-5,
            "Null remains distinct from unknown content": metrics["unknown"]["null_prediction_gap"] > 2.0,
            "Evaluation does not admit labels or change counts": rf.Category.vocabulary(model, "/code") == vocabulary
            and rf.Category.counts(model, "/code") == counts,
        }
    )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5501)

# %% [markdown]
# </details>
