# %% [markdown]
# ---
# title: Can one number representation serve additive and multiplicative structure?
# categories:
# - Representation
# evidence-id: P079
# description: Compare Number, which embeds raw and log lanes and reads out standardized log-scale deviations, with
#   Quantile ranks on additive, moderate-tail, heavy-tail, and far-from-zero targets.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Heavy tails, Number
#   metric:
#   - heavy
#   - number_number
#   - normalized_mae
#   format: error
# - label: Far from zero, Number
#   metric:
#   - narrow
#   - number_number
#   - normalized_mae
#   format: error
# - label: Additive, Number
#   metric:
#   - additive
#   - number_number
#   - normalized_mae
#   format: error
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     'Additive: Number reaches 1.5 times the oracle MAE': true
#     'Additive: Number stays within 10 percent of Quantile inputs': true
#     Every arm produces finite predictions: true
#     'Far from zero: Number reaches 1.5 times the oracle MAE': true
#     'Far from zero: Number stays within 10 percent of Quantile inputs': true
#     'Heavy tails: Number reaches 1.5 times the oracle MAE': true
#     'Heavy tails: Number stays within 10 percent of Quantile inputs': true
#     'Moderate tails: Number reaches 1.5 times the oracle MAE': true
#     'Moderate tails: Number stays within 10 percent of Quantile inputs': true
#     Shuffled inputs lose the relationship in every arm: true
#   duration_seconds: 160.589
#   metrics:
#     additive:
#       baseline_mae: 33.230921049070666
#       bulk_rows: null
#       number_number:
#         beyond_support_error_share: 0.0
#         beyond_support_rows: 0
#         finite: true
#         mae: 1.9333745825465352
#         median_distance: 1.062471290618725
#         normalized_mae: 0.058179987839988075
#         oracle_ratio: 1.215323254145294
#         shuffled_normalized_mae: 1.3902584594649083
#         steps: 500
#       oracle_mae: 1.5908315552689958
#       rank_number:
#         beyond_support_error_share: 0.008213336775824953
#         beyond_support_rows: 22
#         finite: true
#         mae: 3.1947011494463045
#         median_distance: 2.7116295147463774
#         normalized_mae: 0.09613640093600859
#         oracle_ratio: 2.008195738175503
#         shuffled_normalized_mae: 1.4445251881458265
#         steps: 500
#       tail_rows: null
#       training_median: 1.0262815748840113
#       training_support:
#       - -98.10397112660722
#       - 100.62500716859951
#     budget: 500
#     heavy:
#       baseline_mae: 79.14908516991858
#       bulk_rows: 2572
#       number_number:
#         beyond_support_error_share: 0.0
#         beyond_support_rows: 0
#         bulk_oracle_ratio: 1.090619567686523
#         finite: true
#         mae: 26.604375658198798
#         median_distance: 16.319786048308693
#         normalized_mae: 0.33612991989843066
#         oracle_ratio: 1.3269451516457105
#         shuffled_normalized_mae: 1.4811931649591692
#         steps: 500
#         tail_oracle_ratio: 1.5129977685524794
#       number_number_long:
#         beyond_support_error_share: 0.0
#         beyond_support_rows: 0
#         bulk_oracle_ratio: 1.0546667108898338
#         finite: true
#         mae: 21.82464357450091
#         median_distance: 9.709787359168917
#         normalized_mae: 0.2757409454278265
#         oracle_ratio: 1.0885466868174716
#         shuffled_normalized_mae: 1.6204925535642425
#         steps: 1500
#         tail_oracle_ratio: 1.1227589709073844
#       number_rank:
#         beyond_support_error_share: 0.0
#         beyond_support_rows: 0
#         bulk_oracle_ratio: 1.048921137407804
#         finite: true
#         mae: 33.16910108701492
#         median_distance: 24.802940097495316
#         normalized_mae: 0.4190711871881645
#         oracle_ratio: 1.6543736427919908
#         shuffled_normalized_mae: 1.3660961698741176
#         steps: 500
#         tail_oracle_ratio: 2.1258748617225507
#       oracle_mae: 20.049340867784466
#       rank_number:
#         beyond_support_error_share: 1.9967565551829775e-06
#         beyond_support_rows: 2
#         bulk_oracle_ratio: 1.1027928730074237
#         finite: true
#         mae: 28.683663958809632
#         median_distance: 19.3010391809064
#         normalized_mae: 0.36240044843514063
#         oracle_ratio: 1.4306537131551744
#         shuffled_normalized_mae: 1.4472828752800304
#         steps: 500
#         tail_oracle_ratio: 1.6808455255526193
#       tail_rows: 422
#       training_median: 19.49389424374008
#       training_support:
#       - 0.021296251049474066
#       - 12924.484930568127
#     long_budget: 1500
#     moderate:
#       baseline_mae: 49.16591498411684
#       bulk_rows: 2618
#       number_number:
#         beyond_support_error_share: 5.157968796357616e-06
#         beyond_support_rows: 4
#         bulk_oracle_ratio: 1.1150487634930548
#         finite: true
#         mae: 17.078212355323778
#         median_distance: 10.111674053394967
#         normalized_mae: 0.3473587822140793
#         oracle_ratio: 1.2488256958563757
#         shuffled_normalized_mae: 1.43025965682008
#         steps: 500
#         tail_oracle_ratio: 1.4075413327151216
#       oracle_mae: 13.675417163491726
#       rank_number:
#         beyond_support_error_share: 7.507774586979434e-06
#         beyond_support_rows: 4
#         bulk_oracle_ratio: 1.006205224305884
#         finite: true
#         mae: 15.793319921014554
#         median_distance: 5.751310910060084
#         normalized_mae: 0.3212249772249051
#         oracle_ratio: 1.1548693346757157
#         shuffled_normalized_mae: 1.530177625261195
#         steps: 500
#         tail_oracle_ratio: 1.3677211518986592
#       tail_rows: 353
#       training_median: 19.96810061554254
#       training_support:
#       - 0.11390242454780923
#       - 6132.076160092331
#     narrow:
#       baseline_mae: 33.851426493424
#       bulk_rows: null
#       number_number:
#         beyond_support_error_share: 0.0
#         beyond_support_rows: 0
#         finite: true
#         mae: 1.8261058263092174
#         median_distance: 0.9377949986346958
#         normalized_mae: 0.05394472302855414
#         oracle_ratio: 1.1703188154905055
#         shuffled_normalized_mae: 1.4313203891816895
#         steps: 500
#       oracle_mae: 1.5603490280927061
#       rank_number:
#         beyond_support_error_share: 0.0
#         beyond_support_rows: 0
#         finite: true
#         mae: 2.27338800689268
#         median_distance: 1.6922348753519398
#         normalized_mae: 0.06715781999125826
#         oracle_ratio: 1.4569740269403426
#         shuffled_normalized_mae: 1.4028447790702516
#         steps: 500
#       tail_rows: null
#       training_median: 1000.0894108196668
#       training_support:
#       - 900.7738772582007
#       - 1101.7515061983338
#     test_rows: 4096
#     train_rows: 8192
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: ac220ea14f44d43e81d06d7812db6c73cb2aebe9601efa3a45ac4dc4b5ea0dd2
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 32798cb89735
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
#     script_sha256: 839e2f849b36d4e2df86fb27ac9889d88e81442d5480b1e4c1a7bce6ca8709cd
#     source_sha256: ceb2434e537f95bed5bd15d2b7456eb2b5ed5fe31bf1a0ca8f5f98a2df2964a9
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7901
#   started_at: '2026-11-01T01:51:06.435467+00:00'
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
# price: 18.4
# rate: 0.62
# ltv: 15.1
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-number-lanes
# //| fig-cap: "Amber cards are hidden prediction targets. Each arm changes only how numbers are represented."
# //| fig-alt: "Model tree with customer, price, rate, ltv. Amber cards are hidden prediction targets. Each arm changes only how numbers are represented."
# #tree(node("customer", kind: "root", children: (
#   node("price", type: "Number / Quantile", detail: "raw and log lanes, or ranks"),
#   node("rate", type: "Number / Quantile", detail: "raw and log lanes, or ranks"),
#   node("ltv", kind: "target", type: "Number / Quantile", detail: "log-scale readout, or ranks"),
# )))
# ```
#
# ## Comparison
#
# Hold the data, model width, optimizer, and update budget fixed. Change only the
# numeric datatype: Number, which embeds a raw lane and a signed-log lane and
# reads targets out as standardized deviations on the signed-log scale, or
# Quantile, which represents values by their training percentile. Score every
# arm in source units against the known conditional median, on the bulk of the
# distribution and on its tails, for an additive task, for multiplicative tasks
# with moderate and very heavy tails, and for a target that sits near 1,000.
#
# ## Result
#
# {{< evidence P079 status >}}
#
# Number has no scale option. Its two lanes make additive and multiplicative
# structure both readable: with moderate tails they are interchangeable, and
# only when the largest input is ten thousand times the median does a raw lane
# alone degrade, because standardization is then set by a handful of extreme
# rows. Its readout starts at the training mean on the signed-log scale and
# predicts standardized deviations, which is what lets a target near 1,000 or
# a heavy-tailed target train at the same budget as a centered one; a plain
# raw-unit head cannot grow a readout that spans either. A monotone target
# transform leaves the median-optimal MAE answer unchanged, so these are
# claims about optimization at a budget, not about the estimand. Quantile
# ranks are a fair input but a poor heavy-tailed target, because the inverse
# continues linearly past the learned support and a slightly high rank
# becomes an enormous source value.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P079 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P079 script >}}
#
# ### Synthetic processes and oracles
#
# The additive task draws `left` and `right` uniformly on [-50, 50] and sets
# `total = left + right + 2 z`; the far-from-zero task adds 1,000 to that
# total, so its target spans roughly 900 to 1,100. The multiplicative tasks
# draw `price = exp(3 + 1.2 z1)` and
# `rate = exp(s z2)` from independent standard normals, then
# `ltv = price * sqrt(rate) * exp(0.3 z3)`. Prices have a median near 20 and a
# long tail; rates have a median of 1. The heavy task uses `s = 2.5`, where the
# largest of 8,192 training rates is about ten thousand times the median, the
# regime of RelBench's study-adverse count. The moderate task uses `s = 1.5`,
# where that ratio is a few hundred. The noise is symmetric in log units, so
# the conditional median is exactly `price * sqrt(rate)`, and the evaluator
# computes it from the visible inputs without any latent variable.
#
# Oracle MAE is the test-set error of the conditional median. Baseline MAE is
# the error of the training-median constant. The multiplicative bulk is the set
# of test rows whose price and rate both fall between the 10th and 90th training
# percentiles; the tail is any row above the 95th percentile of either input.
# Every arm also records how much of its absolute error comes from predictions
# outside the training target range.
#
# ### Arms and budget
#
# Inputs embed as `Number` or as `Quantile` ranks; targets train as a `Number`
# or as a `Quantile` with the MAE objective. Every task compares the two input
# choices with a Number target; the heavy task also compares the two target
# choices with Number inputs, and gives the Number arm three times the budget
# once.
#
# Every arm uses the `xs` preset, 8,192 training rows, 1,024 validation rows,
# batch size 128, learning rate 0.002, and 500 AdamW updates (1,500 for the
# long arms), scored once on 4,096 fresh rows. A shuffled-input control pairs
# each test target with another row's inputs. Lane weight norms of the first
# projection are recorded for the combined arms as a diagnostic.
#
# ### Gates
#
# Two earlier full runs ablated the lanes and the target scale through options
# that no longer exist. The first, on moderate tails only, refuted a bulk-blur
# gate for the raw lane; a sweep located the raw lane's degradation at
# `s >= 2.5`; the second met every gate with both regimes and showed a raw-unit
# target lagging a log-unit target even at three times the budget. Those runs
# remain in the evidence under their own code fingerprints, and their findings
# became the datatype's defaults. This version gates the contract of the
# resulting Number against Quantile.
#
# Gates remain provisional until ten-seed calibration. Number must reach at most
# 1.5 times the oracle MAE on every task and stay within 10 percent of Quantile
# inputs on every task. Every arm must produce finite predictions and lose the
# relationship under shuffled inputs. The Quantile target, the long-budget arm,
# and the beyond-support error shares are diagnostics without gates.
#
# ### Remaining work
#
# The tasks are two-input and noise-free apart from symmetric log or additive
# noise. They do not cover bounded or bimodal bulks, where a rank lane may add
# information beyond the log lane, or mixed-sign heavy tails. Mixed precision
# is covered only by the recorded side experiments: an fp32 embedder and
# readout under bf16 autocast matched fp32 scores, which is now the default.
#
# ### Complete experiment code
#

# %%
"""P079: compare Number's lanes and log-scale readout with Quantile ranks across numeric regimes."""

from collections.abc import Callable, Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P079"
BUDGET = 500
LONG_BUDGET = 1500
BATCH_SIZE = 128
TRAIN_ROWS = 8192
TEST_ROWS = 4096
PRICE_SCALE = 1.2
HEAVY_SCALE = 2.5
MODERATE_SCALE = 1.5
NOISE = 0.3
SPAN = 50.0
ADDITIVE_NOISE = 2.0
OFFSET = 1000.0
INPUTS = {
    "heavy": ("price", "rate"),
    "moderate": ("price", "rate"),
    "additive": ("left", "right"),
    "narrow": ("left", "right"),
}
TARGETS = {"heavy": "ltv", "moderate": "ltv", "additive": "total", "narrow": "total"}
ARMS = (
    ("heavy", "number", "number", BUDGET),
    ("heavy", "rank", "number", BUDGET),
    ("heavy", "number", "rank", BUDGET),
    ("heavy", "number", "number", LONG_BUDGET),
    ("moderate", "number", "number", BUDGET),
    ("moderate", "rank", "number", BUDGET),
    ("additive", "number", "number", BUDGET),
    ("additive", "rank", "number", BUDGET),
    ("narrow", "number", "number", BUDGET),
    ("narrow", "rank", "number", BUDGET),
)


def multiplicative(*, rows: int, seed: int, scale: float = HEAVY_SCALE) -> Iterator[dict]:
    """Lognormal prices and rates multiply under symmetric log noise; ``scale`` sets the rate tail."""
    rng = np.random.default_rng(seed)
    price = np.exp(3.0 + PRICE_SCALE * rng.standard_normal(rows))
    rate = np.exp(scale * rng.standard_normal(rows))
    ltv = price * np.sqrt(rate) * np.exp(NOISE * rng.standard_normal(rows))
    for index in range(rows):
        yield {"price": float(price[index]), "rate": float(rate[index]), "ltv": float(ltv[index])}


def additive(*, rows: int, seed: int, offset: float = 0.0) -> Iterator[dict]:
    """Two uniform terms add under symmetric noise; ``offset`` moves the total away from zero."""
    rng = np.random.default_rng(seed)
    left = rng.uniform(-SPAN, SPAN, rows)
    right = rng.uniform(-SPAN, SPAN, rows)
    total = offset + left + right + ADDITIVE_NOISE * rng.standard_normal(rows)
    for index in range(rows):
        yield {"left": float(left[index]), "right": float(right[index]), "total": float(total[index])}


RECORDS: dict[str, Callable[..., Iterator[dict]]] = {
    "heavy": multiplicative,
    "moderate": partial(multiplicative, scale=MODERATE_SCALE),
    "additive": additive,
    "narrow": partial(additive, offset=OFFSET),
}


def median(task: str, rows: list[dict]) -> np.ndarray:
    """Conditional median of the target given the visible inputs."""
    if task == "additive":
        return np.asarray([row["left"] + row["right"] for row in rows])
    if task == "narrow":
        return np.asarray([OFFSET + row["left"] + row["right"] for row in rows])
    return np.asarray([row["price"] * np.sqrt(row["rate"]) for row in rows])


def build(task: str, inputs: str, target: str) -> rf.Model:
    scalar = rf.Quantile if inputs == "rank" else rf.Number
    fields: dict[str, object] = {name: scalar for name in INPUTS[task]}
    if target == "rank":
        fields[TARGETS[task]] = rf.Quantile(objective="mae", mask=True)
    else:
        fields[TARGETS[task]] = rf.Number(mask=True)
    return rf.Model.xs(fields=fields)


def fit(model: rf.Model, train, validate, seed: int, steps: int | None, accelerator: str, budget: int) -> int:
    data = rf.SyntheticDataModule(train=train, validate=validate, seed=seed)
    model.fit(
        data,
        batch_size=BATCH_SIZE,
        learning_rate=0.002,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget if steps is None else min(steps, budget),
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return model.trainer.global_step


def predict(model: rf.Model, task: str, rows: list[dict]) -> np.ndarray:
    visible = [{name: row[name] for name in INPUTS[task]} for row in rows]
    predictions = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row[TARGETS[task]]["content"] for row in predictions], dtype=np.float64)


def evaluate(output: np.ndarray, control: np.ndarray, reference: dict, bulk: np.ndarray | None) -> dict:
    target, oracle = reference["target"], reference["oracle"]
    error = np.abs(output - target)
    mae = float(error.mean())
    beyond = (output < reference["support"][0]) | (output > reference["support"][1])
    result = {
        "mae": mae,
        "oracle_ratio": mae / reference["oracle_mae"],
        "normalized_mae": mae / reference["baseline_mae"],
        "median_distance": float(np.mean(np.abs(output - oracle))),
        "shuffled_normalized_mae": float(np.mean(np.abs(control - target))) / reference["baseline_mae"],
        "beyond_support_rows": int(beyond.sum()),
        "beyond_support_error_share": float(error[beyond].sum() / error.sum()) if error.sum() else 0.0,
        "finite": bool(np.isfinite(output).all() and np.isfinite(control).all()),
    }
    if bulk is not None:
        tail = reference["tail"]
        irreducible = np.abs(target - oracle)
        result["bulk_oracle_ratio"] = float(error[bulk].mean() / irreducible[bulk].mean())
        result["tail_oracle_ratio"] = float(error[tail].mean() / irreducible[tail].mean())
    return result


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    stream = seed * 100
    metrics: dict = {"train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "budget": BUDGET, "long_budget": LONG_BUDGET}
    for offset, task in ((0, "heavy"), (20, "moderate"), (10, "additive"), (30, "narrow")):
        records, name = RECORDS[task], TARGETS[task]
        training = list(records(rows=TRAIN_ROWS, seed=stream + offset + 1))
        test = list(records(rows=TEST_ROWS, seed=stream + offset + 3))
        target = np.asarray([row[name] for row in test])
        oracle = median(task, test)
        observed = np.asarray([row[name] for row in training])
        constant = float(np.median(observed))
        reference = {
            "target": target,
            "oracle": oracle,
            "oracle_mae": float(np.mean(np.abs(target - oracle))),
            "baseline_mae": float(np.mean(np.abs(target - constant))),
            "support": (float(observed.min()), float(observed.max())),
        }
        bulk = None
        if task in ("heavy", "moderate"):
            columns = {key: np.asarray([row[key] for row in training]) for key in INPUTS[task]}
            low = {key: np.percentile(values, 10) for key, values in columns.items()}
            high = {key: np.percentile(values, 90) for key, values in columns.items()}
            extreme = {key: np.percentile(values, 95) for key, values in columns.items()}
            bulk = np.all([[low[key] <= row[key] <= high[key] for key in INPUTS[task]] for row in test], axis=1)
            reference["tail"] = np.any([[row[key] > extreme[key] for key in INPUTS[task]] for row in test], axis=1)
        permutation = np.random.default_rng(stream + offset + 4).permutation(TEST_ROWS)
        shuffled = [{**test[index], name: row[name]} for row, index in zip(test, permutation, strict=True)]
        metrics[task] = {
            "oracle_mae": reference["oracle_mae"],
            "baseline_mae": reference["baseline_mae"],
            "training_median": constant,
            "training_support": list(reference["support"]),
            "bulk_rows": None if bulk is None else int(bulk.sum()),
            "tail_rows": None if bulk is None else int(reference["tail"].sum()),
        }
        for arm_task, inputs, space, budget in ARMS:
            if arm_task != task:
                continue
            lit.seed_everything(seed, workers=True)
            model = build(task, inputs, space)
            updates = fit(
                model,
                partial(records, rows=TRAIN_ROWS, seed=stream + offset + 1),
                partial(records, rows=1024, seed=stream + offset + 2),
                seed,
                steps,
                accelerator,
                budget,
            )
            result = evaluate(predict(model, task, test), predict(model, task, shuffled), reference, bulk)
            result["steps"] = updates
            metrics[task][f"{inputs}_{space}" + ("_long" if budget != BUDGET else "")] = result
    arms = [value for task in TARGETS for value in metrics[task].values() if isinstance(value, dict)]
    checks = {}
    for task in TARGETS:
        label = {
            "heavy": "Heavy tails",
            "moderate": "Moderate tails",
            "additive": "Additive",
            "narrow": "Far from zero",
        }[task]
        checks[f"{label}: Number reaches 1.5 times the oracle MAE"] = (
            metrics[task]["number_number"]["oracle_ratio"] <= 1.5
        )
        checks[f"{label}: Number stays within 10 percent of Quantile inputs"] = (
            metrics[task]["number_number"]["mae"] <= 1.10 * metrics[task]["rank_number"]["mae"]
        )
    checks["Shuffled inputs lose the relationship in every arm"] = all(
        arm["shuffled_normalized_mae"] >= 0.90 for arm in arms
    )
    checks["Every arm produces finite predictions"] = all(arm["finite"] for arm in arms)
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7901)

# %% [markdown]
# </details>
