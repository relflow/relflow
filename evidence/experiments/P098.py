# %% [markdown]
# ---
# title: Can lookup handle absent keys and several matches?
# categories:
# - Relational
# description: A fresh query key selects zero or more source payloads, with hidden existence and mean-payload outputs.
# evidence-readout:
# - label: One-match error
#   metric:
#   - intact
#   - matches_1
#   - normalized_rmse
#   format: number
# - label: Three-match error
#   metric:
#   - intact
#   - matches_3
#   - normalized_rmse
#   format: number
# evidence-group: Relational
# evidence-diagram:
#   caption: Visible inputs and the hidden prediction target for this controlled task.
#   alt: A fresh query key selects zero or more source payloads, with hidden existence and mean-payload outputs.
# evidence-review:
#   question: Can lookup handle absent keys and several matches?
#   modeling: Create a fresh query ID with zero, one, two, or three matching source records. Hide both whether any
#     match exists and the average of matching payloads; the synthetic no-match payload is zero. Score match detection
#     separately for every match count.
#   comparison:
#     metric: Accuracy
#     kind: Recorded conditions
#     conditions:
#     - label: 0 matching sources · match detection
#       path:
#       - intact
#       - matches_0
#       - found_accuracy
#       format: percent
#     - label: 1 matching sources · match detection
#       path:
#       - intact
#       - matches_1
#       - found_accuracy
#       format: percent
#     - label: 2 matching sources · match detection
#       path:
#       - intact
#       - matches_2
#       - found_accuracy
#       format: percent
#     - label: 3 matching sources · match detection
#       path:
#       - intact
#       - matches_3
#       - found_accuracy
#       format: percent
#     definition: Fraction of predictions matching the expected answer, over the population named in each condition.
#     direction: Higher is better
#     domain:
#     - 0
#     - 1
#   note: The primary chart shows whether any match was found. Numeric retrieval, record reordering, and broken-key
#     controls must pass their own criteria; this is not an exact database join.
# evidence-plot:
#   target: answer
#   focus: Can the model distinguish missing, unique, and duplicated matches?
#   comparison: Separate match probability from numeric answer error at each declared multiplicity.
#   limit: A correct found/not-found flag does not establish the right numeric aggregation for duplicates.
#   renderer: plot
# ---
#
# The query selects every source record with the same identity. In the illustration, owl matches 0.2 and
# 0.8, so found is true and answer is their mean 0.5. With no match, found is false and the numeric answer
# is defined to be zero. Both outputs are hidden prediction targets. A single source position, persistent
# identity meaning, or global payload average is insufficient.
#
# ## Example
#
# ```yaml
# source:
#   - {entity_id: owl, value: 0.2}
#   - {entity_id: fern, value: -0.6}
#   - {entity_id: owl, value: 0.8}
#   - {entity_id: lake, value: 0.5}
# query_key: owl
# found: true
# answer: 0.5
# ```
#
# ```typst
# #tree(node("lookup", kind: "root", children: (node("source", kind: "branch", children: (node("entity_id", type: "Hash"), node("value", type: "Number"))), node("query_key", type: "Hash"), node("found", kind: "target", type: "Boolean"), node("answer", kind: "target", type: "Number"),)))
# ```
#
# ## Comparison
#
# Balance zero, one, two, and three matches while independently varying source length. Evaluate
# whole-record permutations and a query-key corruption that preserves the collection of query names but
# breaks the record-local links. Compare payload error with the mean of every source payload, which ignores
# the requested key. Report each match-count condition separately so the three positive conditions cannot
# hide failure on absence.
#
# ## Result
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# ## Method
#
# ### Process, controls, and held-out evaluation
#
# Each independent row contains 4-8 source records with Uniform(-1,1) payloads. Exactly 0/1/2/3 records
# share its query key; all distractor keys are unique. A separate stream seed is embedded in every name,
# keeping train, validation, and test namespaces disjoint. Input source order is independently permuted.
# Training/validation/test contain 4096/512/1024 balanced rows. One frozen control model has two layers and
# uncompressed source tokens, and receives 600 AdamW updates, batch 64, learning rate 0.002, weight decay
# 0.01, FP32, with no checkpoint selection. Numeric errors normalize against zero; the no-match condition
# uses a separate source-unit error gate because its target variance is zero. Broken queries retain
# original answers only for the dependency diagnostic; they are not presented as new valid target labels.
# Reordered quality is gated, but exact permutation invariance is not assumed of positional attention. This
# learned approximate retrieval task is not an exact database-join contract.
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
# capability = "structure"
# types = ["Hash"]
# conditions = ["unseen-identities", "changed-associations", "transformations", "varying-collections"]
# ///

"""P098: learn explicit no-match and multi-match retrieval semantics for fresh keys."""

import sys
from collections.abc import Iterator
from copy import deepcopy
from pathlib import Path

# Shared reporting and frozen policies travel with the evidence checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
import torch
from policies import CONTROL
from reporting import experiment, report

BUDGET = 600
TRAIN_ROWS = 4096
TEST_ROWS = 1024


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    multiplicities = np.resize(np.arange(4), rows)
    rng.shuffle(multiplicities)
    for row_index, matches in enumerate(multiplicities):
        length = int(rng.integers(4, 9))
        query = f"match-{seed}-{row_index}"
        values = rng.uniform(-1, 1, length)
        source = [
            {"entity_id": query if index < matches else f"other-{seed}-{row_index}-{index}", "value": float(value)}
            for index, value in enumerate(values)
        ]
        order = rng.permutation(length)
        yield {
            "source": [source[index] for index in order],
            "query_key": query,
            "found": bool(matches),
            "answer": float(values[:matches].mean()) if matches else 0.0,
        }


class Source(rf.Branch):
    entity_id = rf.Hash(n_hashes=4, n_bands=8)
    value = rf.Number


class Model(rf.Model):
    source = Source(length=8, overflow="error", reduction=None, n_layers=2)
    query_key = rf.Hash(n_hashes=4, n_bands=8)
    found = rf.Boolean(mask=True)
    answer = rf.Number(mask=True, objective="mse")


def build() -> rf.Model:

    model = Model(preset=CONTROL.name, n_layers=2, n_heads=4, reduction=None)
    return model


def predict(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    visible = [{"source": row["source"], "query_key": row["query_key"]} for row in rows]
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(visible), batch_size=128))["predictions"].to_list()
    return np.asarray([row["found"]["content"]["probability"] for row in output]), np.asarray(
        [row["answer"]["content"] for row in output]
    )


def multiplicity(row: dict) -> int:
    return sum(item["entity_id"] == row["query_key"] for item in row["source"])


def score(probability: np.ndarray, predicted: np.ndarray, rows: list[dict]) -> dict:
    counts = np.asarray([multiplicity(row) for row in rows])
    result = {}
    for count in range(4):
        indices = np.flatnonzero(counts == count)
        target = np.asarray([rows[i]["answer"] for i in indices])
        reference = np.asarray([np.mean([item["value"] for item in rows[i]["source"]]) for i in indices])
        error = float(np.sqrt(np.mean((predicted[indices] - target) ** 2)))
        result[f"matches_{count}"] = {
            "rows": len(indices),
            "found_accuracy": float(np.mean((probability[indices] >= 0.5) == bool(count))),
            "payload_rmse": error,
            "source_mean_rmse": float(np.sqrt(np.mean((reference - target) ** 2))),
            "normalized_rmse": error / float(np.sqrt(np.mean(target**2))) if count else None,
        }
    return result


def fit(model: rf.Model, train: list[dict], validate: list[dict], *, seed: int, budget: int, accelerator: str) -> int:
    """Own the fixed optimizer budget and independent synthetic splits."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validate),
        seed=seed,
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
        max_epochs=-1,
        max_steps=budget,
        deterministic=True,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P098", seed=9801)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1))
    validate = list(records(rows=512, seed=seed * 100 + 2))
    test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3))
    lit.seed_everything(seed, workers=True)
    model = build()
    completed = fit(
        model,
        train,
        validate,
        seed=seed,
        budget=BUDGET if steps is None else min(steps, BUDGET),
        accelerator=accelerator,
    )
    before = deepcopy(model.state_dict())
    probability, predicted = predict(model, test)
    metrics = {"steps": completed, "intact": score(probability, predicted, test)}
    rng = np.random.default_rng(seed + 11)
    shuffled = [{**row, "source": [row["source"][i] for i in rng.permutation(len(row["source"]))]} for row in test]
    shuffled_probability, shuffled_output = predict(model, shuffled)
    metrics["reordered"] = score(shuffled_probability, shuffled_output, test)
    changed = [{**row, "query_key": test[(index + 1) % len(test)]["query_key"]} for index, row in enumerate(test)]
    changed_probability, changed_output = predict(model, changed)
    matched = np.asarray([row["found"] for row in test])
    metrics["corruption"] = {
        "mean_found_probability_drop": float(np.mean(probability[matched] - changed_probability[matched])),
        "payload_rmse_against_original": float(
            np.sqrt(np.mean((changed_output - np.asarray([row["answer"] for row in test])) ** 2))
        ),
    }
    checks = {
        "Full budget completed": completed == BUDGET,
        "Finite public predictions": bool(
            np.isfinite(probability).all() and np.isfinite(predicted).all() and np.isfinite(changed_output).all()
        ),
        "Broken keys lower match confidence by at least 0.20": metrics["corruption"]["mean_found_probability_drop"]
        > 0.20,
    }
    for condition in ("intact", "reordered"):
        for count in range(4):
            measured = metrics[condition][f"matches_{count}"]
            checks[f"{condition}/{count} matches: existence accuracy at least 0.85"] = (
                measured["found_accuracy"] >= 0.85
            )
            if count:
                checks[f"{condition}/{count} matches: payload normalized RMSE below 0.55"] = (
                    measured["normalized_rmse"] < 0.55
                )
                checks[f"{condition}/{count} matches: beats the unfiltered source mean by 20 percent"] = (
                    measured["payload_rmse"] < 0.8 * measured["source_mean_rmse"]
                )
            else:
                checks[f"{condition}: no-match payload RMSE below 0.15"] = measured["payload_rmse"] < 0.15
    checks["Evaluation preserves full model state"] = all(
        (
            torch.equal(value, model.state_dict()[name])
            if isinstance(value, torch.Tensor)
            else value == model.state_dict()[name]
        )
        for name, value in before.items()
    )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from matplotlib.ticker import PercentFormatter

    from evidence.diagnostics.data import human
    from evidence.diagnostics.plots import COLORS, MARKERS, axes_pair

    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.7)
    conditions = [name for name in ["intact", "reordered", "changed_query"] if name in m]
    if not conditions:
        conditions = [name for name, value in m.items() if isinstance(value, dict) and "matches_0" in value]
    for index, name in enumerate(conditions):
        x = list(range(4))
        values = m[name]
        axes[0].plot(
            x,
            [values[f"matches_{count}"]["found_accuracy"] for count in x],
            marker=MARKERS[index],
            color=COLORS[index],
            label=human(name),
        )
        axes[1].plot(
            x,
            [values[f"matches_{count}"]["payload_rmse"] for count in x],
            marker=MARKERS[index],
            color=COLORS[index],
            label=human(name),
        )
    axes[0].set(
        title="Does a matching key exist?",
        xlabel="Number of matches",
        ylabel="Found/not-found accuracy",
        xticks=range(4),
        ylim=(0, 1.03),
    )
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].set(
        title="Is the numeric answer right?",
        xlabel="Number of matches",
        ylabel="Payload RMSE (source units)",
        xticks=range(4),
    )
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.18)
    frame = next((frame for frame in bundle.frames if frame["function"] == "predict"))
    rows = bundle.value(frame["values"]["rows"])
    probability, predicted = bundle.value(frame["result"])
    samples = [
        {
            "coordinate": [i],
            "input": row,
            "actual": row["answer"],
            "predicted": float(predicted[i]),
            "found_probability": float(probability[i]),
            "matches": sum((item["entity_id"] == row["query_key"] for item in row["source"])),
        }
        for i, row in enumerate(rows[:16])
    ]
    return (
        fig,
        "No match has a defined zero payload in this generator. Multiple matches require the mean of their matched values, so existence accuracy and payload error remain separate. Source-unit RMSE keeps the zero-match condition defined.",
        {
            "kind": "numeric",
            "conditions": [{"label": "Intact lookup", "field": "answer", "n": len(rows), "records": samples}],
        },
    )


if __name__ == "__main__":
    report(run)
