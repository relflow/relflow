# %% [markdown]
# ---
# title: Can lookup handle absent keys and several matches?
# categories:
# - Relational
# description: A fresh query key selects zero or more source payloads, with hidden existence and mean-payload
#   outputs.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
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
# ```{typst}
# //| label: fig-evidence-p098
# //| fig-cap: "Visible inputs and the hidden prediction target for this controlled task."
# //| fig-alt: "A fresh query key selects zero or more source payloads, with hidden existence and mean-payload outputs."
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
# {{< evidence P098 status >}}
#
# These checks define a new experiment. A full recorded run is required before claiming that the model
# meets them. Lower normalized error is better; a value of one matches the stated constant reference. The
# numerical gates are provisional and must be calibrated on separate validation seeds rather than adjusted
# after seeing acceptance-test outcomes.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P098 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P098 script >}}
#
# ### Process, controls, and held-out evaluation
#
# Each independent row contains 4-8 source records with Uniform(-1,1) payloads. Exactly 0/1/2/3 records
# share its query key; all distractor keys are unique. A separate stream seed is embedded in every name,
# keeping train, validation, and test namespaces disjoint. Input source order is independently permuted.
# Training/validation/test contain 4096/512/1024 balanced rows. One xs model has two layers and
# uncompressed source tokens, and receives 600 AdamW updates, batch 64, learning rate 0.002, weight decay
# 0.01, FP32, with no checkpoint selection. Numeric errors normalize against zero; the no-match condition
# uses a separate source-unit error gate because its target variance is zero. Broken queries retain
# original answers only for the dependency diagnostic; they are not presented as new valid target labels.
# Reordered quality is gated, but exact permutation invariance is not assumed of positional attention. This
# learned approximate retrieval task is not an exact database-join contract.
#
# ### Complete experiment code

# %%
"""P098: learn explicit no-match and multi-match retrieval semantics for fresh keys."""

from collections.abc import Iterator
from copy import deepcopy

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from relflow.presets.base import resolve
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


def build() -> rf.Model:
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(
        schema=resolve(
            control_preset,
            n_layers=2,
            n_heads=4,
            reduction=None,
            source=rf.Branch(
                length=8,
                overflow="error",
                reduction=None,
                n_layers=2,
                entity_id=rf.Hash(n_hashes=4, n_bands=8),
                value=rf.Number,
            ),
            query_key=rf.Hash(n_hashes=4, n_bands=8),
            found=rf.Boolean(mask=True),
            answer=rf.Number(mask=True, objective="mse"),
        )
    )
    model.preset = control_preset
    return model


def predict(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    visible = [{"source": row["source"], "query_key": row["query_key"]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible), batch_size=128)["predictions"].to_pylist()
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


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
