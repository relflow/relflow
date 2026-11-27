# %% [markdown]
# ---
# title: Can it follow IDs through larger lists and a second lookup?
# description: Retrieve a source value by ID with increasing numbers of distractors, then add an intermediate ID link
#   for a two-step lookup.
# evidence-readout:
# - label: Direct lookup, four sources
#   metric:
#   - hops_1
#   - ladder
#   - '4'
#   - intact
#   - normalized_rmse
#   format: error
# - label: Two-step lookup, four sources
#   metric:
#   - hops_2
#   - ladder
#   - '4'
#   - intact
#   - normalized_rmse
#   format: error
# evidence-group: Relational
# evidence-review:
#   question: Can relflow follow IDs through larger lists and a second lookup?
#   modeling: Create an ID-keyed source list and ask for a payload through either a direct match or an intermediate
#     link list. IDs and list orders are fresh and independent. In the primary two-hop, four-source test, rotate link
#     IDs while keeping original answers and compare RMSE.
#   comparison:
#     metric: RMSE
#     kind: Controlled comparison
#     conditions:
#     - label: Two-hop lookup · links changed
#       path:
#       - hops_2
#       - ladder
#       - '4'
#       - broken
#       - rmse
#       format: number
#     - label: Two-hop lookup · correct links
#       path:
#       - hops_2
#       - ladder
#       - '4'
#       - intact
#       - rmse
#       format: number
#     definition: Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.
#     direction: Lower error is better
#   note: This is one slice of the lookup-size ladder. Every recorded size, direct lookup, and reordering condition
#     has its own gates; the primary slice cannot establish them all.
# evidence-plot:
#   target: target/value
#   focus: Where do distractor count and an extra relation break retrieval?
#   comparison: Separate one-hop and two-hop conditions, capacities, intact links, broken links, and record permutations.
#   limit: Capacity failures and seed failures remain separate; a successful small condition cannot stand in for the
#     grid.
#   renderer: plot
# evidence-cost:
#   baseline: 168
#   candidate: 169
# ---
#
# Each observation supplies a small lookup table of fresh identities and numeric payloads, then
# requests two payloads by identity. Larger lists add distractors without changing the rule. The
# two-step arm inserts another relation, requiring a query identity to lead to a source identity
# before retrieving its value.
#
# ## Example
#
# ```yaml
# source:
#   - {entity_id: S0, value: 0.7}
#   - {entity_id: S1, value: -0.2}
#   - {entity_id: S2, value: 0.4}
#   - {entity_id: S3, value: -0.9}
# target:
#   - {entity_id: S1, value: -0.2}
#   - {entity_id: S0, value: 0.7}
# ```
#
# Target values are hidden. This example shows a direct lookup; the two-step arm adds a separate links
# collection.
#
#
# The illustrative direct request for `S1` should return −0.2 and the request for `S0` should return
# 0.7. These names stand for fresh keys in the generator. In a two-step illustration, a target key
# `Q1` would first follow a link `Q1 → S1` and then retrieve −0.2. The model sees relations, not a
# generating array index.
#
# ## Comparison
#
# Change lookup keys while keeping answers fixed. Compare with averaging source values, and reorder
# the records without changing matches.
#
# Broken-key tests keep source payloads and hidden target values fixed while changing the relation
# that points to them. A model following that altered relation should now disagree with the original
# answer. Reordering complete source, link, and target records preserves valid matches and tests
# robustness to position. A per-record source-mean baseline also checks whether apparent success is
# merely averaging the available values.
#
# ## Result
#
# The selected run misses the declared intact-retrieval gates for both direct and two-step requests.
# Capability failures remain reported, rather than being redefined as successful evidence of an
# expected limitation.
#
# The selected run does not establish reliable retrieval across the tested source counts. Errors and
# failed conditions remain separate for direct and two-step requests. These lengths already occur
# during fitting, so the ladder tests learned capacity rather than extrapolation to longer lists. Even
# a successful condition would not establish an exact join or behavior for missing and duplicate keys.
#
# ## Method
#
# ### Process and split unit
#
# Each observation samples independent source values from `Uniform(-1, 1)`.
# Two randomly chosen sources supply hidden targets; remaining sources are
# distractors. Every identity is fresh, including across train, validation, and
# test namespaces. Source and target orders are independent. In the two-hop arm,
# a separately shuffled link collection maps query keys to source keys. The
# model must compose those two equality relationships to recover a payload.
# Every requested key has exactly one source and, where applicable, one link.
# Missing and duplicate keys are excluded, so no undocumented tie or null
# convention enters the task. No observation contains a generating index field.
#
# The paired broken-link control retains targets and all key marginals while
# rotating the links' source IDs. Its visible relation points to wrong answers.
#
# ### Model tree and training
#
# Train separate one-hop and two-hop models for 600 AdamW updates each, batch
# size 64, learning rate 0.003. The frozen Evidence control policy uses two attention layers here;
# root and every branch retain tokens. One-hop training balances source counts
# 2/4/8/16, and two-hop training balances 2/4. Thus the ladder tests capacity at
# observed lengths, not length extrapolation. Fixed source capacity 16 and two
# target coordinates prevent the parameter budget changing with test length.
# Each arm has 4,096 train and 512 independent validation rows. Every ladder
# condition has 2,048 fresh test rows; validation never selects a checkpoint.
#
# Gates are declared before training: each condition must achieve normalized
# RMSE at most 0.50, beat its per-record source-mean baseline by at least 20%,
# and have broken-key normalized RMSE at least 0.80 with a gap at least 0.30.
# Complete-record permutation must preserve the learning gate and change
# normalized RMSE by at most 0.10. This is a performance robustness gate, not
# exact permutation invariance; branch attention retains positional capacity.
# Each failed condition is reported independently, including two-hop failures.
#
# ### Remaining work
#
# The three-seed panel retained the original gates and 600 updates per arm.
# Diagnose the one-hop optimization instability and two-hop information route
# using training and validation data before designing a follow-up experiment.
# A single successful one-hop seed does not meet the suite's stability contract.
# Even stable success would not establish exact joins, unobserved collection
# lengths, missing-key behavior, duplicate aggregation, or collision immunity.
# Source-unit errors and every condition failure remain in the evidence.
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

"""P075: test fresh-key retrieval with distractors and an intermediate relation."""

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
from visibility import TARGET

BUDGET = 600
BATCH_SIZE = 64
TRAIN_ROWS = 4096
TEST_ROWS = 2048
CAPACITIES = (2, 4, 8, 16)


def records(
    *, rows: int, seed: int, namespace: str, hops: int, capacity: int | None = None, broken: bool = False
) -> Iterator[dict]:
    """Fresh unique keys with a matched key-rotation control and fixed targets."""
    rng = np.random.default_rng(seed)
    choices = CAPACITIES if hops == 1 else CAPACITIES[:2]
    sizes = np.resize(choices, rows) if capacity is None else np.full(rows, capacity)
    rng.shuffle(sizes)
    for row, size in enumerate(sizes):
        keys = [f"{namespace}-{row:06d}-s{index:02d}" for index in range(size)]
        values = rng.uniform(-1.0, 1.0, size)
        selected = rng.choice(size, size=2, replace=False)
        source = [{"entity_id": keys[index], "value": float(values[index])} for index in rng.permutation(size)]
        if hops == 1:
            target = [
                {"entity_id": keys[(index + int(broken)) % size], "value": float(values[index])} for index in selected
            ]
            yield {"source": source, "target": target}
        else:
            query_keys = [f"{namespace}-{row:06d}-q{index:02d}" for index in range(size)]
            links = [
                {"entity_id": query_keys[index], "source_id": keys[(index + int(broken)) % size]}
                for index in rng.permutation(size)
            ]
            target = [{"entity_id": query_keys[index], "value": float(values[index])} for index in selected]
            yield {"source": source, "links": links, "target": target}


def permute(rows: list[dict], seed: int) -> list[dict]:
    """Reorder complete records independently, preserving each visible relation."""
    rng = np.random.default_rng(seed)
    return [
        {name: [items[index] for index in rng.permutation(len(items))] for name, items in row.items()} for row in rows
    ]


class Source(rf.Branch):
    entity_id = rf.Hash(n_hashes=4, n_bands=8)
    value = rf.Number


class Target(rf.Branch):
    entity_id = rf.Hash(n_hashes=4, n_bands=8)
    value = rf.Number(**TARGET, objective="mse")


class Link(rf.Branch):
    entity_id = rf.Hash(n_hashes=4, n_bands=8)
    source_id = rf.Hash(n_hashes=4, n_bands=8)


class Model(rf.Model):
    source = Source(length=16, reduction=None, n_layers=2)
    target = Target(length=2, reduction=None, n_layers=2)


class LinkedModel(Model):
    links = Link(length=16, reduction=None, n_layers=2)


def build(hops: int) -> rf.Model:
    model_type = LinkedModel if hops == 2 else Model
    return model_type(preset=CONTROL.name, n_layers=2, n_heads=4, reduction=None)


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{**row, "target": [{"entity_id": item["entity_id"]} for item in row["target"]]} for row in rows]
    output = pl.DataFrame(model.predict(pa.Table.from_pylist(visible)))["predictions"].to_list()
    return np.asarray([[item["value"]["content"] for item in row["target"]] for row in output])


def score(predicted: np.ndarray, rows: list[dict], training_mean: float) -> dict:
    actual = np.asarray([[item["value"] for item in row["target"]] for row in rows])
    source_mean = np.asarray([np.mean([item["value"] for item in row["source"]]) for row in rows])[:, None]
    baseline = float(np.sqrt(np.mean((actual - training_mean) ** 2)))
    rmse = float(np.sqrt(np.mean((predicted - actual) ** 2)))
    return {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "source_mean_rmse": float(np.sqrt(np.mean((actual - source_mean) ** 2))),
        "normalized_rmse": rmse / baseline,
    }


@experiment("P075", seed=7501)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for hops in (1, 2):
        lit.seed_everything(seed, workers=True)
        model = build(hops)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, namespace="train", hops=hops),
            validate=partial(records, rows=512, seed=seed * 100 + 2, namespace="validate", hops=hops),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=BATCH_SIZE,
            learning_rate=0.003,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            max_epochs=-1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        trainer = model.trainer
        model.eval()
        train = records(rows=TRAIN_ROWS, seed=seed * 100 + 1, namespace="train", hops=hops)
        training_mean = float(np.mean([item["value"] for row in train for item in row["target"]]))
        metrics[f"hops_{hops}"] = {"steps": trainer.global_step, "training_mean": training_mean, "ladder": {}}
        for capacity in CAPACITIES if hops == 1 else CAPACITIES[:2]:
            options = {
                "rows": TEST_ROWS,
                "seed": seed * 100 + 10 + capacity,
                "namespace": f"test-{hops}-{capacity}",
                "hops": hops,
                "capacity": capacity,
            }
            test = list(records(**options))
            broken = list(records(**options, broken=True))
            permuted = permute(test, seed * 100 + 40 + capacity)
            intact = score(predict(model, test), test, training_mean)
            corrupted = score(predict(model, broken), broken, training_mean)
            reordered = score(predict(model, permuted), permuted, training_mean)
            gap = corrupted["normalized_rmse"] - intact["normalized_rmse"]
            delta = abs(reordered["normalized_rmse"] - intact["normalized_rmse"])
            metrics[f"hops_{hops}"]["ladder"][capacity] = {
                "intact": intact,
                "broken": corrupted,
                "permuted": reordered,
                "corruption_gap": gap,
                "permutation_nrmse_delta": delta,
            }
            label = f"{hops} hop, {capacity} sources"
            checks.update(
                {
                    f"{label}: finite source-unit errors": bool(
                        np.isfinite([*intact.values(), *corrupted.values(), *reordered.values()]).all()
                    ),
                    f"{label}: intact nRMSE at most 0.50": intact["normalized_rmse"] <= 0.50,
                    f"{label}: beats source-mean RMSE by 20%": intact["rmse"] <= 0.8 * intact["source_mean_rmse"],
                    f"{label}: broken-key nRMSE at least 0.80": corrupted["normalized_rmse"] >= 0.80,
                    f"{label}: corruption increases nRMSE by at least 0.30": gap >= 0.30,
                    f"{label}: reordered nRMSE at most 0.50": reordered["normalized_rmse"] <= 0.50,
                    f"{label}: permutation nRMSE delta at most 0.10": delta <= 0.10,
                }
            )
    return metrics, checks


def plot(bundle, spec):
    """Render the captured observations for this experiment."""
    from evidence.diagnostics.plots import COLORS, MARKERS, axes_pair

    m = bundle.record["metrics"]
    fig, axes = axes_pair(width=10, height=4.6)
    for ax, hops in zip(axes, [1, 2], strict=True):
        values = m[f"hops_{hops}"]["ladder"]
        capacities = sorted((int(key) for key in values))
        for index, (condition, label) in enumerate(
            [("intact", "Intact keys"), ("broken", "Broken keys"), ("permuted", "Complete records reordered")]
        ):
            y = []
            for capacity in capacities:
                item = values[str(capacity)]
                measured = item.get(condition, item.get("reordered") if condition == "permuted" else None)
                y.append(None if measured is None else measured["normalized_rmse"])
            present = [i for i, value in enumerate(y) if value is not None]
            if present:
                ax.plot(
                    [capacities[i] for i in present],
                    [y[i] for i in present],
                    marker=MARKERS[index],
                    color=COLORS[index],
                    lw=1.8,
                    label=label,
                )
        source = [
            values[str(c)]["intact"]["source_mean_rmse"] / values[str(c)]["intact"]["baseline_rmse"] for c in capacities
        ]
        ax.plot(capacities, source, ls=":", color="gray", label="Source-mean shortcut")
        ax.axhline(0.5, ls="--", color=COLORS[3], label="Intact learning gate")
        ax.set(
            xlabel="Source collection capacity",
            ylabel="Test RMSE / baseline RMSE",
            title=f"{hops}-hop retrieval",
            xscale="log",
        )
        ax.set_xticks(capacities, labels=capacities)
        ax.minorticks_off()
        ax.grid(alpha=0.18)
        ax.legend(fontsize=8)
    return (
        fig,
        "Every capacity is an evaluated condition. One-hop and two-hop models have separate panels. A broken-key gap and the learning gate are both required; the source-mean shortcut is an explicit baseline, not the correct retrieval answer.",
        {},
    )


if __name__ == "__main__":
    report(run)
