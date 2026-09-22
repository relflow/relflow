# %% [markdown]
# ---
# title: Identity Retrieval with Distractors and Two Hops
# categories: [Sibling entity transfer]
# proof-id: P075
# description: Measure fresh-key retrieval across collection sizes and a separately trained two-hop relation.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P075 status >}}
#
# ## Insights
#
# **One-hop retrieval succeeded in one of three CPU seeds; two-hop retrieval
# failed in all three.** Seed 7503 met every one-hop gate, with source-unit RMSE
# 0.106–0.152 and large error increases when keys were corrupted. The other two
# seeds stayed near the per-record source-average baseline and barely reacted
# to key changes. An average across seeds would hide this difference.
#
# | Source count | One-hop nRMSE, seeds 7501–7502 | One-hop nRMSE, seed 7503 |
# | --- | --- | --- |
# | 2 | 0.728–0.756 | 0.185 |
# | 4 | 0.869–0.876 | 0.204 |
# | 8 | 0.937–0.953 | 0.238 |
# | 16 | 0.971–0.974 | 0.267 |
#
# For seed 7503, broken-key nRMSE was 1.373–1.415, and record permutation kept
# every retrieval gate passing. In the unsuccessful one-hop runs, key
# corruption changed nRMSE by less than 0.001. Better-than-constant prediction
# alone therefore does not demonstrate identity use.
#
# Two-hop nRMSE was 0.704–0.710 for two sources and 0.864–0.882 for four sources.
# These errors were close to source averaging, and rotating the intermediate
# links changed nRMSE by less than 0.000001. Every full run failed the original
# capability gates. The result identifies seed-sensitive one-hop learning and
# an unestablished two-hop capability at this fixed budget; it does not turn
# the failed gates into a passing limitation demonstration.

# %%
"""P075: test fresh-key retrieval with distractors and an intermediate relation."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P075"
BUDGET = 600
BATCH_SIZE = 64
TRAIN_ROWS = 4096
TEST_ROWS = 2048
CAPACITIES = (2, 4, 8, 16)

# %% [markdown]
# ## Process and split unit
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
# The one-hop targets select two values among four sources. Target values are
# always hidden, and the other two source rows are distractors.
#
# ```yaml
# source:
#   - {entity_id: S1, value: -0.2}
#   - {entity_id: S0, value: 0.7}
# links:
#   - {entity_id: Q0, source_id: S1}
#   - {entity_id: Q1, source_id: S0}
# target:
#   - {entity_id: Q1, value: 0.7}
#   - {entity_id: Q0, value: -0.2}
# ```
#
# Two-hop queries use a different key namespace. Link values bridge the keys.
#
# ```yaml
# source:
#   - {entity_id: S1, value: -0.2}
#   - {entity_id: S0, value: 0.7}
# links:
#   - {entity_id: Q0, source_id: S0}
#   - {entity_id: Q1, source_id: S1}
# target:
#   - {entity_id: Q1, value: 0.7}
#   - {entity_id: Q0, value: -0.2}
# ```
#
# The paired broken-link control retains targets and all key marginals while
# rotating the links' source IDs. Its visible relation points to wrong answers.


# %%
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


def build(hops: int) -> rf.Model:
    fields = {
        "source": rf.Branch(
            length=16,
            reduction=None,
            n_layers=2,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            value=rf.Number,
        ),
        "target": rf.Branch(
            length=2,
            reduction=None,
            n_layers=2,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            value=rf.Number(mask=True, objective="mse"),
        ),
    }
    if hops == 2:
        fields["links"] = rf.Branch(
            length=16,
            reduction=None,
            n_layers=2,
            entity_id=rf.Hash(n_hashes=4, n_bands=8),
            source_id=rf.Hash(n_hashes=4, n_bands=8),
        )
    model = rf.Model.xs(n_layers=2, n_heads=4, reduction=None, batch_size=BATCH_SIZE, fields=fields)
    model.optimizer = rf.adamw(learning_rate=0.003)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{**row, "target": [{"entity_id": item["entity_id"]} for item in row["target"]]} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([[item["content"] for item in row["/target/value"]] for row in output])


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


# %% [markdown]
# ## Model tree and training
#
# ```{typst}
# //| label: fig-proof-retrieval-capacity
# //| fig-cap: "All routes retain tokens. The two-hop arm adds a visible link collection."
# //| fig-alt: "A lookup has source Hash IDs and Number values, optional links between two Hash IDs, and target Hash IDs with hidden Number values."
# #tree(node("lookup", kind: "root", children: (
#   node("source", kind: "branch", repeated: true, children: (
#     node("entity_id", type: "Hash"),
#     node("value", type: "Number"),
#   )),
#   node("links", kind: "branch", repeated: true, detail: "Two-hop arm only", children: (
#     node("entity_id", type: "Hash"),
#     node("source_id", type: "Hash"),
#   )),
#   node("target", kind: "branch", repeated: true, children: (
#     node("entity_id", type: "Hash"),
#     node("value", kind: "target", type: "Number"),
#   )),
# )))
# ```
#
# Train separate one-hop and two-hop models for 600 AdamW updates each, batch
# size 64, learning rate 0.003. The `xs` preset uses two attention layers here;
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


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for hops in (1, 2):
        lit.seed_everything(seed, workers=True)
        model = build(hops)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, namespace="train", hops=hops),
            validate=partial(records, rows=512, seed=seed * 100 + 2, namespace="validate", hops=hops),
            seed=seed,
        )
        trainer = lit.Trainer(
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
        trainer.fit(model, datamodule=data)
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


# %% [markdown]
# ## Evidence
#
# {{< proof P075 evidence >}}
#
# ## Remaining work
#
# The three-seed panel retained the original gates and 600 updates per arm.
# Diagnose the one-hop optimization instability and two-hop information route
# using training and validation data before designing a follow-up experiment.
# A single successful one-hop seed does not meet the suite's stability contract.
# Even stable success would not establish exact joins, unobserved collection
# lengths, missing-key behavior, duplicate aggregation, or collision immunity.
# Source-unit errors and every condition failure remain in the evidence.
#
# ## Reproduce
#
# {{< proof P075 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7501)
