# %% [markdown]
# ---
# title: Vocabulary growth preserves labels across admission orders
# categories: [Vocabulary and OOV]
# proof-id: P057
# description: Compare rare-first and common-first admission with automatic storage growth; both arms must admit all labels and learn their effects.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P057 status >}}
#
# ## Insights
#
# Vocabulary storage is managed internally. Training admits every observed
# label and grows storage when needed, preserving earlier IDs. Encounter order
# changes which labels receive the first IDs; it does not exclude later labels.
# This proof requires equal coverage and learned common-label distinctions from
# both encounter orders, without specifying a vocabulary size.
# It exercises one consumer with shuffling disabled, not distributed ordering.
#
# ## Setup

# %%
"""P057: vocabulary growth retains admission order and learnable distinctions."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P057"
BUDGET = 300
EFFECTS = {"rare-left": -1.0, "rare-right": 1.0, "cold": -1.0, "hot": 1.0}

# %% [markdown]
# ## Data and model
#
# A training pass has 4,096 rows: one of each rare label and 4,094 balanced
# common labels. The target is the label's effect plus Gaussian noise with SD
# 0.02. For common-first, move the first cold and hot observations to the front
# without duplicating, deleting, or otherwise changing any record.
# Validation uses 512 independent observations. Test generation uses 2,050
# independent observations and scores only its 2,048 common-label rows.
#
# ```yaml
# code: rare-left
# target: -1.0
# ```
#
# Together with rare-right, this singleton takes the first two IDs in the
# rare-first arms. Those IDs must survive admission of the common labels.
#
# ```yaml
# code: cold
# target: -1.0
# ```
#
# Cold must obtain its own identity regardless of which labels arrive first.
#
# ```yaml
# code: hot
# target: 1.0
# ```
#
# Every arm must distinguish cold from hot. A constant prediction over their
# balanced mixture is approximately zero and supplies the error baseline.
#
# ```{typst}
# //| label: fig-proof-capacity-admission-order
# //| fig-cap: "Category identity distinguishes the signed target across encounter orders with automatic growth."
# //| fig-alt: "Record has Category code and hidden Number target. Each arm admits all four labels; the arms differ in encounter order, with the same observations and update budget."
# #tree(node("record", kind: "root", children: (
#   node("code", type: "Category", body: [Growing vocabulary with stable IDs]),
#   node("target", kind: "target", type: "Number"),
# )))
# ```


# %%
def records(*, rows: int, seed: int, common_first: bool = False) -> Iterator[dict]:
    if rows < 4:
        raise ValueError("capacity proof requires at least four observations")
    rng = np.random.default_rng(seed)
    common = np.resize(np.asarray(["cold", "hot"]), rows - 2)
    rng.shuffle(common)
    labels = ["rare-left", "rare-right", *common.tolist()]
    noise = rng.normal(0.0, 0.02, rows)
    observations = [
        {"code": label, "target": float(EFFECTS[label] + deviation)}
        for label, deviation in zip(labels, noise, strict=True)
    ]
    order = list(range(rows))
    if common_first:
        front = [labels.index("cold"), labels.index("hot")]
        order = front + [index for index in order if index not in front]
    for index in order:
        yield observations[index]


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=64,
        code=rf.Category(p_unavailable=0.0),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["record/target"]["content"] for row in output], dtype=np.float64)


# %% [markdown]
# ## Training and controls
#
# Each arm uses the same seed and 300 AdamW updates at learning rate 0.002,
# with `shuffle=False`, `num_workers=0`, and one device. Every arm must admit
# all four labels, retain the first two IDs, have common-label coverage of one,
# and achieve nRMSE below 0.15. Swapping cold and hot must change predictions
# by more than 1.5 on average, demonstrating use of their distinct identities.
# Both arms must grow to hold all four labels.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = [row for row in records(rows=2050, seed=seed + 3) if row["code"] in {"cold", "hot"}]
    train = [row for row in records(rows=4096, seed=seed + 1) if row["code"] in {"cold", "hot"}]
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    metrics = {"baseline_rmse": baseline, "arms": {}}
    checks = {}
    for name, common_first in (("rare_first", False), ("common_first", True)):
        lit.seed_everything(seed, workers=True)
        model = build()
        initial_size = model.nodes["record/code"].embedder.size
        model.optimizer = rf.adamw(learning_rate=0.002)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=4096, seed=seed + 1, common_first=common_first),
            validate=partial(records, rows=512, seed=seed + 2),
            seed=seed,
            shuffle=False,
            num_workers=0,
        )
        trainer = lit.Trainer(
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
        trainer.fit(model, datamodule=data)
        model.eval()
        vocabulary = rf.Category.vocabulary(model, "record/code")
        predicted = prediction(model, test)
        renamed = [{**row, "code": "hot" if row["code"] == "cold" else "cold"} for row in test]
        measured = {
            "steps": trainer.global_step,
            "initial_size": initial_size,
            "allocated_size": model.nodes["record/code"].embedder.size,
            "vocabulary": vocabulary,
            "counts": rf.Category.counts(model, "record/code"),
            "common_coverage": float(np.mean([row["code"] in vocabulary for row in test])),
            "nrmse": float(np.sqrt(np.mean((actual - predicted) ** 2))) / baseline,
            "identity_swap_drift": float(np.mean(np.abs(predicted - prediction(model, renamed)))),
        }
        first = ("cold", "hot") if common_first else ("rare-left", "rare-right")
        checks[f"{name}: all four labels are admitted"] = len(vocabulary) == 4 and set(vocabulary) == set(EFFECTS)
        checks[f"{name}: first label IDs are preserved"] = vocabulary[:2] == first
        checks[f"{name}: allocation holds all four labels"] = measured["allocated_size"] == 4
        checks[f"{name}: storage grows automatically"] = measured["allocated_size"] > initial_size
        checks[f"{name}: all labels have training exposures"] = all(
            measured["counts"].get(label, 0) > 0 for label in EFFECTS
        )
        checks[f"{name}: common-label coverage is complete"] = measured["common_coverage"] == 1.0
        checks[f"{name}: common-label nRMSE below 0.15"] = measured["nrmse"] < 0.15
        checks[f"{name}: common identities have distinct learned effects"] = measured["identity_swap_drift"] > 1.5
        checks[f"{name}: prediction does not change admission"] = (
            rf.Category.vocabulary(model, "record/code") == vocabulary
        )
        metrics["arms"][name] = measured
    return metrics, checks


# %% [markdown]
# ## Scope of evidence
#
# Earlier fixed-capacity runs used different gates. Assess this revision using
# a full run whose source fingerprint matches the current script; earlier
# scores do not establish its autoscaling behavior.
#
# {{< proof P057 evidence >}}
#
# This checks one-device admission and common-label learning. The singleton
# rare labels establish admission and ID order, not reliable rare-label
# predictions. It does not establish worker/DDP ordering or optimizer-state
# preservation after growth later in training. Dedicated runtime tests cover
# those invariants separately, including two-rank DDP, persistent workers,
# gradient accumulation, and checkpoint resume.
#
# ## Reproduce
#
# {{< proof P057 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5701)
