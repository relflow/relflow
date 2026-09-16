# %% [markdown]
# ---
# title: Early rare labels can exhaust vocabulary capacity
# categories: [Vocabulary and OOV]
# proof-id: P057
# description: Compare rare-first and common-first admission with equal observations, then restore headroom as a positive control.
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
# Vocabulary capacity is first-come admission, not a top-frequency filter.
# A label can occur thousands of times during training yet remain unavailable
# because earlier labels filled every slot. Such labels are model-OOV, even
# though they are not data-unseen. Reordering the same observations or reserving
# more capacity provides a matched positive control. This proof intentionally
# exercises one consumer with shuffling disabled; it is not a worker-order test.
#
# ## Setup

# %%
"""P057: first-come capacity saturation removes a learnable category distinction."""

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
# without duplicating, deleting, or otherwise changing any record. A third
# model keeps rare-first order but has four slots instead of two.
# Validation uses 512 independent observations. Test generation uses 2,050
# independent observations and scores only its 2,048 common-label rows.
#
# ```yaml
# code: rare-left
# target: -1.0
# ```
#
# Together with rare-right, an early singleton can consume all two slots.
#
# ```yaml
# code: cold
# target: -1.0
# ```
#
# Cold can then occur repeatedly without obtaining a learnable identity.
#
# ```yaml
# code: hot
# target: 1.0
# ```
#
# If neither common label is admitted, their encoder representations coincide.
# The best prediction over the balanced mixture is approximately zero.
#
# ```{typst}
# //| label: fig-proof-capacity-admission-order
# //| fig-cap: "Only category identity can distinguish the signed target; admission order controls which identities survive."
# //| fig-alt: "Record has Category code and hidden Number target. The arms differ in encounter order or vocabulary headroom, not observations or update budget."
# #tree(node("record", kind: "root", children: (
#   node("code", type: "Category", body: [Bounded first-come vocabulary]),
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


def build(size: int) -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=64,
        code=rf.Category(size=size, p_unavailable=0.0),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> np.ndarray:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return np.asarray([row["record/target"]["content"] for row in output], dtype=np.float64)


# %% [markdown]
# ## Training and controls
#
# Each arm uses the same seed and 300 AdamW updates at learning rate 0.002,
# with `shuffle=False`, `num_workers=0`, and one device. The rare-first two-slot
# model must have zero coverage of common labels and nRMSE near one. The
# common-first and headroom controls must cover both labels and achieve nRMSE
# below 0.15. Renaming one overflow label to the other must have no effect on
# the saturated model. Gates are declared before the first full run.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = [row for row in records(rows=2050, seed=seed + 3) if row["code"] in {"cold", "hot"}]
    train = [row for row in records(rows=4096, seed=seed + 1) if row["code"] in {"cold", "hot"}]
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    metrics = {"baseline_rmse": baseline, "arms": {}}
    checks = {}
    for name, size, common_first in (("rare_first", 2, False), ("common_first", 2, True), ("headroom", 4, False)):
        lit.seed_everything(seed, workers=True)
        model = build(size)
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
        measured = {
            "steps": trainer.global_step,
            "capacity": size,
            "vocabulary": vocabulary,
            "counts": rf.Category.counts(model, "record/code"),
            "common_coverage": float(np.mean([row["code"] in vocabulary for row in test])),
            "nrmse": float(np.sqrt(np.mean((actual - predicted) ** 2))) / baseline,
        }
        if name == "rare_first":
            renamed = [{**row, "code": "hot" if row["code"] == "cold" else "cold"} for row in test]
            measured["overflow_renaming_drift"] = float(np.max(np.abs(predicted - prediction(model, renamed))))
            checks["First rare labels occupy every slot"] = vocabulary == ("rare-left", "rare-right")
            checks["Frequently seen common labels remain OOV"] = measured["common_coverage"] == 0.0
            checks["Overflow identities cannot be distinguished"] = measured["overflow_renaming_drift"] < 1e-5
            checks["Saturated model remains near the constant baseline"] = 0.90 < measured["nrmse"] < 1.10
        else:
            checks[f"{name}: common-label coverage is complete"] = measured["common_coverage"] == 1.0
            checks[f"{name}: common-label nRMSE below 0.15"] = measured["nrmse"] < 0.15
        checks[f"{name}: prediction does not change admission"] = (
            rf.Category.vocabulary(model, "record/code") == vocabulary
        )
        metrics["arms"][name] = measured
    return metrics, checks


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P057 evidence >}}
#
# This is an admission-order limitation, not a guarantee about shuffled or
# distributed loaders. The headroom arm is trained from scratch; changing the
# capacity of a trained model is a different mutation with different state
# retention behavior. Thresholds remain provisional pending calibration.
#
# ## Reproduce
#
# {{< proof P057 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5701)
