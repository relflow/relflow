# %% [markdown]
# ---
# title: "Can ordered outputs fill ranks never shown during training?"
# categories: ["Representation"]
# description: "Compare Ordinal, rounded Number, and Category targets when integer ranks or declared intermediate labels are withheld from training."
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: "Unseen integer accuracy"
#   metric: ["domains", "integer", "ordinal", "unseen", "exact_accuracy"]
#   format: percent
# - label: "Unseen ordered-label accuracy"
#   metric: ["domains", "labels", "ordinal", "unseen", "exact_accuracy"]
#   format: percent
# - label: "Number integer accuracy"
#   metric: ["domains", "integer", "number", "unseen", "exact_accuracy"]
#   format: percent
# evidence-group: Representation
# evidence-requires:
# - "relflow stage/0.5.0 wheel exporting Ordinal"
# ---
#
# Compare Ordinal, rounded Number, and Category targets when integer ranks or declared intermediate
# labels are withheld from training.
#
# ## Example
#
# ```yaml
# signal: 2.02
# rank: 2
# ```
#
# In the integer task, training answers are odd ranks from -9 to 9. A new signal near 2 should still
# permit the even rank 2. It is not an unknown vocabulary label for an implicit Ordinal.
#
# The second task declares five ordered labels but trains only on positions zero, two, and four.
# Positions one and three are held out. This research experiment needs the stage/0.5.0 wheel; merely
# declaring an order does not guarantee the model will learn to interpolate it.
#
# ```{typst}
# //| label: fig-evidence-p100
# //| fig-cap: "The target is hidden. The diagram shows the informative representation; controls change only the declared condition."
# //| fig-alt: "Can ordered outputs fill ranks never shown during training? Visible inputs lead to one hidden prediction target."
# #tree(node("record", kind: "root", children: (node("signal", type: "Number"), node("rank", kind: "target", type: "Ordinal"),)))
# ```
#
# ## Comparison
#
# Keep numeric evidence, initialization seed, sample counts, and updates matched across Ordinal, rounded
# Number, and Category decoders. Category can name only the target classes actually learned, providing
# an explicit output-vocabulary information boundary. Evaluate familiar and withheld ranks separately.
#
# ## Result
#
# {{< evidence P100 status >}}
#
# The numerical checks below are provisional, predeclared hypotheses, not measured successes. No full
# learning result is claimed until a matching run is recorded. Smoke runs establish execution only.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P100 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P100 script >}}
#
# ### Population, controls, and scoring
#
# Each of six fits receives 4,096 training, 512 validation, and 2,048 independent test rows per
# condition, with 400 updates. Inputs equal the generating rank plus Gaussian noise of standard
# deviation 0.03. Integer training ranks are odd and test ranks even, all within the observed numeric
# range; this is interpolation, not arbitrary extrapolation. Declared labels have ranks 0 through 4 with
# gaps preserved. Number outputs are rounded with ties to even, matching Ordinal decoding. Report exact
# accuracy and rank error. These gates are provisional and do not establish that order is appropriate
# for every categorical domain.
#
# ### Complete experiment code

# %%
"""P100: interpolate integer ranks and declared ordered labels absent from training."""

from collections.abc import Iterator

import lightning.pytorch as lit
import numpy as np
import polars as pl
import pyarrow as pa
import relflow as rf
from relflow.presets.base import resolve
from reporting import experiment, report

BUDGET = 400
TRAIN_ROWS = 4096
TEST_ROWS = 2048
LABELS = ("very_low", "low", "medium", "high", "very_high")


def records(*, rows: int, seed: int, domain: str, novel: bool, arm: str) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    ranks = np.arange(-8, 9, 2) if novel else np.arange(-9, 10, 2)
    if domain == "labels":
        ranks = np.asarray([1, 3] if novel else [0, 2, 4])
    selected = rng.choice(ranks, rows)
    for rank, error in zip(selected, rng.normal(0, 0.03, rows), strict=True):
        value = int(rank)
        target = LABELS[value] if domain == "labels" and arm != "number" else value
        if arm == "category" and domain == "integer":
            target = f"rank_{value}"
        yield {"signal": float(value + error), "rank": target}


def build(domain: str, arm: str) -> rf.Model:
    if not hasattr(rf, "Ordinal"):
        raise RuntimeError(
            "P100 requires the stage/0.5.0 relflow wheel exporting rf.Ordinal; the ordinary core wheel cannot execute this research experiment"
        )
    if arm == "ordinal":
        target = (
            rf.Ordinal(values=LABELS, mask=True, objective="mse")
            if domain == "labels"
            else rf.Ordinal(mask=True, objective="mse")
        )
    elif arm == "number":
        target = rf.Number(mask=True, objective="mse")
    else:
        target = rf.Category(mask=True)
    # Freeze the experimental architecture independently of SDK preset changes.
    control_preset = rf.presets.Preset(
        name="xs",
        d_model=64,
        root=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        branch=rf.presets.BranchDefaults(n_layers=1, n_heads=2, reduction=rf.Attention()),
        leaf=rf.presets.LeafDefaults(n_heads=2),
    )
    model = rf.Model(schema=resolve(control_preset, signal=rf.Number, rank=target))
    model.preset = control_preset
    return model


def prediction(model: rf.Model, rows: list[dict], domain: str, arm: str) -> np.ndarray:
    output = pl.DataFrame(
        model.predict(pa.Table.from_pylist([{"signal": row["signal"]} for row in rows]), batch_size=256)
    )["predictions"].to_list()
    values = [row["rank"]["content"] for row in output]
    if arm == "category":
        values = [item["value"] for item in values]
    if domain == "labels" and arm != "number":
        return np.asarray([LABELS.index(value) for value in values], dtype=np.int64)
    if arm == "category":
        return np.asarray([int(value.removeprefix("rank_")) for value in values], dtype=np.int64)
    return np.rint(np.asarray(values, dtype=np.float64)).astype(np.int64)


def actual(rows: list[dict], domain: str, arm: str) -> np.ndarray:
    if domain == "labels" and arm != "number":
        return np.asarray([LABELS.index(row["rank"]) for row in rows], dtype=np.int64)
    if arm == "category":
        return np.asarray([int(row["rank"].removeprefix("rank_")) for row in rows], dtype=np.int64)
    return np.asarray([row["rank"] for row in rows], dtype=np.int64)


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


@experiment("P100", seed=10001)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics = {"budget": BUDGET, "domains": {}}
    checks = {}
    for domain in ("integer", "labels"):
        metrics["domains"][domain] = {}
        for arm in ("ordinal", "number", "category"):
            lit.seed_everything(seed, workers=True)
            model = build(domain, arm)
            train = list(records(rows=TRAIN_ROWS, seed=seed + 1, domain=domain, novel=False, arm=arm))
            validation = list(records(rows=512, seed=seed + 2, domain=domain, novel=False, arm=arm))
            updates = fit(
                model, pa.Table.from_pylist(train), pa.Table.from_pylist(validation), seed, steps, accelerator
            )
            scores = {"steps": updates}
            for condition, novel in (("seen", False), ("unseen", True)):
                rows = list(records(rows=TEST_ROWS, seed=seed + 3, domain=domain, novel=novel, arm=arm))
                predicted = prediction(model, rows, domain, arm)
                expected = actual(rows, domain, arm)
                scores[condition] = {
                    "exact_accuracy": float(np.mean(predicted == expected)),
                    "rank_mae": float(np.mean(np.abs(predicted - expected))),
                }
            metrics["domains"][domain][arm] = scores
            checks[f"{domain}, {arm}: full budget retained"] = updates == BUDGET
            checks[f"{domain}, {arm}: seen accuracy at least 0.85"] = scores["seen"]["exact_accuracy"] >= 0.85
            if arm != "category":
                checks[f"{domain}, {arm}: unseen rank accuracy at least 0.80"] = (
                    scores["unseen"]["exact_accuracy"] >= 0.80
                )
            else:
                checks[f"{domain}: Category cannot name unseen target classes"] = (
                    scores["unseen"]["exact_accuracy"] == 0.0
                )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
