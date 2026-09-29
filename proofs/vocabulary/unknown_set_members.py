# %% [markdown]
# ---
# title: Unknown Set identities and an untrained fallback
# categories: [Vocabulary and OOV]
# proof-id: P058
# description: Known membership bits omit unknown identities; an untrained neutral fallback still coincides with empty input.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P058 status >}}
#
# ## Insights
#
# Unknown members have no identity-specific membership bit. Set now retains
# their count and has a learned unavailable-input contribution. This experiment
# disables augmentation and never trains on unknown members, so that initially
# neutral contribution remains untrained: unknown and empty inputs still have
# identical predictions here. P064 tests learning that distinction.
#
# Unknown target identities still cannot supply positive-label supervision or
# appear in predictions. Historical runs predate the unavailable contribution;
# the current checks compare membership bits, not the complete content tree.
#
# ## Setup

# %%
"""P058: learn known membership while exposing empty/all-OOV and null boundaries."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P058"
BUDGET = 400
PATTERNS = ((), ("red",), ("blue",), ("red", "blue"), None)

# %% [markdown]
# ## Data and model
#
# Each record samples one of five equally likely patterns: empty, red, blue,
# both, or null. `target = 1*red - 1.5*blue` for valued sets and 3 for null.
# A separate masked Set head reconstructs the original membership. Inputs and
# targets have independent, automatically grown vocabularies. Disable simulated
# unavailability to isolate actual OOV behavior. Splits contain 4,096 training,
# 512 validation, and 2,048 test records.
#
# ```yaml
# tags: [red, never-seen]
# labels: [red, never-seen]
# target: 1.0
# ```
#
# Only red has a represented membership bit. Unknown labels cannot appear in
# the reconstruction head's output candidates.
#
# ```yaml
# tags: [never-seen]
# labels: [never-seen]
# target: 0.0
# ```
#
# The known-membership bits are identical to an empty list. The target here explicitly
# scores only known membership; other applications may need a different policy.
#
# ```yaml
# tags: null
# labels: null
# target: 3.0
# ```
#
# Null has its own state and is learnably different from empty and all-unknown.
#
# ```{typst}
# //| label: fig-proof-unknown-set-members
# //| fig-cap: "Known membership supports regression and reconstruction; unknown membership has no bit."
# //| fig-alt: "Record has Set tags, hidden Set labels, and hidden Number target. With no unavailable-input training, empty and unknown sets produce the same embedding."
# #tree(node("record", kind: "root", children: (
#   node("tags", type: "Set"),
#   node("labels", kind: "target", type: "Set"),
#   node("target", kind: "target", type: "Number"),
# )))
# ```


# %%
def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for index in rng.integers(len(PATTERNS), size=rows):
        pattern = PATTERNS[index]
        tags = None if pattern is None else list(pattern)
        target = 3.0 if pattern is None else float(("red" in pattern) - 1.5 * ("blue" in pattern))
        yield {"tags": tags, "labels": None if tags is None else list(tags), "target": target}


def build() -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        batch_size=64,
        tags=rf.Set(p_unavailable=0.0),
        labels=rf.Set(p_unavailable=0.0, threshold=0.5, mask=True),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, list[set[str]]]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    values = np.asarray([row["/target"]["content"] for row in output], dtype=np.float64)
    labels = [{candidate["value"] for candidate in row["/labels"]["content"]} for row in output]
    return values, labels


# %% [markdown]
# ## Training and controls
#
# Fit for 400 AdamW updates at learning rate 0.002. Known membership must give
# regression nRMSE below 0.15 and at least 0.95 exact-set accuracy on valued
# targets. Shuffling tags removes regression accuracy. Adding OOV members and
# duplicates must leave predictions unchanged. An all-OOV target and an empty
# target must have the same encoded membership bits, without learning labels
# from evaluation. Null versus empty predictions must remain well separated.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    model.optimizer = rf.adamw(learning_rate=0.002)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
        seed=seed,
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
    vocabularies = {name: rf.Set.vocabulary(model, f"/{name}") for name in ("tags", "labels")}
    train = list(records(rows=4096, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    actual = np.asarray([row["target"] for row in test])
    baseline = float(np.sqrt(np.mean((actual - np.mean([row["target"] for row in train])) ** 2)))
    values, labels = prediction(model, test)
    augmented = [
        {**row, "tags": None if row["tags"] is None else [*row["tags"], *row["tags"], "never-seen"]} for row in test
    ]
    repeated, repeated_labels = prediction(model, augmented)
    order = np.random.default_rng(seed + 4).permutation(len(test))
    shuffled = [{**row, "tags": test[index]["tags"]} for row, index in zip(test, order, strict=True)]
    broken, _ = prediction(model, shuffled)
    probes = [
        {"tags": [], "labels": [], "target": 0.0},
        {"tags": ["never-seen"], "labels": ["never-seen"], "target": 0.0},
        {"tags": None, "labels": None, "target": 3.0},
        {"tags": ["red", "never-seen"], "labels": ["red", "never-seen"], "target": 1.0},
    ]
    boundary, boundary_labels = prediction(model, probes)
    fields = model.encode(pa.Table.from_pylist(probes), strata="test")
    input_field = fields["/tags"]
    target_bits = fields["/labels"].targets[rf.TensorKey.content]["membership"]
    metrics = {
        "steps": trainer.global_step,
        "vocabularies": vocabularies,
        "baseline_rmse": baseline,
        "nrmse": float(np.sqrt(np.mean((values - actual) ** 2))) / baseline,
        "known_exact_set_accuracy": float(
            np.mean(
                [
                    predicted == set(row["labels"])
                    for row, predicted in zip(test, labels, strict=True)
                    if row["labels"] is not None
                ]
            )
        ),
        "shuffled_nrmse": float(np.sqrt(np.mean((broken - actual) ** 2))) / baseline,
        "unknown_and_duplicate_drift": float(np.max(np.abs(values - repeated))),
        "boundary_predictions": dict(
            zip(("empty", "all_unknown", "null", "partial_unknown"), boundary.tolist(), strict=True)
        ),
        "all_unknown_target_exact_match": boundary_labels[1] == {"never-seen"},
    }
    return metrics, {
        "Known membership regression nRMSE below 0.15": metrics["nrmse"] < 0.15,
        "Known valued sets reconstruct above 0.95 exact accuracy": metrics["known_exact_set_accuracy"] >= 0.95,
        "Shuffled tags remove predictive accuracy": metrics["shuffled_nrmse"] > 0.85,
        "Unknown and duplicate members leave regression unchanged": metrics["unknown_and_duplicate_drift"] < 1e-5,
        "Unknown and duplicate members leave reconstructed sets unchanged": labels == repeated_labels,
        "All-unknown and empty inputs have identical known membership bits": bool(
            input_field.state[:2].eq(rf.Tokens.valued).all()
        )
        and bool((input_field.content["membership"][0] == input_field.content["membership"][1]).all()),
        "All-unknown and empty targets have identical membership bits": bool((target_bits[0] == target_bits[1]).all()),
        "Untrained fallback leaves all-unknown and empty predictions identical": abs(boundary[0] - boundary[1]) < 1e-5
        and boundary_labels[0] == boundary_labels[1],
        "Null remains distinct from valued empty content": abs(boundary[2] - boundary[0]) > 2.0,
        "Unknown target labels cannot be recovered": not metrics["all_unknown_target_exact_match"]
        and all("never-seen" not in row for row in boundary_labels),
        "Evaluation does not grow either vocabulary": all(
            rf.Set.vocabulary(model, f"/{name}") == vocabulary for name, vocabulary in vocabularies.items()
        ),
    }


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P058 evidence >}}
#
# This demonstrates the absence of identity information about unknown members,
# not an unavoidable empty/unknown collision. P063 checks coverage and P064
# trains the new fallback. None can recover novel output names. Gates remain
# provisional; earlier evidence records the previous representation.
#
# ## Reproduce
#
# {{< proof P058 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=5801)
