# %% [markdown]
# ---
# title: Learning a closed set of classes
# categories: [Enum]
# proof-id: P069
# description: Declared Enum inputs predict a hidden Enum target while shuffled context removes the signal.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P069 status >}}
#
# ## Insights
#
# A fixed declaration supplies stable class IDs and a complete output head.
# Learning still has to associate visible evidence with each answer. This proof
# trains an Enum-to-Enum classifier, compares it with shuffled evidence, and
# checks that fitting and checkpoint restoration never grow or reorder classes.
# The claim concerns new records drawn from familiar declared combinations,
# rather than unseen identities or semantic interpretation of label strings.
#
# Across ten CPU seeds, informative accuracy was 100%; the shuffled-training
# control scored 23.68–25.00% on independently shuffled test rows, around the
# 25% chance baseline. All gates passed on those seeds and in a separate
# RTX 3090 run, with the original gates and training budget unchanged.

# %%
"""P069: learn a finite Enum relationship without changing declared class storage."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P069"
BUDGET = 300
ROUTES = ("arch", "bay", "cove", "dune", "elm", "ford", "glen", "hill")
CHANNELS = ("web", "store")
LABELS = ("amber", "blue", "coral", "green")

# %% [markdown]
# ## Data and observability
#
# Draw balanced route/channel combinations, then set the label index to
# `(route_index + 2 * channel_index) % 4`. All 16 visible combinations appear
# equally often. Their opaque names carry no built-in meaning. Independent row
# streams supply 4,096 training, 512 validation, and 2,048 test records.
# Repeated combinations are intentional: this tests learning a closed finite
# relationship, not generalization to combinations absent during fitting.
#
# ```yaml
# route: bay
# channel: store
# label: green
# ```
#
# Keeping the route and changing the channel changes the required label:
#
# ```yaml
# route: bay
# channel: web
# label: blue
# ```
#
# A shuffled control can instead pair the original answer with another context:
#
# ```yaml
# route: arch
# channel: web
# label: green
# ```
#
# The route and channel jointly identify the hidden answer. The negative
# control permutes whole visible contexts between rows while leaving targets
# in place, preserving all input and target marginals but removing their link.
# A constant-label baseline scores exactly 0.25 on the balanced test set.
#
# ```{typst}
# //| label: fig-proof-enum-fixed-class-learning
# //| fig-cap: "Two declared input vocabularies identify a hidden four-class answer."
# //| fig-alt: "Record contains Enum route with eight values, Enum channel with two values, and a hidden Enum label with four values."
# #tree(node("record", kind: "root", children: (
#   node("route", type: "Enum", body: [Eight declared routes]),
#   node("channel", type: "Enum", body: [Two declared channels]),
#   node("label", kind: "target", type: "Enum", body: [Four declared answers]),
# )))
# ```


# %%
def records(*, rows: int, seed: int, shuffled: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    combinations = np.resize(np.arange(len(ROUTES) * len(CHANNELS)), rows)
    rng.shuffle(combinations)
    routes = combinations % len(ROUTES)
    channels = combinations // len(ROUTES)
    labels = (routes + 2 * channels) % len(LABELS)
    context = rng.permutation(rows) if shuffled else np.arange(rows)
    for row, source in enumerate(context):
        yield {"route": ROUTES[routes[source]], "channel": CHANNELS[channels[source]], "label": LABELS[labels[row]]}


def build() -> rf.Model:
    return rf.Model.xs(
        batch_size=64,
        route=rf.Enum(values=ROUTES, p_unavailable=0.0),
        channel=rf.Enum(values=CHANNELS, p_unavailable=0.0),
        label=rf.Enum(values=LABELS, p_unavailable=0.0, topk=[4], mask=True),
    )


def storage(model: rf.Model) -> dict:
    """Inspect declared IDs and parameter/count row counts, excluding learned values."""
    result = {}
    for name in ("route", "channel", "label"):
        address = rf.Address(name)
        node = model.nodes[address]
        result[name] = {
            "labels": rf.Enum.vocabulary(model, address),
            "embedding_rows": node.embedder.embeddings["content"].num_embeddings,
            "count_rows": node.embedder.counters["content"].counts.numel(),
            "decoder_rows": node.decoder.linears["content"].out_features if name == "label" else None,
        }
    return result


def prediction(model: rf.Model, rows: list[dict]) -> list[dict]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    return [row["/label"]["content"] for row in output]


def accuracy(output: list[dict], rows: list[dict]) -> float:
    return float(np.mean([value["value"] == row["label"] for value, row in zip(output, rows, strict=True)]))


# %% [markdown]
# ## Training and gates
#
# Both models use the `xs` preset and 300 AdamW updates at learning rate 0.002,
# batch size 64, one device, and no loader workers. Their parameter initialization
# and source rows are paired; only the context permutation differs. Input
# unavailability augmentation is disabled to isolate the information control.
# No checkpoint or hyperparameter is selected from the test split.
#
# Predeclared gates require at least 0.95 held-out accuracy with real context,
# at most 0.35 on independently shuffled test context for both trained models,
# and a 0.55 gap between informative accuracy and the matched shuffled control.
# The shuffled-trained model's informative-test accuracy is diagnostic: with
# only 16 distinct contexts, accidental agreement with the finite mapping need
# not concentrate around chance merely because test rows repeat those contexts.
# Declaration order, allocated rows, and evaluation-frozen exposure counts are
# separate mechanical gates. A checkpoint must preserve those invariants and
# held-out probabilities.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=2048, seed=seed + 3))
    shuffled_test = list(records(rows=2048, seed=seed + 3, shuffled=True))
    metrics = {"chance_accuracy": 0.25, "train_rows": 4096, "validation_rows": 512, "test_rows": 2048}
    checks = {}
    for shuffled in (False, True):
        lit.seed_everything(seed, workers=True)
        model = build()
        initial = storage(model)
        model.optimizer = rf.adamw(learning_rate=0.002)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=4096, seed=seed + 1, shuffled=shuffled),
            validate=partial(records, rows=512, seed=seed + 2, shuffled=shuffled),
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
        key = "shuffled_control" if shuffled else "informative"
        counts = {name: rf.Enum.counts(model, rf.Address(name)) for name in ("route", "channel", "label")}
        output = prediction(model, test)
        measured = accuracy(output, test)
        permuted = accuracy(prediction(model, shuffled_test), shuffled_test)
        metrics[key] = {
            "steps": trainer.global_step,
            "accuracy": measured,
            "shuffled_test_accuracy": permuted,
            "storage": storage(model),
            "training_counts": counts,
        }
        checks[f"{key}: declaration order and storage remain fixed"] = storage(model) == initial
        checks[f"{key}: prediction leaves all exposure counts unchanged"] = counts == {
            name: rf.Enum.counts(model, rf.Address(name)) for name in counts
        }
        if not shuffled:
            with TemporaryDirectory(prefix="relflow-enum-") as directory:
                path = Path(directory) / "model.pt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device)
                loaded.eval()
                restored = prediction(loaded, test)
                checks["Checkpoint preserves declared storage and counts"] = storage(loaded) == initial and counts == {
                    name: rf.Enum.counts(loaded, rf.Address(name)) for name in counts
                }
                checks["Checkpoint preserves held-out predictions"] = all(
                    before["value"] == after["value"] and abs(before["probability"] - after["probability"]) < 1e-5
                    for before, after in zip(output, restored, strict=True)
                )
    informative = metrics["informative"]["accuracy"]
    control = metrics["shuffled_control"]["shuffled_test_accuracy"]
    checks["Informative accuracy reaches 0.95"] = informative >= 0.95
    checks["Shuffled-context control stays below 0.35 on independent shuffled test rows"] = control <= 0.35
    checks["Shuffling test context removes learned accuracy"] = metrics["informative"]["shuffled_test_accuracy"] <= 0.35
    checks["Informative accuracy exceeds the control by at least 0.55"] = informative - control >= 0.55
    return metrics, checks


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P069 evidence >}}
#
# This finite task does not establish robustness to undeclared input values,
# which Enum rejects, or transfer to new combinations and labels. It also does
# not establish probability calibration or synchronization across distributed ranks.
#
# ## Reproduce
#
# {{< proof P069 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=6901)
