# %% [markdown]
# ---
# title: Can it learn classes declared in advance?
# categories:
# - Enum
# evidence-id: P069
# description: Declare the allowed input and output classes with Enum, then learn which output belongs to each visible
#   combination.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Informative training
#   metric:
#   - informative
#   - accuracy
#   format: percent
# - label: Shuffled training and test
#   metric:
#   - shuffled_control
#   - shuffled_test_accuracy
#   format: percent
# evidence-group: Representation
# evidence-result:
#   accelerator: cpu
#   checks:
#     Checkpoint preserves declared storage and counts: true
#     Checkpoint preserves held-out predictions: true
#     Informative accuracy exceeds the control by at least 0.55: true
#     Informative accuracy reaches 0.95: true
#     Shuffled-context control stays below 0.35 on independent shuffled test rows: true
#     Shuffling test context removes learned accuracy: true
#     'informative: declaration order and storage remain fixed': true
#     'informative: prediction leaves all exposure counts unchanged': true
#     'shuffled_control: declaration order and storage remain fixed': true
#     'shuffled_control: prediction leaves all exposure counts unchanged': true
#   duration_seconds: 15.352
#   metrics:
#     chance_accuracy: 0.25
#     informative:
#       accuracy: 1.0
#       shuffled_test_accuracy: 0.25
#       steps: 300
#       storage:
#         channel:
#           count_rows: 2
#           decoder_rows: null
#           embedding_rows: 2
#           labels:
#           - web
#           - store
#         label:
#           count_rows: 4
#           decoder_rows: 4
#           embedding_rows: 4
#           labels:
#           - amber
#           - blue
#           - coral
#           - green
#         route:
#           count_rows: 8
#           decoder_rows: null
#           embedding_rows: 8
#           labels:
#           - arch
#           - bay
#           - cove
#           - dune
#           - elm
#           - ford
#           - glen
#           - hill
#       training_counts:
#         channel:
#           store: 9591
#           web: 9609
#         label:
#           amber: 4804
#           blue: 4783
#           coral: 4814
#           green: 4799
#         route:
#           arch: 2409
#           bay: 2400
#           cove: 2413
#           dune: 2391
#           elm: 2390
#           ford: 2395
#           glen: 2406
#           hill: 2396
#     shuffled_control:
#       accuracy: 0.25
#       shuffled_test_accuracy: 0.2568359375
#       steps: 300
#       storage:
#         channel:
#           count_rows: 2
#           decoder_rows: null
#           embedding_rows: 2
#           labels:
#           - web
#           - store
#         label:
#           count_rows: 4
#           decoder_rows: 4
#           embedding_rows: 4
#           labels:
#           - amber
#           - blue
#           - coral
#           - green
#         route:
#           count_rows: 8
#           decoder_rows: null
#           embedding_rows: 8
#           labels:
#           - arch
#           - bay
#           - cove
#           - dune
#           - elm
#           - ford
#           - glen
#           - hill
#       training_counts:
#         channel:
#           store: 9627
#           web: 9573
#         label:
#           amber: 4804
#           blue: 4783
#           coral: 4814
#           green: 4799
#         route:
#           arch: 2401
#           bay: 2415
#           cove: 2395
#           dune: 2395
#           elm: 2408
#           ford: 2408
#           glen: 2394
#           hill: 2384
#     test_rows: 2048
#     train_rows: 4096
#     validation_rows: 512
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: c0d49561f49f8503cd6e9ecab2b1374d918e1354da3b0d8db69308300cef8ec8
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: bd307da698c5
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.2
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     relflow_source_sha256: 77953506049ed76c92adc5f32a9bcc32432a788068a4e366b2f78bcf709f29a2
#     script_sha256: 861c7f6bb7f821b82d60125ffc8c31bd70ea10d0247af9f6d8fe8114346f4b27
#     source_sha256: b0f7484969f9200778ce12ba3fa3de2e45f6a62f8c0891bae34cf44809a2cbaa
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 6901
#   started_at: '2026-11-02T17:54:38.273419+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41494093900-1
#   candidate:
#     payload_sha256: 2169c968a7733085889a6eed6e895bcb2590059bcc46243d49a61ffacc42f67f
#     revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   baseline:
#     payload_sha256: 2a6af4bd3869faac23f6b7327f2c59e8f5b367580b2f2fa609710958775c08ed
#     revision: 7dec5e860dc389c37c3642d8904867249e831773
#     wheel_sha256: d3da50b65e0c0c7b98059f095b82bf6bf5d0ea0c5ccd0f302015118075f1dd85
#   evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#   executor_revision: 92b3c887eabc598ee2fc6c9dba4e05b9166cb8ba
#   manifest_sha256: 4f0926f3fc366fd8d14bf728b2d06f4a475d61852679176704816c110ae51cfd
#   passed: true
#   github_run:
#     event: push
#     head_branch: main
#     head_sha: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     html_url: https://github.com/relflow/relflow/actions/runs/41494093900
#     path: .github/workflows/ci.yml
#     run_attempt: 1
#     run_id: 41494093900
#     workflow_id: 403231172
#   release: https://github.com/relflow/relflow/releases/tag/evidence-ci-41494093900-attempt-1
#   reports_sha256: b1bfee5622ffbd389b1777443dc05e6d8f08dccff9e2540a601f0c2608948594
# ---
#
# ## Example
#
# ```yaml
# route: bay
# channel: store
# label: green
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-enum-fixed-class-learning
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, route, channel, label. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("route", type: "Enum", body: [Eight declared routes]),
#   node("channel", type: "Enum", body: [Two declared channels]),
#   node("label", kind: "target", type: "Enum", body: [Four declared answers]),
# )))
# ```
#
# ## Comparison
#
# Compare informative training with shuffled evidence, and check that fitting and save/load preserve class order and size.
#
# ## Result
#
# {{< evidence P069 status >}}
#
# The declared classes support reliable learning in this small task. This covers new records of familiar combinations, rather than new identities or meanings inferred from label names.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P069 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P069 script >}}
#
# ### Data and observability
#
# Draw balanced route/channel combinations, then set the label index to
# `(route_index + 2 * channel_index) % 4`. All 16 visible combinations appear
# equally often. Their opaque names carry no built-in meaning. Independent row
# streams supply 4,096 training, 512 validation, and 2,048 test records.
# Repeated combinations are intentional: this tests learning a closed finite
# relationship, not generalization to combinations absent during fitting.
#
# The route and channel jointly identify the hidden answer. The negative
# control permutes whole visible contexts between rows while leaving targets
# in place, preserving all input and target marginals but removing their link.
# A constant-label baseline scores exactly 0.25 on the balanced test set.
#
# ### Training and gates
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
#
# ### Complete experiment code
#

# %%
"""P069: learn a finite Enum relationship without changing declared class storage."""

from collections.abc import Iterator
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P069"
BUDGET = 300
ROUTES = ("arch", "bay", "cove", "dune", "elm", "ford", "glen", "hill")
CHANNELS = ("web", "store")
LABELS = ("amber", "blue", "coral", "green")


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
    return [row["label"]["content"] for row in output]


def accuracy(output: list[dict], rows: list[dict]) -> float:
    return float(np.mean([value["value"] == row["label"] for value, row in zip(output, rows, strict=True)]))


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=2048, seed=seed + 3))
    shuffled_test = list(records(rows=2048, seed=seed + 3, shuffled=True))
    metrics = {"chance_accuracy": 0.25, "train_rows": 4096, "validation_rows": 512, "test_rows": 2048}
    checks = {}
    for shuffled in (False, True):
        lit.seed_everything(seed, workers=True)
        model = build()
        initial = storage(model)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=4096, seed=seed + 1, shuffled=shuffled),
            validate=partial(records, rows=512, seed=seed + 2, shuffled=shuffled),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=64,
            learning_rate=0.002,
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
        trainer = model.trainer
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=6901)

# %% [markdown]
# </details>
