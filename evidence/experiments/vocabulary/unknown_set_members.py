# %% [markdown]
# ---
# title: What happens to Set members never seen in training?
# categories:
# - Vocabulary
# evidence-id: P058
# description: Train a Set field on known members without ever teaching it how unavailable members should affect prediction.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Prediction for empty input
#   metric:
#   - boundary_predictions
#   - empty
#   format: number
# - label: Prediction for all-unknown input
#   metric:
#   - boundary_predictions
#   - all_unknown
#   format: number
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     All-unknown and empty inputs have identical known membership bits: true
#     All-unknown and empty targets have identical membership bits: true
#     Evaluation does not grow either vocabulary: true
#     Known membership regression nRMSE below 0.15: true
#     Known valued sets reconstruct above 0.95 exact accuracy: true
#     Null remains distinct from valued empty content: true
#     Shuffled tags remove predictive accuracy: true
#     Unknown and duplicate members leave reconstructed sets unchanged: true
#     Unknown and duplicate members leave regression unchanged: true
#     Unknown target labels cannot be recovered: true
#     Untrained fallback leaves all-unknown and empty predictions identical: true
#   duration_seconds: 11.689
#   metrics:
#     all_unknown_target_exact_match: false
#     baseline_rmse: 1.5152353198723871
#     boundary_predictions:
#       all_unknown: 0.012366260163679532
#       empty: 0.012366260163679532
#       'null': 3.045052232681446
#       partial_unknown: 1.0116792574173123
#     known_exact_set_accuracy: 1.0
#     nrmse: 0.017415799624031225
#     shuffled_nrmse: 1.4449872691101777
#     steps: 400
#     unknown_and_duplicate_drift: 0.0
#     vocabularies:
#       labels:
#       - blue
#       - red
#       tags:
#       - blue
#       - red
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 11097c729f34dbcb9e6a6e5afca73a245226bd63f0f13c91426d1dd7f7b10f54
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 1499679ce76b
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
#     script_sha256: 260f927a4bb5fe69c6f205da1bdd39d90533dfa2fcd15ac1207991ced4769da5
#     source_sha256: 40878a19950292f1e7e1079dc8defe6b387ca289bc1bf6fafa948adde36ad557
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 5801
#   started_at: '2026-11-02T17:53:53.903205+00:00'
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
# tags: [red, never-seen]
# labels: [red, never-seen]
# target: 1.0
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-unknown-set-members
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with record, tags, labels, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("tags", type: "Set"),
#   node("labels", kind: "target", type: "Set"),
#   node("target", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Compare empty, all-unknown, and partly known sets. Check which member names the output can actually emit.
#
# ## Result
#
# {{< evidence P058 status >}}
#
# The untrained fallback treats all-unknown input like empty input here. Unknown output members also cannot be named. The next fallback test shows how training can distinguish nonempty input.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P058 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P058 script >}}
#
# ### Data and model
#
# Each record samples one of five equally likely patterns: empty, red, blue,
# both, or null. `target = 1*red - 1.5*blue` for valued sets and 3 for null.
# A separate masked Set head reconstructs the original membership. Inputs and
# targets have independent, automatically grown vocabularies. Disable simulated
# unavailability to isolate actual OOV behavior. Splits contain 4,096 training,
# 512 validation, and 2,048 test records.
#
# ### Training and controls
#
# Fit for 400 AdamW updates at learning rate 0.002. Known membership must give
# regression nRMSE below 0.15 and at least 0.95 exact-set accuracy on valued
# targets. Shuffling tags removes regression accuracy. Adding OOV members and
# duplicates must leave predictions unchanged. An all-OOV target and an empty
# target must have the same encoded membership bits, without learning labels
# from evaluation. Null versus empty predictions must remain well separated.
#
# ### Complete experiment code
#

# %%
"""P058: learn known membership while exposing empty/all-OOV and null boundaries."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P058"
BUDGET = 400
PATTERNS = ((), ("red",), ("blue",), ("red", "blue"), None)


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
        tags=rf.Set(p_unavailable=0.0),
        labels=rf.Set(p_unavailable=0.0, threshold=0.5, mask=True),
        target=rf.Number(mask=True, objective="mse"),
    )


def prediction(model: rf.Model, rows: list[dict]) -> tuple[np.ndarray, list[set[str]]]:
    output = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    values = np.asarray([row["target"]["content"] for row in output], dtype=np.float64)
    labels = [{candidate["value"] for candidate in row["labels"]["content"]} for row in output]
    return values, labels


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    lit.seed_everything(seed, workers=True)
    model = build()
    data = rf.SyntheticDataModule(
        train=partial(records, rows=4096, seed=seed + 1),
        validate=partial(records, rows=512, seed=seed + 2),
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
    target_bits = fields["/labels"].targets[rf.TensorKey.content]["indices"]
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
        and bool((input_field.content["indices"][0] == input_field.content["indices"][1]).all()),
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


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5801)

# %% [markdown]
# </details>
