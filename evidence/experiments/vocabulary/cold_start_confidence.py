# %% [markdown]
# ---
# title: How confident should it be about a new identity?
# categories:
# - Vocabulary
# evidence-id: P061
# description: Predict a yes/no outcome using a numeric signal and an identity-specific effect. New identities remove
#   only the latter information.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Weak signal, new identity, 20% training rate
#   metric:
#   - '0.2'
#   - novel
#   - ambiguous_confidence
#   format: percent
# - label: Strong signal, new identity, 20% training rate
#   metric:
#   - '0.2'
#   - novel
#   - decisive_confidence
#   format: percent
# evidence-group: Vocabulary
# evidence-result:
#   accelerator: cpu
#   checks:
#     '0.01: evaluation preserves mapping and exposures': true
#     '0.01: unseen spelling is uninformative': true
#     '0.2: evaluation preserves mapping and exposures': true
#     '0.2: unseen spelling is uninformative': true
#     Ambiguous unknown IDs retain confidence below 0.70: true
#     Decisive context retains confidence above 0.90: true
#     Known-ID probability RMSE below 0.30: true
#     Unknown-ID predictions beat the constant baseline: true
#     Unknown-ID probability RMSE below 0.10: true
#   duration_seconds: 31.477
#   metrics:
#     '0.01':
#       admission_mean_drift: 0.0003071454702876508
#       known:
#         accuracy: 0.802978515625
#         ambiguous_confidence: 0.6822458485209202
#         brier: 0.13774539864425028
#         constant_rmse: 0.41307056095828854
#         decisive_confidence: 0.979890597188566
#         nll: 0.4295551679401508
#         oracle_rmse: 0.24325100463038368
#       novel:
#         accuracy: 0.7852783203125
#         ambiguous_confidence: 0.5714815263234233
#         brier: 0.14561742906462838
#         constant_rmse: 0.3257906780529331
#         decisive_confidence: 0.9823842794232616
#         nll: 0.4490004669477017
#         oracle_rmse: 0.06678319238509788
#       renaming_drift: 0.0
#     '0.2':
#       admission_mean_drift: 0.0002874272468034178
#       known:
#         accuracy: 0.797119140625
#         ambiguous_confidence: 0.6495810806708707
#         brier: 0.13915601835738278
#         constant_rmse: 0.41307056095828854
#         decisive_confidence: 0.9797623936131904
#         nll: 0.43151307221366303
#         oracle_rmse: 0.24652841853289564
#       novel:
#         accuracy: 0.77880859375
#         ambiguous_confidence: 0.5768502663919366
#         brier: 0.14515934309868395
#         constant_rmse: 0.3257906780529331
#         decisive_confidence: 0.9813166697057495
#         nll: 0.44575804827618704
#         oracle_rmse: 0.060065833648231204
#       renaming_drift: 0.0
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 59d0294b92debef0b8e862c7a4f55242c270e7bc6a6414348aa8c05771f9a1d0
#     cuda: null
#     evidence_revision: d2857b43bd13e08d36a8bd921c6ecbfa16e11595
#     gpu: null
#     host: 3dbd002cd3eb
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
#     script_sha256: 23bdffa796469f1c6b0177837edefc95ac2b3fc62d171a1fedb5743576d8177c
#     source_sha256: 6666797683e0cec1b37c03cdd0651f4e3a62ccfdf0cc8f00290e14984f7ba4e7
#     wheel_sha256: 2d56b21a29d294a412f67d17223f38d49b25ae4b2151cee12eb6e6c19073355d
#   seed: 7611
#   started_at: '2026-11-02T17:54:05.948701+00:00'
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
# x: 0.0
# entity: id-42
# label: yes
# ```
#
# The outcome is hidden and randomly sampled. Scores compare predicted probabilities with the generator’s known probabilities.
#
# ```{typst}
# //| label: fig-evidence-cold-start-confidence
# //| fig-cap: "Amber cards are hidden prediction targets. The numeric signal remains available for unfamiliar identities."
# //| fig-alt: "Model tree with record, x, entity, label. Amber cards are hidden prediction targets. The numeric signal remains available for unfamiliar identities."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Generalizable evidence"),
#   node("entity", type: "Category", detail: "Persistent identity"),
#   node("label", kind: "target", type: "Category", detail: "Known yes/no labels"),
# )))
# ```
#
# ## Comparison
#
# Compare training with 1% or 20% of input identities made unavailable. Score new identities separately where the numeric signal is weak or strong.
#
# ## Result
#
# {{< evidence P061 status >}}
#
# A new identity does not always imply a 50% prediction: strong remaining context can still justify confidence. Neither unavailability rate is claimed to be best for every dataset.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P061 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P061 script >}}
#
# ### Training and controls
#
# Use 32,768 training rows over 4,096 identities, automatically growing storage,
# and 600 updates. Independent validation has 1,024 rows. Evaluation has 8,192
# known-ID rows and 8,192 entity-disjoint rows, with identical visible x and
# known yes/no output labels. The latent sign is only available to the evaluator.
#
# The 20% arm must beat the constant baseline, retain decisive confidence above
# 0.9, and keep ambiguous new-ID confidence below 0.7. Record both arms without
# demanding that one stochastic run of stronger masking always beats 1%.
# A final admission-only probe measures the prediction change without learning;
# this is a diagnostic limitation, not a promise that admission is safe.
#
# ### Complete experiment code
#

# %%
"""P061: confidence with sparse known IDs, unknown IDs, and decisive context."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P061"
BUDGET = 600
ENTITIES = 4096


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    identities = rng.integers(0, ENTITIES, rows)
    effects = np.random.default_rng(918).choice([-2.5, 2.5], ENTITIES)
    values = rng.uniform(-3, 3, rows)
    probabilities = 1 / (1 + np.exp(-(2 * values + effects[identities])))
    labels = rng.random(rows) < probabilities
    for x, identity, label in zip(values, identities, labels, strict=True):
        yield {"x": float(x), "entity": f"{'new' if novel else 'id'}-{identity}", "label": "yes" if label else "no"}


def build(unavailable: float) -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        x=rf.Number,
        entity=rf.Category(p_unavailable=unavailable),
        label=rf.Category(mask=True),
    )
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = [row["label"]["content"] for row in predictions]
    return np.asarray([item["probability"] if item["value"] == "yes" else 1 - item["probability"] for item in content])


def score(p: np.ndarray, rows: list[dict], novel: bool) -> dict[str, float]:
    x = np.asarray([row["x"] for row in rows])
    if novel:
        oracle = (1 / (1 + np.exp(-(2 * x - 2.5))) + 1 / (1 + np.exp(-(2 * x + 2.5)))) / 2
    else:
        effects = np.random.default_rng(918).choice([-2.5, 2.5], ENTITIES)
        identity = np.asarray([int(row["entity"].split("-")[1]) for row in rows])
        oracle = 1 / (1 + np.exp(-(2 * x + effects[identity])))
    y = np.asarray([row["label"] == "yes" for row in rows], dtype=float)
    clipped = p.clip(1e-7, 1 - 1e-7)
    confidence = np.maximum(p, 1 - p)
    return {
        "oracle_rmse": float(np.sqrt(np.mean((p - oracle) ** 2))),
        "constant_rmse": float(np.sqrt(np.mean((0.5 - oracle) ** 2))),
        "nll": float(np.mean(-y * np.log(clipped) - (1 - y) * np.log1p(-clipped))),
        "brier": float(np.mean((p - y) ** 2)),
        "accuracy": float(np.mean((p > 0.5) == y)),
        "ambiguous_confidence": float(confidence[np.abs(x) < 0.5].mean()),
        "decisive_confidence": float(confidence[np.abs(x) > 2.5].mean()),
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for unavailable in (0.01, 0.2):
        lit.seed_everything(seed, workers=True)
        model = build(unavailable)
        data = rf.SyntheticDataModule(
            train=partial(records, rows=32768, seed=seed + 1),
            validate=partial(records, rows=1024, seed=seed + 2),
            seed=seed,
        )
        model.fit(
            data,
            batch_size=128,
            learning_rate=0.002,
            early_stopping=False,
            rollback=False,
            precision="32-true",
            strategy="auto",
            accelerator=accelerator,
            devices=1,
            max_epochs=-1,
            max_steps=BUDGET if steps is None else min(steps, BUDGET),
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            deterministic=True,
            num_sanity_val_steps=0,
        )
        model.eval()
        vocabulary = rf.Category.vocabulary(model, "/entity")
        counts = rf.Category.counts(model, "/entity")
        arm = {}
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            arm["novel" if novel else "known"] = score(probability(model, rows), rows, novel)
        rows = list(records(rows=2048, seed=seed + 4, novel=True))
        before = probability(model, rows)
        renamed = [{**row, "entity": "another-unseen-identity"} for row in rows]
        arm["renaming_drift"] = float(np.max(np.abs(before - probability(model, renamed))))
        checks[f"{unavailable}: evaluation preserves mapping and exposures"] = vocabulary == rf.Category.vocabulary(
            model, "/entity"
        ) and counts == rf.Category.counts(model, "/entity")
        checks[f"{unavailable}: unseen spelling is uninformative"] = arm["renaming_drift"] < 1e-5
        model.encode(pa.Table.from_pylist(rows), strata="train")
        arm["admission_mean_drift"] = float(np.mean(np.abs(before - probability(model, rows))))
        metrics[str(unavailable)] = arm
        if unavailable == 0.2:
            checks.update(
                {
                    "Known-ID probability RMSE below 0.30": arm["known"]["oracle_rmse"] < 0.30,
                    "Unknown-ID probability RMSE below 0.10": arm["novel"]["oracle_rmse"] < 0.10,
                    "Unknown-ID predictions beat the constant baseline": arm["novel"]["oracle_rmse"]
                    < arm["novel"]["constant_rmse"],
                    "Ambiguous unknown IDs retain confidence below 0.70": arm["novel"]["ambiguous_confidence"] < 0.70,
                    "Decisive context retains confidence above 0.90": arm["novel"]["decisive_confidence"] > 0.90,
                }
            )
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7611)

# %% [markdown]
# </details>
