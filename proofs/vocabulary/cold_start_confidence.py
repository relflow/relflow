# %% [markdown]
# ---
# title: Cold-Start Confidence
# categories: [Vocabulary]
# proof-id: P061
# description: Unseen input identities should lose identity-specific information without losing decisive evidence from other fields.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P061 status >}}
#
# ## Insights
#
# OOV is not a reason to force every prediction toward 50%. When identity is
# informative, its absence removes that information; strong numerical context
# can still support confident predictions. This experiment knows both the
# identity-conditional and identity-marginal probabilities, allowing direct
# probability-error measurement without treating sampled outcomes as certainty.
#
# Ordinary uniform input unavailability is compared at 1% and 20%. Neither
# policy uses label frequencies or validation vocabulary. This is an empirical
# fallback check, not a claim that 20% is optimal across datasets.

# %%
"""P061: confidence with sparse known IDs, unknown IDs, and decisive context."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P061"
BUDGET = 600
ENTITIES = 4096

# %% [markdown]
# ## Examples
#
# ```yaml
# x: 0.0
# entity: id-42
# label: yes
# ```
#
# A familiar identity can have a persistent positive or negative effect.
#
# ```yaml
# x: 0.0
# entity: new-42
# label: no
# ```
#
# An unseen identity has no inferable sign: its identity-marginal probability
# at x=0 is 0.5. The ID spelling conveys no semantics.
#
# ```yaml
# x: 2.8
# entity: new-42
# label: yes
# ```
#
# Large positive x supports high confidence even for a new identity. Blanket
# OOV confidence suppression would be wrong here.
#
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-cold-start-confidence
# //| fig-cap: "Numerical evidence remains useful when categorical identity is unavailable."
# //| fig-alt: "A root contains Number x, a large-vocabulary Category entity, and an always-hidden binary Category label."
# #tree(node("record", kind: "root", children: (
#   node("x", type: "Number", detail: "Generalizable evidence"),
#   node("entity", type: "Category", detail: "Persistent identity"),
#   node("label", kind: "target", type: "Category", detail: "Known yes/no labels"),
# )))
# ```


# %%
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
        batch_size=128,
        x=rf.Number,
        entity=rf.Category(p_unavailable=unavailable),
        label=rf.Category(mask=True),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def probability(model: rf.Model, rows: list[dict]) -> np.ndarray:
    predictions = model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
    content = [row["record/label"]["content"] for row in predictions]
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


# %% [markdown]
# ## Training and controls
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


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {}, {}
    for unavailable in (0.01, 0.2):
        lit.seed_everything(seed, workers=True)
        model = build(unavailable)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=32768, seed=seed + 1),
            validate=partial(records, rows=1024, seed=seed + 2),
            seed=seed,
        )
        trainer = lit.Trainer(
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
        trainer.fit(model, datamodule=data)
        model.eval()
        vocabulary = rf.Category.vocabulary(model, "record/entity")
        counts = rf.Category.counts(model, "record/entity")
        arm = {}
        for novel in (False, True):
            rows = list(records(rows=8192, seed=seed + 3, novel=novel))
            arm["novel" if novel else "known"] = score(probability(model, rows), rows, novel)
        rows = list(records(rows=2048, seed=seed + 4, novel=True))
        before = probability(model, rows)
        renamed = [{**row, "entity": "another-unseen-identity"} for row in rows]
        arm["renaming_drift"] = float(np.max(np.abs(before - probability(model, renamed))))
        checks[f"{unavailable}: evaluation preserves mapping and exposures"] = vocabulary == rf.Category.vocabulary(
            model, "record/entity"
        ) and counts == rf.Category.counts(model, "record/entity")
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


# %% [markdown]
# ## Evidence and remaining work
#
# {{< proof P061 evidence >}}
#
# Gates remain provisional. Uniform masking assumes that observed identity
# effects represent the missing-identity population. Time drift, unseen groups
# with different effects, and rare known-ID overfitting can violate this.
# Newly admitted random embeddings still need training. This probe's training
# encode also updates Number normalization, so admission drift is not isolated
# to identity. P066 holds other input resources fixed and gates prediction and
# reconstruction-NLL stability; the current implementation fails that gate.
# No output-OOV detection or universal probability calibration is established.
#
# ## Reproduce
#
# {{< proof P061 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7611)
