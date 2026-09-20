# %% [markdown]
# ---
# title: Learning an unavailable Set fallback
# categories: [Vocabulary]
# proof-id: P064
# description: Simulated input unavailability should teach a nonempty unknown-set fallback without inventing member identities.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# {{< proof P064 status >}}
#
# ## Insights
#
# A Set can learn that unavailable membership differs from genuinely empty
# membership. Half the rows are empty; the others contain one of 32 equally
# likely labels. The Number target is one for nonempty sets and zero otherwise.
# Train with and without input unavailability. The latter is a matched negative
# control: a never-trained, neutral fallback should still predict empty.
#
# Independent row splits test known labels; disjoint spellings test all-OOV
# inputs. No identity-dependent target is recoverable in this experiment.
#
# ```yaml
# tags: [tag-12]
# nonempty: 1.0
# ```
#
# ```yaml
# tags: [novel-12, novel-12]
# nonempty: 1.0
# ```
#
# Duplicates must not alter the unavailable-member count or prediction.
#
# ```yaml
# tags: []
# nonempty: 0.0
# ```
#
# ```{typst}
# //| label: fig-proof-set-unavailable-fallback
# //| fig-cap: "An unavailable-member signal preserves nonempty versus empty."
# //| fig-alt: "A root contains Set tags and hidden Number nonempty."
# #tree(node("record", kind: "root", children: (
#   node("tags", type: "Set", detail: "Known or unavailable membership"),
#   node("nonempty", kind: "target", type: "Number"),
# )))
# ```

# %%
"""P064: input corruption teaches an unavailable Set representation."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
from reporting import report

import relflow as rf

PROOF_ID = "P064"
BUDGET = 400


def records(*, rows: int, seed: int, novel: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for label, nonempty in zip(rng.integers(32, size=rows), rng.random(rows) < 0.5, strict=True):
        yield {"tags": [f"{'novel' if novel else 'tag'}-{label}"] if nonempty else [], "nonempty": float(nonempty)}


def build(p_unavailable: float) -> rf.Model:
    model = rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        batch_size=64,
        dropout=0.0,
        tags=rf.Set(size=128, p_unavailable=p_unavailable),
        nonempty=rf.Number(mask=True, objective="mse"),
    )
    model.optimizer = rf.adamw(learning_rate=0.002)
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    return np.asarray(
        [
            row["record/nonempty"]["content"]
            for row in model.predict(pa.Table.from_pylist(rows))["predictions"].to_pylist()
        ]
    )


# %% [markdown]
# ## Training and controls
#
# Each arm trains for 400 AdamW updates on 8,192 rows, with 512 independent
# validation rows. Evaluate 2,048 known and disjoint-label rows. The constant
# 0.5 predictor has RMSE 0.5. Both arms must learn familiar sets (RMSE < 0.10).
# Only the augmented arm must transfer to unknown sets (RMSE < 0.10); its
# gain over the untrained-fallback control must exceed 0.40 RMSE. Duplicate
# and unknown-spelling changes must leave predictions unchanged.


# %%
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    metrics, checks = {"constant_rmse": 0.5}, {}
    for probability in (0.0, 0.5):
        lit.seed_everything(seed, workers=True)
        model = build(probability)
        data = rf.SyntheticDataModule(
            model=model,
            train=partial(records, rows=8192, seed=seed + 1),
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
        vocabulary = rf.Set.vocabulary(model, "record/tags")
        arm = "augmented" if probability else "control"
        for novel in (False, True):
            rows = list(records(rows=2048, seed=seed + 3, novel=novel))
            expected = np.asarray([row["nonempty"] for row in rows])
            values = predict(model, rows)
            key = f"{arm}_{'unknown' if novel else 'known'}"
            metrics[key] = float(np.sqrt(np.mean((values - expected) ** 2)))
            if not novel or probability:
                checks[f"{key}: RMSE below 0.10"] = metrics[key] < 0.10
            if novel:
                changed = [{**row, "tags": ["another", "another"] if row["tags"] else []} for row in rows]
                drift = float(np.max(np.abs(values - predict(model, changed))))
                metrics[f"{arm}_duplicate_and_identity_drift"] = drift
                checks[f"{arm}: unknown spelling and duplicates are invariant"] = drift < 1e-5
        checks[f"{arm}: unknown evaluation leaves vocabulary frozen"] = (
            rf.Set.vocabulary(model, "record/tags") == vocabulary
        )
    checks["Trained fallback improves unknown-input RMSE by more than 0.40"] = (
        metrics["control_unknown"] - metrics["augmented_unknown"] > 0.40
    )
    return metrics, checks


# %% [markdown]
# ## Evidence and limitations
#
# {{< proof P064 evidence >}}
#
# The learned signal says only that some members are unavailable. It cannot
# distinguish their identities, guarantee calibration after population shifts,
# or recover unknown output labels. Gates are provisional.
#
# ## Reproduce
#
# {{< proof P064 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7641)
