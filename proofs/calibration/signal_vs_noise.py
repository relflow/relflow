# %% [markdown]
# ---
# title: Signal and Noise
# categories:
# - Calibration
# proof-id: P017
# description: A held-out evaluation should recognize an obvious signal and leave independent
#   noise at chance.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# ---
#
# Before interpreting a difficult learning task, check that the evaluation can
# separate an easy relationship from noise. This proof changes one Boolean
# input while keeping the model, distractors, and training budget the same.
#
# {{< proof P017 status >}}
#
# ## Insights
#
# **A meaningful learning check should recognize a real signal and remain at chance when that signal is
# absent.** The two models have the same schema and training budget; only the relationship between the
# visible Boolean and the target changes.
#
# The positive-signal and independent-noise checks distinguish useful learning from a model that appears
# successful regardless of the data. Separate train, validation, and test streams help distinguish learned
# relationships from memorized observations.
#
# Use this as a basic check before interpreting harder proof failures. Copying a Boolean does not establish
# aggregation or relational reasoning, and the one-seed result does not measure training efficiency.
#
# ## Setup

# %%
"""P017: distinguish an obvious Boolean signal from independent noise.

Only the relationship between ``leak`` and ``target`` changes. Both models use
independent train, validation, and test streams and the same training budget.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import torch
from reporting import report

import relflow as rf

PROOF_ID = "P017"

# %% [markdown]
# ## Examples
#
# These records illustrate the two generating processes. `target` is hidden
# from embedding by `mask=True`; the displayed labels are supervision.
#
# ### Visible positive signal
#
# ```yaml
# x: -0.7
# segment: segment-3
# tags:
#   - round
#   - hot
# leak: true
# target: true
# ```
#
# In the signal process, `leak` always equals `target`. The other fields are
# independent distractors.
#
# ### Visible negative signal
#
# ```yaml
# x: 0.2
# segment: segment-1
# tags:
#   - red
# leak: false
# target: false
# ```
#
# The rule applies to both labels. Copying the visible Boolean would solve
# this positive-control task.
#
# ### An independent-noise record
#
# ```yaml
# x: -0.7
# segment: segment-3
# tags:
#   - round
#   - hot
# leak: true
# target: false
# ```
#
# This record is possible in the separately generated noise process. Here the
# Boolean and target are independent, so either target can accompany the same
# inputs. This is a sampled label, not a reliably inferable answer.
#
# ## Synthetic data and controls


# %%
def records(*, rows: int, seed: int, signal: bool) -> Iterator[dict]:
    """Yield balanced labels with matching or independent Boolean inputs."""
    rng = np.random.default_rng(seed)
    target = np.tile(np.array([False, True]), (rows + 1) // 2)[:rows]
    rng.shuffle(target)
    independent = rng.integers(0, 2, size=rows).astype(bool)
    tags = np.array(["red", "blue", "round", "square", "hot", "cold"])
    tag_rows = [rng.choice(tags, size=int(rng.integers(0, 4)), replace=False).tolist() for _ in range(rows)]
    values = rng.normal(size=rows)
    segments = rng.integers(0, 4, size=rows)
    for index in range(rows):
        yield {
            "x": float(values[index]),
            "segment": f"segment-{segments[index]}",
            "tags": tag_rows[index],
            "leak": bool(target[index] if signal else independent[index]),
            "target": bool(target[index]),
        }


# %% [markdown]
# ## Model tree
#
# ```{typst}
# //| label: fig-proof-calibration-signal-vs-noise
# //| fig-cap: "The two variants change only the leak–target relationship. The target is always hidden from model inputs."
# //| fig-alt: "Calibration contains Number x, Category segment, Set tags, a Boolean leak that equals the target in the signal variant and is independent in the control, and a Boolean target always hidden from input."
# #tree(node("calibration", kind: "root", children: (
#   node("x", type: "Number"),
#   node("segment", type: "Category"),
#   node("tags", type: "Set"),
#   node("leak", type: "Boolean", width: 150pt, body: [
#     - *Signal:* equals target
#     - *Control:* independent
#   ]),
#   node("target", kind: "target", type: "Boolean", body: [
#     - *Input:* always hidden
#   ]),
# )))
# ```
#
# ## How it works
#
# The positive model can learn to copy the visible Boolean through the shared
# context. A separately trained control sees the same schema, but `leak` is
# independent of the balanced target. That model should have no reliable way
# to rank positive examples above negative ones.
#
# Each model trains on 1,024 rows, validates on 512, and tests on 2,048. The
# three splits use independent random streams. Both runs use the `xs` preset
# and 12 deterministic epochs, so a difference in the information available
# explains the contrast.
#
# ## Training and evaluation


# %%
def fit(*, signal: bool, seed: int, steps: int | None, accelerator: str) -> float:
    """Fit one relationship and measure its held-out Boolean AUC."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        batch_size=128,
        x=rf.Number,
        segment=rf.Category(p_unavailable=0.0),
        tags=rf.Set(p_unavailable=0.0),
        leak=rf.Boolean,
        target=rf.Boolean(mask=True),
    )
    model.optimizer = lambda module: torch.optim.AdamW(module.parameters(), lr=5e-3)
    data = rf.SyntheticDataModule(
        model=model,
        train=partial(records, rows=1024, seed=seed + 1, signal=signal),
        validate=partial(records, rows=512, seed=seed + 2, signal=signal),
        test=partial(records, rows=2048, seed=seed + 3, signal=signal),
        seed=seed,
    )
    trainer = lit.Trainer(
        accelerator=accelerator,
        max_epochs=12,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer.fit(model=model, datamodule=data)
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics[".target/test.auc.content"])


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    signal_auc = fit(signal=True, seed=seed, steps=steps, accelerator=accelerator)
    noise_auc = fit(signal=False, seed=seed, steps=steps, accelerator=accelerator)
    gap = signal_auc - noise_auc
    return {"signal_auc": signal_auc, "noise_auc": noise_auc, "auc_gap": gap}, {
        "Signal AUC is at least 0.98": signal_auc >= 0.98,
        "Independent noise AUC remains between 0.42 and 0.58": 0.42 <= noise_auc <= 0.58,
        "Signal exceeds noise by at least 0.40 AUC": gap >= 0.40,
    }


# %% [markdown]
# ## Evidence
#
# {{< proof P017 evidence >}}
#
# ## Remaining work
#
# Run three paired core seeds and a calibration panel of at least ten seeds.
# Record the resulting distributions before freezing the gates, and reconcile
# the implemented 2,048 test rows with the proof plan's 4,096-row requirement.
#
# ## Reproduce
#
# {{< proof P017 script >}}

# %%
if __name__ == "__main__":
    report(PROOF_ID, run, seed=7)
