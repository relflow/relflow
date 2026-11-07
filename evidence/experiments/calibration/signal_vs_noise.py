# %% [markdown]
# ---
# title: Can it learn a signal without inventing one?
# categories:
# - Calibration
# description: Predict a yes/no answer from a visible yes/no input.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Matching answers
#   metric:
#   - signal_auc
#   format: auc
# - label: Unrelated answers
#   metric:
#   - noise_auc
#   format: auc
# evidence-group: Calibration
# ---
#
# This is a sanity check for interpreting learning experiments: can the same training setup learn an
# obvious relationship and avoid reporting predictive skill when that relationship is removed? The
# visible Boolean is called leak deliberately; it supplies a perfect signal in one arm, rather than
# representing an accidental hidden data leak.
#
# ## Example
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
# leak is the visible yes/no signal; the other inputs are unrelated. target is hidden.
#
# In the signal arm, leak: true always accompanies target: true, and false accompanies false. The
# numeric x, segment, and tags are unrelated distractions. In the noise arm, leak is generated
# independently: the same visible true can accompany either target. A single illustrated row cannot
# show the difference; it is a relationship across many independent examples.
#
# ```{typst}
# //| label: fig-evidence-calibration-signal-vs-noise
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-alt: "Model tree with calibration, x, segment, tags, leak, target. Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("calibration", kind: "root", children: (
#   node("x", type: "Number"),
#   node("segment", type: "Category"),
#   node("tags", type: "Set"),
#   node("leak", type: "Boolean", width: 150pt,),
#   node("target", kind: "target", type: "Boolean",),
# )))
# ```
#
# ## Comparison
#
# Train one model where input and answer match, and another where they are unrelated. Both receive
# the same training budget.
#
# Two separately trained models use the same schema, initialization seed, and training budget. Only
# the relationship between leak and target changes. Train, validation, and test use independent
# streams, so remembering random training labels cannot solve the noise test. The target is
# balanced, preventing a dominant class from masquerading as strong classification.
#
# ## Result
#
# {{< evidence P017 status >}}
#
# The real signal is learned; unrelated answers remain near chance. This basic check makes harder
# learning results easier to interpret.
#
#
# AUC measures whether positive targets tend to receive higher scores than negative ones: 0.5 is
# chance ranking and 1 is perfect ranking. The checks require high signal AUC, noise AUC near 0.5,
# and a large gap between them. This basic control does not prove calibrated probabilities or
# establish that more complex tasks are free from leakage.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P017 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P017 script >}}
#
# ### How it works
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
# ### Remaining work
#
# Run three paired core seeds and a calibration panel of at least ten seeds.
# Record the resulting distributions before freezing the gates, and reconcile
# the implemented 2,048 test rows with the evidence plan's 4,096-row requirement.
#
# ### Complete experiment code
#

# %%
"""P017: distinguish an obvious Boolean signal from independent noise.

Only the relationship between ``leak`` and ``target`` changes. Both models use
independent train, validation, and test streams and the same training budget.
"""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import experiment, report


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


def fit(*, signal: bool, seed: int, steps: int | None, accelerator: str) -> float:
    """Fit one relationship and measure its held-out Boolean AUC."""
    lit.seed_everything(seed, workers=True)
    model = rf.Model.xs(
        x=rf.Number,
        segment=rf.Category(p_unavailable=0.0),
        tags=rf.Set(p_unavailable=0.0),
        leak=rf.Boolean,
        target=rf.Boolean(mask=True),
    )
    data = rf.SyntheticDataModule(
        train=partial(records, rows=1024, seed=seed + 1, signal=signal),
        validate=partial(records, rows=512, seed=seed + 2, signal=signal),
        test=partial(records, rows=2048, seed=seed + 3, signal=signal),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        optimizer=lambda module: torch.optim.AdamW(module.parameters(), lr=0.005),
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="ddp" if accelerator in ("gpu", "auto") and torch.cuda.device_count() > 1 else "auto",
        accelerator=accelerator,
        max_epochs=12,
        max_steps=steps if steps is not None else -1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
    )
    trainer = model.trainer
    metrics = trainer.test(model=model, datamodule=data, verbose=False)[0]
    return float(metrics[".target/test.auc.content"])


@experiment("P017", seed=7)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    signal_auc = fit(signal=True, seed=seed, steps=steps, accelerator=accelerator)
    noise_auc = fit(signal=False, seed=seed, steps=steps, accelerator=accelerator)
    gap = signal_auc - noise_auc
    return {"signal_auc": signal_auc, "noise_auc": noise_auc, "auc_gap": gap}, {
        "Signal AUC is at least 0.98": signal_auc >= 0.98,
        "Independent noise AUC remains between 0.42 and 0.58": 0.42 <= noise_auc <= 0.58,
        "Signal exceeds noise by at least 0.40 AUC": gap >= 0.40,
    }


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
