# %% [markdown]
# ---
# title: Does pretraining reduce the labels needed for a new task?
# categories:
# - Representation
# description: Use noisy categorical and geometric views to predict a hidden four-way
#   style with a small labeled sample.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Eight-label adaptation
#   metric:
#   - budgets
#   - '8'
#   - fine_tuned
#   - accuracy
#   format: percent
# - label: Eight-label scratch
#   metric:
#   - budgets
#   - '8'
#   - scratch
#   - accuracy
#   format: percent
# evidence-group: Representation
# ---
#
# Reconstruction can produce valid embeddings without making later predictions easier. This experiment asks
# whether unlabeled training reduces the supervised examples needed, rather than merely separating four
# clusters.
#
# ## Example
#
# ```yaml
# first: code-0
# second: code-1
# geometry: [-1.0, -1.0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
# ```
#
# Use noisy categorical and geometric views to predict a hidden four-way style with a small labeled sample.
# The example is illustrative; its answer is hidden during scoring.
#
# ```{typst}
# //| label: fig-evidence-p085
# //| fig-cap: "Visible fields supply evidence; prediction targets are hidden."
# //| fig-alt: "Use noisy categorical and geometric views to predict a hidden four-way style with a small labeled sample."
# #tree(node("record",kind:"root",body:[Export context],children:(node("first",type:"Enum"),node("second",type:"Enum"),node("geometry",type:"Vector(16)"))))
# ```
#
# ## Comparison
#
# Compare a frozen pretrained encoder, the same encoder fine-tuned with labels, an encoder trained from
# scratch, independently shuffled pretraining views, and standardized raw features. Every arm uses the same
# 32-to-four linear softmax head, labeled subsets, optimizer updates, and batch order. Shuffled labels
# remove meaningful supervision.
#
# ## Result
#
# {{< evidence P085 status >}}
#
# No result is claimed before a full run is recorded. The checks below are provisional hypotheses, not
# established capabilities. Lower error and proper loss are better; larger accuracy is better. Controls and
# each evaluation condition remain separate.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P085 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P085 script >}}
#
# ### Process, splits, and interpretation
#
# Four balanced hidden classes generate two independently corrupted eight-way codes and a sixteen-coordinate
# vector with two signal coordinates. Pretraining/validation/labeled-pool/test streams use distinct seeds
# and 4096/512/512/2048 rows. Hidden labels never enter reconstruction. Nested label budgets use the same
# first 8, 32, or 128 examples per class. Pretraining and each downstream fit use 400 fixed updates without
# checkpoint selection. Root context extraction uses the native differentiable forward path for fine-tuning;
# all downstream arms use an identical linear head. Raw scaling fits only unlabeled training records.
# Comparative gains are new provisional gates, not changes to P073. Test evaluation never updates
# parameters; one seed remains provisional.
#
# ### Complete experiment code
# %%
"""P085: compare frozen, adapted, and raw representations at fixed label budgets."""

from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
import torch
from reporting import experiment, report

BUDGET = 400
ADAPT = 400
COUNTS = (8, 32, 128)
CODES = tuple(f"code-{i}" for i in range(8))
CENTERS = np.asarray([[-1.0, -1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, 1.0]])


def equal(left, right) -> bool:
    """Compare complete model state, including non-tensor vocabulary metadata."""
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return (
            left.shape == right.shape
            and left.dtype == right.dtype
            and (
                bool(torch.allclose(left, right, rtol=0.0, atol=0.0, equal_nan=True))
                if left.is_floating_point() or left.is_complex()
                else torch.equal(left, right)
            )
        )
    if isinstance(left, np.ndarray):
        return (
            bool(np.array_equal(left, right, equal_nan=True))
            if left.dtype.kind in "fc"
            else bool(np.array_equal(left, right))
        )
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(equal(left[key], right[key]) for key in left)
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(equal(a, b) for a, b in zip(left, right, strict=True))
    if isinstance(left, float) and np.isnan(left):
        return bool(np.isnan(right))
    return bool(left == right)


def sample(rows: int, seed: int) -> tuple[list[dict], np.ndarray]:
    rng = np.random.default_rng(seed)
    labels = np.resize(np.arange(4), rows)
    rng.shuffle(labels)
    codes = [
        2 * np.where(rng.random(rows) < 0.25, rng.integers(4, size=rows), labels) + rng.integers(2, size=rows)
        for _ in range(2)
    ]
    geometry = rng.normal(0.0, 2.0, (rows, 16))
    geometry[:, :2] = CENTERS[labels] + rng.normal(0.0, 0.8, (rows, 2))
    return [
        {"first": CODES[a], "second": CODES[b], "geometry": vector.tolist()}
        for a, b, vector in zip(*codes, geometry, strict=True)
    ], labels


def records(rows: int, seed: int):
    yield from sample(rows, seed)[0]


def build() -> rf.Model:
    mask = rf.Mask(rate=0.35, reconstruct=True)
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        embed=True,
        first=rf.Enum(values=CODES, mask=mask),
        second=rf.Enum(values=CODES, mask=mask),
        geometry=rf.Vector(n_dim=16, mask=mask),
    )


def features(model: rf.Model, rows: list[dict]) -> torch.Tensor:
    """Keep root context differentiable so the identical linear probe can fine-tune it."""
    batch = model.encode(pa.Table.from_pylist(rows), strata="predict").to(model.device)
    output = model(batch, strata="predict")
    roots = [
        item.payload[rf.TensorKey.embedding]
        for item in output
        if item.address == rf.Address() and rf.TensorKey.embedding in item.payload
    ]
    if len(roots) != 1:
        raise ValueError("P085 requires one root embedding per observation")
    return torch.nn.functional.normalize(roots[0].reshape(len(rows), -1).float(), dim=-1)


def raw(rows: list[dict]) -> np.ndarray:
    identity = np.eye(8)
    return np.asarray(
        [
            np.concatenate([identity[CODES.index(row["first"])], identity[CODES.index(row["second"])], row["geometry"]])
            for row in rows
        ]
    )


def probe(
    model: rf.Model | None,
    train: list[dict],
    labels: np.ndarray,
    test: list[dict],
    actual: np.ndarray,
    seed: int,
    budget: int,
    *,
    tune: bool = False,
    mean=None,
    scale=None,
) -> dict:
    """Train one 32-to-four softmax head with identical initialization and batch order."""
    torch.manual_seed(seed)
    device = model.device if model is not None else torch.device("cpu")
    head = torch.nn.Linear(32, 4).to(device)
    parameters = list(head.parameters())
    if model is not None:
        model.eval().requires_grad_(tune)
        if tune:
            parameters += list(model.parameters())
        else:
            with torch.no_grad():
                fixed = features(model, train).detach()
    else:
        fixed = torch.nn.functional.normalize(
            torch.tensor((raw(train) - mean) / scale, dtype=torch.float32, device=device), dim=-1
        )
    optimizer = torch.optim.AdamW(parameters, lr=0.003, weight_decay=0.01)
    target = torch.tensor(labels, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    for _ in range(budget):
        indices = rng.choice(len(train), size=min(64, len(train)), replace=False)
        inputs = features(model, [train[i] for i in indices]) if tune else fixed[indices]
        loss = torch.nn.functional.cross_entropy(head(inputs), target[indices])
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    before = deepcopy(model.state_dict()) if model is not None else {}
    with torch.no_grad():
        inputs = (
            features(model, test)
            if model is not None
            else torch.nn.functional.normalize(
                torch.tensor((raw(test) - mean) / scale, dtype=torch.float32, device=device), dim=-1
            )
        )
        probability = head(inputs).softmax(-1).cpu().numpy()
    preserved = model is None or equal(before, model.state_dict())
    return {
        "accuracy": float(np.mean(probability.argmax(1) == actual)),
        "nll": float(-np.log(probability[np.arange(len(actual)), actual].clip(1e-7, 1.0)).mean()),
        "head_parameters": sum(p.numel() for p in head.parameters()),
        "encoder_trainable": 0 if model is None or not tune else sum(p.numel() for p in model.parameters()),
        "updates": budget,
        "evaluation_preserves_parameters": preserved,
    }


def fit(model: rf.Model, train: list[dict], validation: list[dict], seed: int, budget: int, accelerator: str) -> int:
    """Fit only the supplied training rows with a fixed optimizer/update budget."""
    data = rf.ArrowDataModule(
        train=pa.Table.from_pylist(train),
        validate=pa.Table.from_pylist(validation),
        num_workers=0,
        persistent_workers=False,
        pin_memory=False,
    )
    model.fit(
        data,
        batch_size=64,
        learning_rate=0.002,
        weight_decay=0.01,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_steps=budget,
        max_epochs=-1,
        logger=False,
        enable_checkpointing=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        deterministic=True,
        num_sanity_val_steps=0,
        limit_val_batches=1,
    )
    model.eval()
    return int(model.trainer.global_step)


@experiment("P085", seed=8501)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    train, _ = sample(4096, seed + 1)
    validation, _ = sample(512, seed + 2)
    pool, labels = sample(4 * 128, seed + 3)
    test, actual = sample(2048, seed + 4)
    prebudget = BUDGET if steps is None else min(steps, BUDGET)
    adapt = ADAPT if steps is None else min(steps, ADAPT)
    training = raw(train)
    mean = training.mean(0)
    scale = np.maximum(training.std(0), 1e-8)
    metrics = {"budgets": {}, "pretraining": {}, "labels_excluded_from_pretraining": True}
    checks = {}
    with TemporaryDirectory(prefix="relflow-p085-") as directory:
        checkpoints = {}
        for arm in ("aligned", "permuted", "scratch"):
            lit.seed_everything(seed, workers=True)
            model = build()
            rows = train
            if arm == "permuted":
                rng = np.random.default_rng(seed + 5)
                orders = [rng.permutation(len(train)) for _ in range(3)]
                rows = [
                    {
                        key: train[order[i]][key]
                        for key, order in zip(("first", "second", "geometry"), orders, strict=True)
                    }
                    for i in range(len(train))
                ]
            if arm != "scratch":
                count = fit(model, rows, validation, seed, prebudget, accelerator)
            else:
                model.encode(pa.Table.from_pylist(train), strata="train")
                count = 0
            path = Path(directory) / f"{arm}.pt"
            model.save(path)
            checkpoints[arm] = path
            metrics["pretraining"][arm] = {"updates": count}
        for count in COUNTS:
            indices = np.concatenate([np.flatnonzero(labels == label)[:count] for label in range(4)])
            selected = [pool[i] for i in indices]
            target = labels[indices]
            scores = {}
            for arm, checkpoint, tune in [
                ("frozen", "aligned", False),
                ("fine_tuned", "aligned", True),
                ("scratch", "scratch", True),
                ("permuted", "permuted", False),
            ]:
                model = rf.Model.load(checkpoints[checkpoint])
                scores[arm] = probe(model, selected, target, test, actual, seed + 50, adapt, tune=tune)
            scores["raw"] = probe(None, selected, target, test, actual, seed + 50, adapt, mean=mean, scale=scale)
            scores["shuffled_labels"] = probe(
                rf.Model.load(checkpoints["aligned"]),
                selected,
                np.random.default_rng(seed + 51).permutation(target),
                test,
                actual,
                seed + 50,
                adapt,
            )
            scores["fine_tuned_gain_over_scratch"] = scores["fine_tuned"]["accuracy"] - scores["scratch"]["accuracy"]
            scores["frozen_gain_over_raw"] = scores["frozen"]["accuracy"] - scores["raw"]["accuracy"]
            metrics["budgets"][str(count)] = scores
            checks[f"{count} labels: finite held-out losses"] = all(
                np.isfinite(scores[name]["nll"])
                for name in ["frozen", "fine_tuned", "scratch", "permuted", "raw", "shuffled_labels"]
            )
            checks[f"{count} labels: evaluation leaves parameters unchanged"] = all(
                scores[name]["evaluation_preserves_parameters"]
                for name in ["frozen", "fine_tuned", "scratch", "permuted", "raw", "shuffled_labels"]
            )
            checks[f"{count} labels: heads have identical parameter counts"] = (
                len(
                    {scores[name]["head_parameters"] for name in ["frozen", "fine_tuned", "scratch", "permuted", "raw"]}
                )
                == 1
            )
        small = metrics["budgets"]["8"]
        large = metrics["budgets"]["128"]
        checks["Adapted pretraining reaches 0.65 accuracy at eight labels"] = small["fine_tuned"]["accuracy"] >= 0.65
        checks["Adapted pretraining exceeds scratch by 0.03 at eight labels"] = (
            small["fine_tuned_gain_over_scratch"] >= 0.03
        )
        checks["At least one learned representation exceeds raw at eight labels"] = (
            max(small["frozen"]["accuracy"], small["fine_tuned"]["accuracy"]) >= small["raw"]["accuracy"] + 0.02
        )
        checks["More labels retain useful adapted quality"] = large["fine_tuned"]["accuracy"] >= 0.70
        checks["Shuffled low-label supervision stays below 0.45 accuracy"] = (
            small["shuffled_labels"]["accuracy"] <= 0.45
        )
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
