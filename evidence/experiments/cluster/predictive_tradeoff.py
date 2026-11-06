# %% [markdown]
# ---
# title: When does grouping identities help prediction?
# categories:
# - Representation
# description: Compare Cluster, Category, and a true-group reference across dense, sparse,
#   and skewed known-identity populations.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Dense Cluster error
#   metric:
#   - conditions
#   - dense
#   - arms
#   - cluster
#   - nrmse
#   format: number
# - label: Dense Category error
#   metric:
#   - conditions
#   - dense
#   - arms
#   - category
#   - nrmse
#   format: number
# evidence-group: Representation
# ---
#
# Several occupied clusters do not establish a useful modeling tradeoff. Here grouping must be judged
# alongside prediction error, ordinary Category performance, actual parameter counts, and how frequently
# each identity was observed.
#
# ## Example
#
# ```yaml
# entity: id-7
# x: 0.5
# answer: 2.6
# ```
#
# Compare Cluster, Category, and a true-group reference across dense, sparse, and skewed known-identity
# populations. The example is illustrative; its answer is hidden during scoring.
#
# ```{typst}
# //| label: fig-evidence-p095
# //| fig-cap: "Visible fields supply evidence; prediction targets are hidden."
# //| fig-alt: "Compare Cluster, Category, and a true-group reference across dense, sparse, and skewed known-identity populations."
# #tree(node("record",kind:"root",children:(node("entity",type:"Cluster or Category"),node("x",type:"Number"),node("answer",kind:"target",type:"Number"))))
# ```
#
# ## Comparison
#
# Fit Cluster, Category, a model given the true group, and a shuffled-ID Category control. Train on dense,
# sparse, and skewed populations, then score every known identity equally. Recovered ARI and gains over
# Category remain diagnostics; the true-group reference validates observability.
#
# ## Result
#
# {{< evidence P095 status >}}
#
# No result is claimed before a full run is recorded. The checks below are provisional hypotheses, not
# established capabilities. Lower error and proper loss are better; larger accuracy is better. Controls and
# each evaluation condition remain separate.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P095 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P095 script >}}
#
# ### Process, splits, and interpretation
#
# Thirty-two opaque identities belong to four fixed balanced groups with intercepts -3,-1,1,3 and slopes
# -0.8,0.8,-0.8,0.8. The example answer 2.6 illustrates intercept three and slope -0.8; it is not a
# generated mapping promise. Independent Gaussian noise has standard deviation 0.15. Train conditions use
# 4,096 balanced,256 sparse,or2048 power-law-skewed observations. Validation 512 and test 2,048 use
# independent draws; test includes each identity equally. All models share width 16, one layer, four heads
# and 500 updates. The oracle is supplied a true-group Enum, so it is an information reference, not
# parameter-matched. Parameters are counted after fitting and not assumed smaller for Cluster.
# Sparse/rare-identity and comparative scores remain visible even when broad capability gates pass.
# Arbitrary unseen identities are outside this task.
#
# ### Complete experiment code
# %%
"""P095: compare Cluster and Category on sparse, skewed known identities."""

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import experiment, report

BUDGET = 500
ENTITIES = 32
GROUPS = 4
INTERCEPT = np.asarray([-3.0, -1.0, 1.0, 3.0])
SLOPE = np.asarray([-0.8, 0.8, -0.8, 0.8])


def partition(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).permutation(np.arange(ENTITIES) % GROUPS)


def records(
    rows: int, seed: int, partition_seed: int, *, skew: bool = False, shuffled: bool = False, balanced: bool = False
):
    rng = np.random.default_rng(seed)
    mapping = partition(partition_seed)
    probability = np.arange(1, ENTITIES + 1, dtype=float) ** (-1.2) if skew else np.ones(ENTITIES)
    probability /= probability.sum()
    entities = np.resize(np.arange(ENTITIES), rows) if balanced else rng.choice(ENTITIES, rows, p=probability)
    rng.shuffle(entities)
    x = rng.uniform(-1.0, 1.0, rows)
    groups = mapping[entities]
    y = INTERCEPT[groups] + SLOPE[groups] * x + rng.normal(0.0, 0.15, rows)
    visible = entities[rng.permutation(rows)] if shuffled else entities
    for entity, group, value, target in zip(visible, mapping[visible], x, y, strict=True):
        yield {"entity": f"id-{entity}", "group": f"group-{group}", "x": float(value), "answer": float(target)}


def build(kind: str) -> rf.Model:
    identity = (
        rf.Cluster(bounds=(2, 8), mask=rf.Mask(rate=0.1, reconstruct=True), p_unavailable=0.0)
        if kind == "cluster"
        else rf.Category(p_unavailable=0.0)
    )
    fields = (
        {"group": rf.Enum(values=tuple(f"group-{i}" for i in range(GROUPS)))}
        if kind == "oracle"
        else {"entity": identity}
    )
    return rf.Model(
        d_model=16,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        fields=fields,
        x=rf.Number,
        answer=rf.Number(mask=True, objective="mse"),
    )


def predict(model: rf.Model, rows: list[dict], kind: str) -> np.ndarray:
    field = "group" if kind == "oracle" else "entity"
    output = model.predict(pa.Table.from_pylist([{field: row[field], "x": row["x"]} for row in rows]))[
        "predictions"
    ].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


def adjusted_rand(actual: np.ndarray, predicted: np.ndarray) -> float:
    _, left = np.unique(actual, return_inverse=True)
    _, right = np.unique(predicted, return_inverse=True)
    cells = np.zeros((left.max() + 1, right.max() + 1), dtype=np.int64)
    np.add.at(cells, (left, right), 1)

    def pair(values):
        return float(np.sum(values * (values - 1) / 2))

    joint = pair(cells)
    a = pair(cells.sum(0))
    b = pair(cells.sum(1))
    total = len(actual) * (len(actual) - 1) / 2
    expected = a * b / total
    maximum = (a + b) / 2
    return 1.0 if maximum == expected else (joint - expected) / (maximum - expected)


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


@experiment("P095", seed=9501)
def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    budget = BUDGET if steps is None else min(steps, BUDGET)
    mapping_seed = seed + 40
    metrics = {"conditions": {}}
    checks = {}
    for condition, rows, skew in [("dense", 4096, False), ("sparse", 256, False), ("skewed", 2048, True)]:
        train = list(records(rows, seed + 1, mapping_seed, skew=skew))
        validation = list(records(512, seed + 2, mapping_seed, skew=skew))
        test = list(records(2048, seed + 3, mapping_seed, balanced=True))
        actual = np.asarray([row["answer"] for row in test])
        scale = float(np.std([row["answer"] for row in train]))
        exposures = np.bincount([int(row["entity"].split("-")[1]) for row in train], minlength=ENTITIES)
        arms = {}
        for kind in ("cluster", "category", "oracle", "shuffled"):
            lit.seed_everything(seed, workers=True)
            model = build(kind)
            fitting = (
                list(records(rows, seed + 1, mapping_seed, skew=skew, shuffled=True)) if kind == "shuffled" else train
            )
            used = fit(model, fitting, validation, seed, budget, accelerator)
            prediction = predict(model, test, kind)
            error = (prediction - actual) ** 2
            per_identity = {
                f"id-{entity}": float(
                    np.sqrt(error[np.asarray([row["entity"] == f"id-{entity}" for row in test])].mean()) / scale
                )
                for entity in range(ENTITIES)
            }
            measured = {
                "updates": used,
                "nrmse": float(np.sqrt(error.mean()) / scale),
                "parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "per_identity_nrmse": per_identity,
                "rare_identity_nrmse": float(np.mean([per_identity[f"id-{i}"] for i in np.argsort(exposures)[:8]])),
            }
            if kind == "cluster":
                assignments = rf.Cluster.assignments(model, rf.Address("entity"))
                assigned = np.asarray([assignments.get(f"id-{i}", {}).get("cluster", -1) for i in range(ENTITIES)])
                measured["ari"] = adjusted_rand(partition(mapping_seed), assigned)
                measured["identity_coverage"] = int((assigned >= 0).sum())
            arms[kind] = measured
            checks[f"{condition} {kind}: finite overall and identity errors"] = bool(
                np.isfinite([measured["nrmse"], *per_identity.values()]).all()
            )
            if kind == "oracle":
                checks[f"{condition}: true-group reference learns below 0.40 nRMSE"] = measured["nrmse"] < 0.40
        arms["cluster_minus_category_nrmse"] = arms["cluster"]["nrmse"] - arms["category"]["nrmse"]
        metrics["conditions"][condition] = {
            "training_rows": rows,
            "identity_exposures": exposures.tolist(),
            "arms": arms,
        }
    dense = metrics["conditions"]["dense"]["arms"]
    checks["Dense Cluster reaches below 0.50 nRMSE"] = dense["cluster"]["nrmse"] < 0.50
    checks["Shuffled-ID dense control remains above 0.80 nRMSE"] = dense["shuffled"]["nrmse"] > 0.80
    return metrics, checks


if __name__ == "__main__":
    report(run)

# %% [markdown]
# </details>
