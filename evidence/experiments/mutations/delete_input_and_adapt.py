# %% [markdown]
# ---
# title: Can it adapt after losing an input?
# categories:
# - Schema mutation
# evidence-id: P050
# description: Delete b from a model trained on y = a + b, then continue training with only a.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: After deletion and adaptation
#   metric:
#   - arms
#   - deleted
#   - after
#   - nrmse
#   format: error
# - label: Best prediction using only a
#   metric:
#   - only_a_oracle
#   - nrmse
#   format: error
# - label: Both inputs retained
#   metric:
#   - arms
#   - continuation
#   - after
#   - nrmse
#   format: error
# evidence-group: Mutations
# evidence-result:
#   accelerator: cpu
#   checks:
#     Continuation retains full-information accuracy below 0.25 nRMSE: true
#     Deleted checkpoint preserves adapted predictions: true
#     Deleted checkpoint preserves schema and learned state: true
#     Deletion and deactivation initially give the same predictions: true
#     Deletion immediately loses the selected information: true
#     Deletion preserves all unselected state entries: true
#     Deletion removes b from schema and runtime: true
#     Deletion removes the selected state entries: true
#     Inactive control preserves all original state: true
#     Re-added input has fresh normalization: true
#     Re-adding the same name does not restore its learned state: true
#     Scratch schema and field order match deletion: true
#     Source learns both-input relationship below 0.25 nRMSE: true
#     Source optimizer covers current parameters: true
#     Unchanged checkpoint restores source predictions: true
#     'continuation: hidden target values cannot affect predictions': true
#     'continuation: optimizer covers current parameters': true
#     'continuation: parameters actually learn': true
#     'deleted after: removed input values cannot affect predictions': true
#     'deleted before: removed input values cannot affect predictions': true
#     'deleted: adapts near the remaining-information limit': true
#     'deleted: hidden target values cannot affect predictions': true
#     'deleted: optimizer covers current parameters': true
#     'deleted: parameters actually learn': true
#     'inactive after: removed input values cannot affect predictions': true
#     'inactive before: removed input values cannot affect predictions': true
#     'inactive: adapts near the remaining-information limit': true
#     'inactive: hidden target values cannot affect predictions': true
#     'inactive: optimizer covers current parameters': true
#     'inactive: parameters actually learn': true
#     'scratch_restricted after: removed input values cannot affect predictions': true
#     'scratch_restricted before: removed input values cannot affect predictions': true
#     'scratch_restricted: adapts near the remaining-information limit': true
#     'scratch_restricted: hidden target values cannot affect predictions': true
#     'scratch_restricted: optimizer covers current parameters': true
#     'scratch_restricted: parameters actually learn': true
#   duration_seconds: 35.747
#   metrics:
#     arms:
#       continuation:
#         after:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.019169681079402732
#           only_a_distance_nrmse: 0.7128238473981036
#           rmse: 0.015534213953197812
#         before:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.04690872616664413
#           only_a_distance_nrmse: 0.6955476101163599
#           rmse: 0.038012640143897596
#         optimizer_covers_current_parameters: true
#         pooling_capacity: 2
#         steps: 256
#         updated_parameter_count: 94
#         validation_curve:
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.045746016186793664
#             only_a_distance_nrmse: 0.6891507070447545
#             rmse: 0.03744648062484148
#           step: 0
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.05706577145937884
#             only_a_distance_nrmse: 0.664809998078735
#             rmse: 0.04671253331808515
#           step: 32
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.010441750662623133
#             only_a_distance_nrmse: 0.7031202680585448
#             rmse: 0.008547341309038881
#           step: 128
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.019132736975576804
#             only_a_distance_nrmse: 0.7030260545229168
#             rmse: 0.015661553161933184
#           step: 256
#       deleted:
#         after:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.719567417972882
#           only_a_distance_nrmse: 0.06714908852544259
#           rmse: 0.5831038178590885
#         before:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.9093240576886299
#           only_a_distance_nrmse: 0.5522490663075059
#           rmse: 0.7368737333926099
#         changed_unselected_state_entries: []
#         optimizer_covers_current_parameters: true
#         pooling_capacity: 1
#         removed_state_entries:
#         - nodes./b.embedder.weights
#         - nodes./b.embedder.max_fourier_input
#         - nodes./b.embedder.embeddings.weight
#         - nodes./b.embedder.counter.counts
#         - nodes./b.embedder.linear.weight
#         - nodes./b.embedder.linear.bias
#         - nodes./b.embedder.scalar.weight
#         - nodes./b.embedder.normalizers.raw.mean
#         - nodes./b.embedder.normalizers.raw.var
#         - nodes./b.embedder.normalizers.raw.count
#         - nodes./b.embedder.normalizers.log.mean
#         - nodes./b.embedder.normalizers.log.var
#         - nodes./b.embedder.normalizers.log.count
#         steps: 256
#         updated_parameter_count: 77
#         validation_curve:
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.8841830369569963
#             only_a_distance_nrmse: 0.5476132720707423
#             rmse: 0.7237688813606898
#           step: 0
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7303437154999871
#             only_a_distance_nrmse: 0.15018668396591892
#             rmse: 0.5978400759592325
#           step: 32
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7092488580939637
#             only_a_distance_nrmse: 0.07121264687097391
#             rmse: 0.5805723828356837
#           step: 128
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7125514110156407
#             only_a_distance_nrmse: 0.06978506180633609
#             rmse: 0.5832757654316483
#           step: 256
#       inactive:
#         after:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.719567417972882
#           only_a_distance_nrmse: 0.06714908852544259
#           rmse: 0.5831038178590885
#         before:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.9093240576886299
#           only_a_distance_nrmse: 0.5522490663075059
#           rmse: 0.7368737333926099
#         optimizer_covers_current_parameters: true
#         pooling_capacity: 1
#         steps: 256
#         updated_parameter_count: 77
#         validation_curve:
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.8841830369569963
#             only_a_distance_nrmse: 0.5476132720707423
#             rmse: 0.7237688813606898
#           step: 0
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7303437154999871
#             only_a_distance_nrmse: 0.15018668396591892
#             rmse: 0.5978400759592325
#           step: 32
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7092488580939637
#             only_a_distance_nrmse: 0.07121264687097391
#             rmse: 0.5805723828356837
#           step: 128
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7125514110156407
#             only_a_distance_nrmse: 0.06978506180633609
#             rmse: 0.5832757654316483
#           step: 256
#       scratch_restricted:
#         after:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 0.7204181251238119
#           only_a_distance_nrmse: 0.07718237016298927
#           rmse: 0.583793191189783
#         before:
#           baseline_rmse: 0.8103532807277047
#           nrmse: 1.6042761997607558
#           only_a_distance_nrmse: 1.4276334320692323
#           rmse: 1.300030481669503
#         optimizer_covers_current_parameters: true
#         pooling_capacity: 1
#         steps: 256
#         updated_parameter_count: 77
#         validation_curve:
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 1.5605450089836743
#             only_a_distance_nrmse: 1.4161397312300665
#             rmse: 1.2774209278571078
#           step: 0
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7361348586927745
#             only_a_distance_nrmse: 0.17174714400543808
#             rmse: 0.6025805528234671
#           step: 32
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.721017170554056
#             only_a_distance_nrmse: 0.15816735849746685
#             rmse: 0.5902056125955059
#           step: 128
#         - scores:
#             baseline_rmse: 0.8185735883959189
#             nrmse: 0.7148752203886061
#             only_a_distance_nrmse: 0.07890003684775426
#             rmse: 0.5851779744088246
#           step: 256
#     downstream_interpretable: true
#     mutation: delete /b; inactive control updates active=False; separate source fork deletes and re-adds b
#     only_a_oracle:
#       baseline_rmse: 0.8103532807277047
#       nrmse: 0.7146740353052158
#       only_a_distance_nrmse: 0.0
#       rmse: 0.5791384491604891
#     optimizer_policy: New AdamW factory and Trainer per phase; matched adaptation data and updates
#     readdition:
#       fresh_normalization:
#         alpha: null
#         count: 0
#         mean: 0.0
#         std: 1.0
#         variance: 1.0
#       readded_field_order:
#       - /a
#       - /y
#       - /b
#       scores:
#         baseline_rmse: 0.8103532807277047
#         nrmse: 0.6670850754403447
#         only_a_distance_nrmse: 0.5186730092533532
#         rmse: 0.5405745794075717
#       source_field_order:
#       - /a
#       - /b
#       - /y
#       source_normalization:
#         alpha: null
#         count: 65536
#         mean: -0.00040430610009385844
#         std: 0.5771294791922702
#         variance: 0.333078435752741
#     source:
#       baseline_rmse: 0.8103532807277047
#       nrmse: 0.04690872616664413
#       only_a_distance_nrmse: 0.6955476101163599
#       rmse: 0.038012640143897596
#     source_fit:
#       optimizer_covers_current_parameters: true
#       steps: 512
#       validation_curve:
#       - scores:
#           baseline_rmse: 0.8227835642459033
#           nrmse: 1.6047738274734205
#           only_a_distance_nrmse: 1.3347337907024106
#           rmse: 1.320381529577121
#         step: 0
#       - scores:
#           baseline_rmse: 0.8227835642459033
#           nrmse: 0.07994827291428727
#           only_a_distance_nrmse: 0.6784996565794528
#           rmse: 0.06578012494372149
#         step: 32
#       - scores:
#           baseline_rmse: 0.8227835642459033
#           nrmse: 0.0252052298977606
#           only_a_distance_nrmse: 0.682235106345186
#           rmse: 0.020738448892916872
#         step: 128
#       - scores:
#           baseline_rmse: 0.8227835642459033
#           nrmse: 0.04553959670577362
#           only_a_distance_nrmse: 0.6833399299837495
#           rmse: 0.03746923169189741
#         step: 512
#     source_prerequisite_met: true
#     test_rows: 2048
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 42fda1ddfe34891c6cf0458179baeb50991da2c463dd3bf0bc614fa296ed5f88
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: b13714abc154
#     packages:
#       lightning: 2.6.4
#       numpy: 2.4.6
#       polars: 2.0.0
#       pyarrow: 24.0.0
#       pydantic: 2.13.4
#       relflow: 0.4.1
#       tensordict: 0.12.4
#       torch: 2.12.0+cpu
#       torchmetrics: 1.9.0
#     platform: Linux-6.8.0-138-generic-x86_64-with-glibc2.41
#     python: 3.12.16
#     relflow_path: /tmp/runtime/lib/python3.12/site-packages/relflow
#     relflow_revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     relflow_source_sha256: 00fdf397aeefb3a4e705d431850e0a6294eef142989513fddcb903bb1aefa873
#     script_sha256: 0e9cac1b2cb1d68bc7107142e28e29deb65793d8e467ab38b56a9625f2f683ea
#     source_sha256: 1b8935b7a845230bfd4396af9b8f6428d013e9e985aad9eabbf755d16f48eb2a
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 5001
#   started_at: '2026-11-01T01:46:48.586178+00:00'
#   steps_override: null
#   threads: 1
# evidence-report:
#   run_id: gh-41261482839-1
#   candidate:
#     revision: 41d6f5fb74c0ba9f0be104c896913a55adb125f5
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   baseline:
#     revision: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#   manifest_sha256: f4b3bfe706e43e38785ab5bed94c3f51eb44fe7314d0a1b2f0dd3c6f3f7bef92
#   passed: true
#   github_run:
#     event: pull_request
#     head_branch: dev/evidence-in-docs
#     head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#     html_url: https://github.com/relflow/relflow/actions/runs/41261482839
#     pull_requests:
#     - base_sha: 1add6d8919e686dc63b7fa6c4cdbdc437be2b570
#       head_sha: 8bb7764d8186c743eb554cb4a91dedb205ad4cb9
#       number: 64
#     run_attempt: 1
#     run_id: 41261482839
#   release: https://github.com/relflow/docs/releases/tag/ci-41261482839-attempt-1
#   reports_sha256: bc67a1cb3725cbf9c5a8deb65844297d70f6d2122cd3b7de5c1d15f56603864f
# ---
#
# ## Example
#
# ```yaml
# a: 0.5
# b: 0.75
# y: 1.25
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mutations-delete-input
# //| fig-alt: "Model tree with record, a, b, y. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number", body: [Retained input]),
#   node("b", type: "Number", body: [Deleted before adaptation]),
#   node("y", type: "Number", kind: "target", body: [Always hidden; still a + b]),
# )))
# ```
#
# ## Comparison
#
# Compare deletion, deactivation, a fresh model with only a, and an unchanged model. Each receives the same additional training budget.
#
# ## Result
#
# {{< evidence P050 status >}}
#
# The edited model approaches the best remaining-information prediction, a. Re-adding the name creates fresh state; it does not restore what was deleted.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P050 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P050 script >}}
#
# ### Remaining work
#
# Three GPU seeds support this case. Calibrate gates on ten independent seeds.
# Deleting a branch subtree or a learned output requires separate experiments.
# Re-adding `b` appends it after `y`, and also initializes fresh state; its
# prediction difference cannot be attributed solely to either change.
# Relearning after re-addition and transfer advantages are not established.
#
# ### Complete experiment code
#

# %%
"""P050: delete an informative input and learn within the reduced information limit."""

from collections.abc import Iterator
from copy import deepcopy
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory

import lightning.pytorch as lit
import numpy as np
import relflow as rf
import torch
from reporting import report

PROOF_ID = "P050"


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "y": a + b}


def build(*, restricted: bool = False) -> rf.Model:
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        fields={"a": rf.Number, **({} if restricted else {"b": rf.Number}), "y": rf.Number(mask=True, objective="mse")},
    )


def prediction(model: rf.Model, rows: list[dict], *, include_b: bool = True, corrupt: bool = False) -> np.ndarray:
    inputs = [{"a": row["a"], **({"b": row["b"]} if include_b else {})} for row in rows]
    if corrupt:
        for row in inputs:
            row["y"] = 1000.0
    output = model.predict(inputs)["predictions"].to_pylist()
    return np.asarray([row["y"]["content"] for row in output], dtype=np.float64)


def scores(rows: list[dict], predicted: np.ndarray, mean: float) -> dict:
    actual = np.asarray([row["y"] for row in rows])
    conditional = np.asarray([row["a"] for row in rows])
    baseline = float(np.sqrt(np.mean((actual - mean) ** 2)))
    rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
    distance = float(np.sqrt(np.mean((conditional - predicted) ** 2)))
    return {
        "rmse": rmse,
        "baseline_rmse": baseline,
        "nrmse": rmse / baseline,
        "only_a_distance_nrmse": distance / baseline,
    }


def equal(first, second) -> bool:
    """Compare learned tensors and extension-owned nested state exactly."""
    if isinstance(first, torch.Tensor):
        return isinstance(second, torch.Tensor) and torch.equal(first, second)
    if isinstance(first, dict):
        return (
            isinstance(second, dict)
            and first.keys() == second.keys()
            and all(equal(value, second[name]) for name, value in first.items())
        )
    if isinstance(first, (tuple, list)):
        return (
            type(first) is type(second)
            and len(first) == len(second)
            and all(equal(a, b) for a, b in zip(first, second, strict=True))
        )
    return type(first) is type(second) and first == second


class Curve(lit.Callback):
    """Measure validation progress at fixed steps without restarting AdamW."""

    def __init__(self, rows: list[dict], mean: float, budget: int):
        self.rows, self.mean = rows, mean
        self.checkpoints = {32, 128, budget}
        self.measurements = []

    def on_train_start(self, trainer, pl_module):
        self.measurements.append({"step": 0, "scores": scores(self.rows, prediction(pl_module, self.rows), self.mean)})

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.checkpoints:
            self.measurements.append(
                {"step": trainer.global_step, "scores": scores(self.rows, prediction(pl_module, self.rows), self.mean)}
            )


def fit(model: rf.Model, *, seed: int, split: int, rows: int, budget: int, accelerator: str) -> dict:
    lit.seed_everything(seed, workers=True)
    mean = float(np.mean([row["y"] for row in records(rows=rows, seed=split)]))
    curve = Curve(list(records(rows=512, seed=split + 1)), mean, budget)
    data = rf.SyntheticDataModule(
        train=partial(records, rows=rows, seed=split),
        validate=partial(records, rows=512, seed=split + 1),
        seed=seed,
    )
    model.fit(
        data,
        batch_size=128,
        learning_rate=0.003,
        fused=False,
        early_stopping=False,
        rollback=False,
        precision="32-true",
        strategy="auto",
        accelerator=accelerator,
        devices=1,
        max_epochs=-1,
        max_steps=budget,
        callbacks=[curve],
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer = model.trainer
    model.eval()
    optimized = {id(p) for group in trainer.optimizers[0].param_groups for p in group["params"]}
    current = {id(p) for p in model.parameters() if p.requires_grad}
    return {
        "steps": trainer.global_step,
        "optimizer_covers_current_parameters": current == optimized,
        "validation_curve": curve.measurements,
    }


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    source_budget = 512 if steps is None else min(steps, 512)
    adapt_budget = 256 if steps is None else min(steps, 256)
    lit.seed_everything(seed, workers=True)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=2048, seed=seed + 3))
    mean = float(np.mean([row["y"] for row in train]))
    tolerance = 1e-6 * float(np.std([row["y"] for row in train]))
    oracle = scores(test, np.asarray([row["a"] for row in test]), mean)
    source = build()
    source_fit = fit(source, seed=seed, split=seed + 1, rows=2048, budget=source_budget, accelerator=accelerator)
    reference = prediction(source, test)
    initial = scores(test, reference, mean)
    learned = initial["nrmse"] < 0.25
    state = deepcopy(source.state_dict())
    selected_state = deepcopy(source.nodes["/b"].state_dict())
    normalization = rf.Number.normalization(source, "/b")
    checks = {
        "Source learns both-input relationship below 0.25 nRMSE": learned,
        "Source optimizer covers current parameters": source_fit["optimizer_covers_current_parameters"],
    }
    arms = {}
    selected = rf.where("address") == "/b"
    flipped = [{**row, "b": -row["b"]} for row in test]

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        source.save(checkpoint)
        for index, arm in enumerate(("deleted", "inactive", "scratch_restricted", "continuation")):
            lit.seed_everything(seed + 100 + index, workers=True)
            model = (
                build(restricted=True).to(source.device)
                if arm == "scratch_restricted"
                else rf.Model.load(checkpoint).to(source.device)
            )
            if arm == "deleted":
                model.delete(selected)
                deleted_schema = model.schema.model_dump()
                current = model.state_dict()
                removed = [name for name in state if name.startswith("nodes./b.")]
                changes = [
                    name
                    for name, value in state.items()
                    if not name.startswith("nodes./b.") and (name not in current or not equal(value, current[name]))
                ]
                checks["Deletion removes b from schema and runtime"] = (
                    "/b" not in model.schema.requests and "/b" not in model.nodes
                )
                checks["Deletion removes the selected state entries"] = bool(removed) and all(
                    name not in current for name in removed
                )
                checks["Deletion preserves all unselected state entries"] = not changes
            elif arm == "inactive":
                model.update(selected, active=False)
                checks["Inactive control preserves all original state"] = equal(state, model.state_dict())
            elif arm == "scratch_restricted":
                checks["Scratch schema and field order match deletion"] = model.schema.model_dump() == deleted_schema
            model.eval()
            before = prediction(model, test)
            immediate = scores(test, before, mean)
            if arm == "deleted":
                deleted_prediction = before.copy()
                checks["Deletion immediately loses the selected information"] = (
                    immediate["nrmse"] > initial["nrmse"] + 0.35 and immediate["nrmse"] >= 0.9 * oracle["nrmse"]
                )
            elif arm == "inactive":
                checks["Deletion and deactivation initially give the same predictions"] = bool(
                    np.allclose(deleted_prediction, before, rtol=1e-5, atol=tolerance)
                )
            elif arm == "continuation":
                checks["Unchanged checkpoint restores source predictions"] = bool(
                    np.allclose(reference, before, rtol=1e-5, atol=tolerance)
                )
            if arm != "continuation":
                checks[f"{arm} before: removed input values cannot affect predictions"] = bool(
                    np.allclose(before, prediction(model, test, include_b=False), rtol=1e-5, atol=tolerance)
                    and np.allclose(before, prediction(model, flipped), rtol=1e-5, atol=tolerance)
                )
            parameters = {name: p.detach().clone() for name, p in model.named_parameters()}
            fitting = fit(
                model, seed=seed + 200, split=seed + 11, rows=4096, budget=adapt_budget, accelerator=accelerator
            )
            after = prediction(model, test)
            final = scores(test, after, mean)
            updated = [name for name, p in model.named_parameters() if not torch.equal(parameters[name], p)]
            arms[arm] = {
                "before": immediate,
                "after": final,
                **fitting,
                "updated_parameter_count": len(updated),
                "pooling_capacity": model.nodes["/"].encoder.pool.mass_capacity,
            }
            checks[f"{arm}: optimizer covers current parameters"] = fitting["optimizer_covers_current_parameters"]
            checks[f"{arm}: parameters actually learn"] = bool(updated)
            checks[f"{arm}: hidden target values cannot affect predictions"] = bool(
                np.allclose(after, prediction(model, test, corrupt=True), rtol=1e-5, atol=tolerance)
            )
            if arm == "continuation":
                checks["Continuation retains full-information accuracy below 0.25 nRMSE"] = final["nrmse"] < 0.25
            else:
                checks[f"{arm}: adapts near the remaining-information limit"] = (
                    oracle["nrmse"] - 0.05 <= final["nrmse"] <= oracle["nrmse"] + 0.10
                    and final["only_a_distance_nrmse"] < 0.20
                )
                checks[f"{arm} after: removed input values cannot affect predictions"] = bool(
                    np.allclose(after, prediction(model, test, include_b=False), rtol=1e-5, atol=tolerance)
                    and np.allclose(after, prediction(model, flipped), rtol=1e-5, atol=tolerance)
                )
            if arm == "deleted":
                arms[arm]["removed_state_entries"] = removed
                arms[arm]["changed_unselected_state_entries"] = changes
                path = Path(directory) / "deleted.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                checks["Deleted checkpoint preserves schema and learned state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump()
                    and equal(model.state_dict(), loaded.state_dict())
                    and "/b" not in loaded.nodes
                )
                checks["Deleted checkpoint preserves adapted predictions"] = bool(
                    np.allclose(after, prediction(loaded, test), rtol=1e-5, atol=tolerance)
                )

        readded = rf.Model.load(checkpoint).to(source.device).eval()
        readded.delete(selected)
        lit.seed_everything(seed + 300, workers=True)
        readded.extend(rf.where("address") == "/", b=rf.Number)
        fresh_normalization = rf.Number.normalization(readded, "/b")
        checks["Re-adding the same name does not restore its learned state"] = not equal(
            selected_state, readded.nodes["/b"].state_dict()
        )
        fresh = build().to(source.device)
        checks["Re-added input has fresh normalization"] = (
            fresh_normalization == rf.Number.normalization(fresh, "/b") and fresh_normalization != normalization
        )
        readdition = {
            "scores": scores(test, prediction(readded, test), mean),
            "source_normalization": normalization,
            "fresh_normalization": fresh_normalization,
            "source_field_order": [str(node.address) for node in source.schema.fields.fields],
            "readded_field_order": [str(node.address) for node in readded.schema.fields.fields],
        }
    return {
        "source": initial,
        "source_fit": source_fit,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "only_a_oracle": oracle,
        "arms": arms,
        "readdition": readdition,
        "test_rows": len(test),
        "optimizer_policy": "New AdamW factory and Trainer per phase; matched adaptation data and updates",
        "mutation": "delete /b; inactive control updates active=False; separate source fork deletes and re-adds b",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=5001)

# %% [markdown]
# </details>
