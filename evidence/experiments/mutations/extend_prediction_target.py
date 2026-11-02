# %% [markdown]
# ---
# title: What happens when a new prediction target is added?
# categories:
# - Schema mutation
# evidence-id: P047
# description: Add hidden target w to a model already trained to predict u and v from a and b.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: New target before training
#   metric:
#   - arms
#   - extended
#   - before
#   - w
#   - nrmse
#   format: error
# - label: New target after training
#   metric:
#   - arms
#   - extended
#   - after
#   - w
#   - nrmse
#   format: error
# evidence-group: Mutations
# evidence-result:
#   accelerator: cpu
#   checks:
#     Added head parameters actually learn: true
#     Diagnostic source pooling scale restores old predictions: true
#     Extended checkpoint preserves predictions: true
#     Extended checkpoint preserves schema and learned state: true
#     Extension immediately preserves old predictions: true
#     Extension preserves every existing state entry: true
#     Hidden target placeholders cannot affect predictions: true
#     Source learns both original tasks below 0.25 nRMSE: true
#     Source optimizer covers current parameters: true
#     'continuation: fresh optimizer covers current parameters': true
#     'extended: all final targets below 0.25 nRMSE': true
#     'extended: fresh optimizer covers current parameters': true
#     'scratch_adaptation: fresh optimizer covers current parameters': true
#     'scratch_adaptation: schema and field order match extension': true
#     'scratch_total: all final targets below 0.25 nRMSE': true
#     'scratch_total: fresh optimizer covers current parameters': true
#     'scratch_total: schema and field order match extension': true
#   duration_seconds: 78.371
#   metrics:
#     arms:
#       continuation:
#         after:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.00957171446484702
#             rmse: 0.012121009026939572
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 0.01184593197621459
#             rmse: 0.015354321871258673
#         before:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.020343005405744612
#             rmse: 0.02576108523333932
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 0.02251114482544502
#             rmse: 0.029178233003061153
#         initial_old_prediction_drift:
#           u: 0.0
#           v: 0.0
#         optimizer_covers_current_parameters: true
#         steps: 256
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.019351463694204672
#               rmse: 0.025358996410222125
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.02283003550008477
#               rmse: 0.029007544471489385
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.042027326101909526
#               rmse: 0.05507442892129822
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.06509475761718794
#               rmse: 0.08270854753749246
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.010685957749210126
#               rmse: 0.014003341994391816
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.005074596360603439
#               rmse: 0.0064477157560493435
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.009018890784129204
#               rmse: 0.011818745219123213
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.011808293946662489
#               rmse: 0.01500346382680654
#           step: 256
#       extended:
#         after:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.011476954166494774
#             rmse: 0.014533683131141987
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 0.023748860596694673
#             rmse: 0.03078252098775151
#           w:
#             baseline_rmse: 1.8289556685792812
#             nrmse: 0.02686363295158643
#             rmse: 0.04913239376543717
#         before:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.020343005405744612
#             rmse: 0.02576108523333932
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 0.02251114482544502
#             rmse: 0.029178233003061153
#           w:
#             baseline_rmse: 1.8289556685792812
#             nrmse: 1.0637946473351922
#             rmse: 1.945633250447997
#         changed_existing_state_entries: []
#         initial_old_prediction_drift:
#           u: 0.0
#           v: 0.0
#         new_head_updated_parameters:
#         - embedder.embeddings.weight
#         - embedder.linear.weight
#         - embedder.scalar.weight
#         - decoder.context_projection.0.weight
#         - decoder.context_projection.1.weight
#         - decoder.context_projection.1.bias
#         - decoder.context_projection.3.weight
#         - decoder.context_projection.3.bias
#         - decoder.pool.queries
#         - decoder.pool.blocks.0.attention_norm.weight
#         - decoder.pool.blocks.0.ffn_norm.weight
#         - decoder.pool.blocks.0.attention.head_scale
#         - decoder.pool.blocks.0.attention.q_proj.weight
#         - decoder.pool.blocks.0.attention.q_proj.bias
#         - decoder.pool.blocks.0.attention.k_proj.weight
#         - decoder.pool.blocks.0.attention.k_proj.bias
#         - decoder.pool.blocks.0.attention.v_proj.weight
#         - decoder.pool.blocks.0.attention.v_proj.bias
#         - decoder.pool.blocks.0.attention.out_proj.weight
#         - decoder.pool.blocks.0.attention.out_proj.bias
#         - decoder.pool.blocks.0.attention.q_norm.weight
#         - decoder.pool.blocks.0.attention.k_norm.weight
#         - decoder.pool.blocks.0.ffn.0.weight
#         - decoder.pool.blocks.0.ffn.0.bias
#         - decoder.pool.blocks.0.ffn.3.weight
#         - decoder.pool.blocks.0.ffn.3.bias
#         - decoder.pool.norm.weight
#         - decoder.classification.weight
#         - decoder.classification.bias
#         - decoder.regression.weight
#         - decoder.regression.bias
#         optimizer_covers_current_parameters: true
#         pooling_scale_control:
#           extended_capacity: 2
#           old_prediction_drift_with_source_capacity:
#             u: 0.0
#             v: 0.0
#           source_capacity: 2
#         steps: 256
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.019351463694204672
#               rmse: 0.025358996410222125
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.02283003550008477
#               rmse: 0.029007544471489385
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 1.0572570203037266
#               rmse: 1.9251934537835258
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.05207456524587171
#               rmse: 0.06824076638344499
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.028988364489227692
#               rmse: 0.036832236729281195
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.11399242949364244
#               rmse: 0.20757249640110714
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.017080650304548292
#               rmse: 0.022383224163401022
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.01078659163823708
#               rmse: 0.013705302238395372
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.007206552652047397
#               rmse: 0.013122644469253332
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.011242001755442023
#               rmse: 0.014732006150281123
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.02366774961594686
#               rmse: 0.030071933069140657
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.027183919010575076
#               rmse: 0.04950007606693989
#           step: 256
#       scratch_adaptation:
#         after:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.07663279853604522
#             rmse: 0.09704289092893363
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 0.021636376649129048
#             rmse: 0.028044386196506937
#           w:
#             baseline_rmse: 1.8289556685792812
#             nrmse: 0.017655127075936087
#             rmse: 0.03229044474502085
#         before:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.9509574923324668
#             rmse: 1.2042319472786187
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 1.2183012852608137
#             rmse: 1.5791235428012558
#           w:
#             baseline_rmse: 1.8289556685792812
#             nrmse: 1.1958415183289888
#             rmse: 2.1871411236702585
#         initial_old_prediction_drift:
#           u: 2.972941364618396
#           v: 3.874038228390918
#         optimizer_covers_current_parameters: true
#         steps: 256
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.9481861341247906
#               rmse: 1.2425441894968343
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 1.2217829909536864
#               rmse: 1.5523814864180465
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 1.2040039176860158
#               rmse: 2.19240961861189
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.03706914587546522
#               rmse: 0.04857701474371898
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.10790508356608132
#               rmse: 0.13710278769523856
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.043629687393450065
#               rmse: 0.07944670685313739
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.022566218565942656
#               rmse: 0.029571750470606815
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.024258886982177503
#               rmse: 0.030823024474129088
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.028031024324868464
#               rmse: 0.051042597492122165
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.07472363670201913
#               rmse: 0.09792109087090556
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.022394812377203213
#               rmse: 0.028454555664618696
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.017450463403573056
#               rmse: 0.03177611239733985
#           step: 256
#       scratch_total:
#         after:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 0.016465449183568473
#             rmse: 0.020850795217472882
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 0.019481216119990676
#             rmse: 0.02525093537177942
#           w:
#             baseline_rmse: 1.8289556685792812
#             nrmse: 0.012089276130194282
#             rmse: 0.02211075010733903
#         before:
#           u:
#             baseline_rmse: 1.2663362526593396
#             nrmse: 1.1295474323570087
#             rmse: 1.4303868626919531
#           v:
#             baseline_rmse: 1.2961683303676375
#             nrmse: 1.1951019704837944
#             rmse: 1.5490533257010533
#           w:
#             baseline_rmse: 1.8289556685792812
#             nrmse: 1.2115082353055224
#             rmse: 2.215794854492517
#         initial_old_prediction_drift:
#           u: 3.4485657620154373
#           v: 3.5955236721569444
#         optimizer_covers_current_parameters: true
#         steps: 768
#         validation_curve:
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 1.1195471091722557
#               rmse: 1.4671030352642593
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 1.204998496933429
#               rmse: 1.5310553278703627
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 1.2006171750902288
#               rmse: 2.1862425896398947
#           step: 0
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.07292020102894163
#               rmse: 0.09555779063262256
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.052846987667688736
#               rmse: 0.06714669125017524
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.09404766719181999
#               rmse: 0.17125443458326622
#           step: 32
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.021180680757718222
#               rmse: 0.02775608169948885
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.018282996634254315
#               rmse: 0.02323013636742937
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.023644794681386885
#               rmse: 0.043055570275224504
#           step: 128
#         - scores:
#             u:
#               baseline_rmse: 1.3104433241303899
#               nrmse: 0.016158436606532756
#               rmse: 0.02117471537941496
#             v:
#               baseline_rmse: 1.2705869192092003
#               nrmse: 0.020244696870878754
#               rmse: 0.025722647027493974
#             w:
#               baseline_rmse: 1.8209322963213432
#               nrmse: 0.012269812272837919
#               rmse: 0.02234249743741055
#           step: 768
#     baseline: Source-training mean for test comparisons; phase-training mean for validation curves
#     downstream_interpretable: true
#     mutation: extend record with hidden Number /w
#     optimizer_policy: New AdamW factory and Trainer for every fit; all available labels rehearsed
#     source:
#       u:
#         baseline_rmse: 1.2663362526593396
#         nrmse: 0.020343005405744612
#         rmse: 0.02576108523333932
#       v:
#         baseline_rmse: 1.2961683303676375
#         nrmse: 0.02251114482544502
#         rmse: 0.029178233003061153
#     source_fit:
#       optimizer_covers_current_parameters: true
#       steps: 512
#       validation_curve:
#       - scores:
#           u:
#             baseline_rmse: 1.295883063514218
#             nrmse: 1.292214157617217
#             rmse: 1.6745584412894436
#           v:
#             baseline_rmse: 1.2898120364298942
#             nrmse: 0.9913997423686597
#             rmse: 1.2787193206205933
#         step: 0
#       - scores:
#           u:
#             baseline_rmse: 1.295883063514218
#             nrmse: 0.09466959946606411
#             rmse: 0.12268073057774713
#           v:
#             baseline_rmse: 1.2898120364298942
#             nrmse: 0.0461261423379143
#             rmse: 0.059494053581520406
#         step: 32
#       - scores:
#           u:
#             baseline_rmse: 1.295883063514218
#             nrmse: 0.017021180930807372
#             rmse: 0.022057460089244446
#           v:
#             baseline_rmse: 1.2898120364298942
#             nrmse: 0.028647231854820883
#             rmse: 0.03694954445674586
#         step: 128
#       - scores:
#           u:
#             baseline_rmse: 1.295883063514218
#             nrmse: 0.020581602559570554
#             rmse: 0.02667135017692836
#           v:
#             baseline_rmse: 1.2898120364298942
#             nrmse: 0.02284597124590518
#             rmse: 0.02946700869689977
#         step: 512
#     source_prerequisite_met: true
#     test_rows: 1024
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 992f3fde60b86f53c391e56bc75d09b7611948f8a7ea1f400344f2ba8c6f43b7
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 036ba4bd8212
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
#     script_sha256: 10231a598b3d57668f501b1cd962db8e891bac696a2b94e17f8d30092c4f3a8b
#     source_sha256: 3bbcc20d514fa1b4fa0f70b4ca9c037ef671c98d404c66208ccb44e4f901db3f
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 4701
#   started_at: '2026-11-01T01:46:31.056659+00:00'
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
# b: 0.25
# u: 1.0
# v: 0.75
# w: 1.75
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-mutations-add-target
# //| fig-alt: "Model tree with record, a, b, u, v, w. Amber cards are hidden prediction targets. Other fields provide the input."
# //| fig-cap: "Amber cards are hidden prediction targets. Other fields provide the input."
# #tree(node("record", kind: "root", children: (
#   node("a", type: "Number"),
#   node("b", type: "Number"),
#   node("u", type: "Number", kind: "target", body: [Existing, always hidden]),
#   node("v", type: "Number", kind: "target", body: [Existing, always hidden]),
#   node("w", type: "Number", kind: "target", body: [Added, always hidden]),
# )))
# ```
#
# ## Comparison
#
# Measure old predictions immediately after the edit and after more training. Keep old targets supervised, and compare unchanged and fresh-model controls.
#
# ## Result
#
# {{< evidence P047 status >}}
#
# Adding w shifts old predictions before training, so the preservation check fails. After training, all three tasks learn. This does not establish retention without old labels.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P047 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P047 script >}}
#
# ### Remaining work
#
# Three GPU seeds reproduce the retention failure and successful adaptation.
# Calibrate the provisional 0.25 nRMSE gate on ten separate seeds.
# Adding visible context, targets in repeated branches, and
# adaptation without old labels remain separate claims. These linear targets
# measure a simple useful extension, not transfer to a new data distribution.
# The failed preservation gate identifies a further design question: should
# permanently hidden outputs change the capacity used to scale visible evidence?
#
# ### Complete experiment code
#

# %%
"""P047: add a hidden regression target to a trained model and adapt."""

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

PROOF_ID = "P047"
TARGETS = ("u", "v", "w")


def records(*, rows: int, seed: int) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    for _ in range(rows):
        a, b = map(float, rng.uniform(-1, 1, size=2))
        yield {"a": a, "b": b, "u": a + 2 * b, "v": 2 * a - b, "w": 3 * a + b}


def build(*, extended: bool = False) -> rf.Model:
    targets = TARGETS if extended else TARGETS[:2]
    return rf.Model(
        d_model=32,
        n_layers=1,
        n_heads=4,
        dropout=0.0,
        fields={
            "a": rf.Number,
            "b": rf.Number,
            **{name: rf.Number(mask=True, objective="mse") for name in targets},
        },
    )


def prediction(model: rf.Model, rows: list[dict], targets: tuple[str, ...], *, corrupt: bool = False) -> dict:
    inputs = [{"a": row["a"], "b": row["b"]} for row in rows]
    if corrupt:
        for row in inputs:
            row.update({name: 1000.0 for name in TARGETS})
    output = model.predict(inputs)["predictions"].to_pylist()
    return {name: np.asarray([row[name]["content"] for row in output], dtype=np.float64) for name in targets}


def scores(rows: list[dict], predicted: dict, means: dict) -> dict:
    measured = {}
    for name, values in predicted.items():
        actual = np.asarray([row[name] for row in rows])
        baseline = float(np.sqrt(np.mean((actual - means[name]) ** 2)))
        rmse = float(np.sqrt(np.mean((actual - values) ** 2)))
        measured[name] = {"rmse": rmse, "baseline_rmse": baseline, "nrmse": rmse / baseline}
    return measured


def equal(first, second) -> bool:
    """Compare all tensor and extension-owned state, including normalizers."""
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
    """Observe fixed validation checkpoints without restarting the optimizer."""

    def __init__(self, rows: list[dict], means: dict, targets: tuple[str, ...], budget: int):
        self.rows, self.means, self.targets = rows, means, targets
        self.checkpoints = {32, 128, budget}
        self.measurements = []

    def on_train_start(self, trainer, pl_module):
        self.measurements.append(
            {
                "step": 0,
                "scores": scores(self.rows, prediction(pl_module, self.rows, self.targets), self.means),
            }
        )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.global_step in self.checkpoints:
            self.measurements.append(
                {
                    "step": trainer.global_step,
                    "scores": scores(self.rows, prediction(pl_module, self.rows, self.targets), self.means),
                }
            )


def fit(model: rf.Model, *, seed: int, split: int, budget: int, accelerator: str, targets: tuple[str, ...]) -> dict:
    lit.seed_everything(seed, workers=True)
    training = list(records(rows=2048, seed=split))
    means = {name: float(np.mean([row[name] for row in training])) for name in TARGETS}
    curve = Curve(list(records(rows=512, seed=split + 1)), means, targets, budget)
    data = rf.SyntheticDataModule(
        train=partial(records, rows=2048, seed=split),
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
    full_budget = 768 if steps is None else min(steps, 768)
    lit.seed_everything(seed, workers=True)
    train = list(records(rows=2048, seed=seed + 1))
    test = list(records(rows=1024, seed=seed + 3))
    means = {name: float(np.mean([row[name] for row in train])) for name in TARGETS}
    scales = {name: float(np.std([row[name] for row in train])) for name in TARGETS}
    source = build()
    source_fit = fit(
        source, seed=seed, split=seed + 1, budget=source_budget, accelerator=accelerator, targets=TARGETS[:2]
    )
    reference = prediction(source, test, TARGETS[:2])
    initial = scores(test, reference, means)
    learned = all(value["nrmse"] < 0.25 for value in initial.values())
    checks = {
        "Source learns both original tasks below 0.25 nRMSE": learned,
        "Source optimizer covers current parameters": source_fit["optimizer_covers_current_parameters"],
    }
    source_state = deepcopy(source.state_dict())
    arms = {}

    with TemporaryDirectory(prefix="relflow-mutation-") as directory:
        checkpoint = Path(directory) / "source.ckpt"
        source.save(checkpoint)
        for index, arm in enumerate(("extended", "continuation", "scratch_adaptation", "scratch_total")):
            lit.seed_everything(seed + 100 + index, workers=True)
            model = (
                build(extended=True).to(source.device)
                if arm.startswith("scratch")
                else rf.Model.load(checkpoint).to(source.device)
            )
            targets = TARGETS[:2] if arm == "continuation" else TARGETS
            if arm == "extended":
                model.extend(rf.where("address") == "/", w=rf.Number(mask=True, objective="mse"))
                extended_schema = model.schema.model_dump()
                retained = model.state_dict()
                changes = [
                    name
                    for name, value in source_state.items()
                    if name not in retained or not equal(value, retained[name])
                ]
                checks["Extension preserves every existing state entry"] = not changes
                head = {name: p.detach().clone() for name, p in model.nodes["/w"].named_parameters()}
            elif arm.startswith("scratch"):
                checks[f"{arm}: schema and field order match extension"] = model.schema.model_dump() == extended_schema
            model.eval()
            before = prediction(model, test, targets)
            drift = {name: float(np.max(np.abs(before[name] - reference[name]))) for name in TARGETS[:2]}
            if arm == "extended":
                checks["Extension immediately preserves old predictions"] = all(
                    np.allclose(reference[name], before[name], rtol=1e-5, atol=1e-6 * scales[name])
                    for name in TARGETS[:2]
                )
                # Isolate a schema-derived scale that is absent from state_dict.
                # Restore the edited scale before any fitting or checkpointing.
                pool = model.nodes["/"].encoder.pool
                capacity = pool.mass_capacity
                source_capacity = source.nodes["/"].encoder.pool.mass_capacity
                try:
                    pool.mass_capacity = source_capacity
                    fixed_scale = prediction(model, test, TARGETS)
                finally:
                    pool.mass_capacity = capacity
                scale_control = {
                    "source_capacity": source_capacity,
                    "extended_capacity": capacity,
                    "old_prediction_drift_with_source_capacity": {
                        name: float(np.max(np.abs(fixed_scale[name] - reference[name]))) for name in TARGETS[:2]
                    },
                }
                checks["Diagnostic source pooling scale restores old predictions"] = all(
                    np.allclose(reference[name], fixed_scale[name], rtol=1e-5, atol=1e-6 * scales[name])
                    for name in TARGETS[:2]
                )
            fitting = fit(
                model,
                seed=seed + 200,
                split=seed + 11,
                budget=full_budget if arm == "scratch_total" else adapt_budget,
                accelerator=accelerator,
                targets=targets,
            )
            after = prediction(model, test, targets)
            final = scores(test, after, means)
            checks[f"{arm}: fresh optimizer covers current parameters"] = fitting["optimizer_covers_current_parameters"]
            arms[arm] = {
                "before": scores(test, before, means),
                "after": final,
                "initial_old_prediction_drift": drift,
                **fitting,
            }
            if arm in ("extended", "scratch_total"):
                checks[f"{arm}: all final targets below 0.25 nRMSE"] = all(
                    value["nrmse"] < 0.25 for value in final.values()
                )
            if arm == "extended":
                arms[arm]["changed_existing_state_entries"] = changes
                arms[arm]["pooling_scale_control"] = scale_control
                changed = [name for name, p in model.nodes["/w"].named_parameters() if not torch.equal(head[name], p)]
                arms[arm]["new_head_updated_parameters"] = changed
                checks["Added head parameters actually learn"] = bool(changed)
                corrupted = prediction(model, test, TARGETS, corrupt=True)
                checks["Hidden target placeholders cannot affect predictions"] = all(
                    np.allclose(after[name], corrupted[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
                path = Path(directory) / "extended.ckpt"
                model.save(path)
                loaded = rf.Model.load(path).to(model.device).eval()
                restored = prediction(loaded, test, TARGETS)
                checks["Extended checkpoint preserves schema and learned state"] = (
                    loaded.schema.model_dump() == model.schema.model_dump()
                    and equal(model.state_dict(), loaded.state_dict())
                )
                checks["Extended checkpoint preserves predictions"] = all(
                    np.allclose(after[name], restored[name], rtol=1e-5, atol=1e-6 * scales[name]) for name in TARGETS
                )
    return {
        "source": initial,
        "source_fit": source_fit,
        "source_prerequisite_met": learned,
        "downstream_interpretable": learned,
        "arms": arms,
        "test_rows": len(test),
        "baseline": "Source-training mean for test comparisons; phase-training mean for validation curves",
        "optimizer_policy": "New AdamW factory and Trainer for every fit; all available labels rehearsed",
        "mutation": "extend record with hidden Number /w",
    }, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=4701)

# %% [markdown]
# </details>
