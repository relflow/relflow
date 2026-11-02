# %% [markdown]
# ---
# title: Can familiar instructions work in new combinations?
# categories:
# - Compositional generalization
# evidence-id: P071
# description: Train on group and operation requests, deliberately leaving out three combinations, such as coral’s
#   maximum.
# execute:
#   enabled: false
#   eval: false
# code-fold: true
# toc: false
# evidence-readout:
# - label: Withheld coral maximum, restricted training
#   metric:
#   - restricted
#   - cells
#   - coral/max
#   - normalized_rmse
#   format: error
# - label: Same request, complete training
#   metric:
#   - complete
#   - cells
#   - coral/max
#   - normalized_rmse
#   format: error
# evidence-group: Composition
# evidence-result:
#   accelerator: cpu
#   checks:
#     'complete: amber/max normalized RMSE below 0.35': true
#     'complete: amber/mean normalized RMSE below 0.35': true
#     'complete: amber/min normalized RMSE below 0.35': true
#     'complete: blue/max normalized RMSE below 0.35': true
#     'complete: blue/mean normalized RMSE below 0.35': true
#     'complete: blue/min normalized RMSE below 0.35': true
#     'complete: coral/max normalized RMSE below 0.35': true
#     'complete: coral/mean normalized RMSE below 0.35': true
#     'complete: coral/min normalized RMSE below 0.35': true
#     'complete: corrupted group normalized RMSE exceeds 0.80': true
#     'complete: corrupting operation increases normalized RMSE': true
#     'complete: corrupting selected_group increases normalized RMSE': true
#     'complete: intact predictions are finite': true
#     'complete: operation control predictions are finite': true
#     'complete: selected_group control predictions are finite': true
#     'restricted: amber/max normalized RMSE below 0.35': true
#     'restricted: amber/mean normalized RMSE below 0.50': true
#     'restricted: amber/min normalized RMSE below 0.35': true
#     'restricted: blue/max normalized RMSE below 0.35': true
#     'restricted: blue/mean normalized RMSE below 0.35': true
#     'restricted: blue/min normalized RMSE below 0.50': true
#     'restricted: coral/max normalized RMSE below 0.50': true
#     'restricted: coral/mean normalized RMSE below 0.35': true
#     'restricted: coral/min normalized RMSE below 0.35': true
#     'restricted: corrupted group normalized RMSE exceeds 0.80': true
#     'restricted: corrupting operation increases normalized RMSE': true
#     'restricted: corrupting selected_group increases normalized RMSE': true
#     'restricted: intact predictions are finite': true
#     'restricted: operation control predictions are finite': true
#     'restricted: selected_group control predictions are finite': true
#   duration_seconds: 122.046
#   metrics:
#     complete:
#       cells:
#         amber/max:
#           baseline_rmse: 0.8831115047625776
#           normalized_rmse: 0.050579054568607486
#           rmse: 0.04466694498955148
#           rows: 256
#           withheld: false
#         amber/mean:
#           baseline_rmse: 1.0017870715984531
#           normalized_rmse: 0.07149707146887575
#           rmse: 0.07162484185467036
#           rows: 256
#           withheld: true
#         amber/min:
#           baseline_rmse: 0.9404379907655642
#           normalized_rmse: 0.06175489987682685
#           rmse: 0.05807665396009163
#           rows: 256
#           withheld: false
#         blue/max:
#           baseline_rmse: 1.0229581561177996
#           normalized_rmse: 0.051611483279818605
#           rmse: 0.05279638777042789
#           rows: 256
#           withheld: false
#         blue/mean:
#           baseline_rmse: 0.8995328184909653
#           normalized_rmse: 0.08605709203223055
#           rmse: 0.07741117854688874
#           rows: 256
#           withheld: false
#         blue/min:
#           baseline_rmse: 0.9387242747363832
#           normalized_rmse: 0.07239985219073719
#           rmse: 0.06796349873877111
#           rows: 256
#           withheld: true
#         coral/max:
#           baseline_rmse: 0.9858002380723858
#           normalized_rmse: 0.05109260825686536
#           rmse: 0.050367105383357015
#           rows: 256
#           withheld: true
#         coral/mean:
#           baseline_rmse: 0.972345983430609
#           normalized_rmse: 0.07460879857698641
#           rmse: 0.07254556562491607
#           rows: 256
#           withheld: false
#         coral/min:
#           baseline_rmse: 0.9469336156041438
#           normalized_rmse: 0.07632444635672943
#           rmse: 0.07227418394756231
#           rows: 256
#           withheld: false
#       corrupt_operation:
#         cells:
#           amber/max:
#             baseline_rmse: 0.8831115047625776
#             normalized_rmse: 0.30483980793448573
#             rmse: 0.26920754149655884
#             rows: 256
#             withheld: false
#           amber/mean:
#             baseline_rmse: 1.0017870715984531
#             normalized_rmse: 0.30635086338433465
#             rmse: 0.3068983343114504
#             rows: 256
#             withheld: true
#           amber/min:
#             baseline_rmse: 0.9404379907655642
#             normalized_rmse: 0.7022572535123195
#             rmse: 0.6604294004936692
#             rows: 256
#             withheld: false
#           blue/max:
#             baseline_rmse: 1.0229581561177996
#             normalized_rmse: 0.24868605729668714
#             rmse: 0.25439543062442455
#             rows: 256
#             withheld: false
#           blue/mean:
#             baseline_rmse: 0.8995328184909653
#             normalized_rmse: 0.32000031228942816
#             rmse: 0.2878507828316984
#             rows: 256
#             withheld: false
#           blue/min:
#             baseline_rmse: 0.9387242747363832
#             normalized_rmse: 0.7368401079344474
#             rmse: 0.6916896959174424
#             rows: 256
#             withheld: true
#           coral/max:
#             baseline_rmse: 0.9858002380723858
#             normalized_rmse: 0.27986236947338106
#             rmse: 0.27588839045436103
#             rows: 256
#             withheld: true
#           coral/mean:
#             baseline_rmse: 0.972345983430609
#             normalized_rmse: 0.2783762819714063
#             rmse: 0.27067805965724356
#             rows: 256
#             withheld: false
#           coral/min:
#             baseline_rmse: 0.9469336156041438
#             normalized_rmse: 0.703751265761235
#             rmse: 0.666405730573279
#             rows: 256
#             withheld: false
#         gap: 0.40385713286647146
#         normalized_rmse: 0.4708815976168349
#       corrupt_selected_group:
#         cells:
#           amber/max:
#             baseline_rmse: 0.8831115047625776
#             normalized_rmse: 1.4641682514470034
#             rmse: 1.2930238277609551
#             rows: 256
#             withheld: false
#           amber/mean:
#             baseline_rmse: 1.0017870715984531
#             normalized_rmse: 1.4199066583415738
#             rmse: 1.4224441332031506
#             rows: 256
#             withheld: true
#           amber/min:
#             baseline_rmse: 0.9404379907655642
#             normalized_rmse: 1.4443876932130402
#             rmse: 1.3583570600917796
#             rows: 256
#             withheld: false
#           blue/max:
#             baseline_rmse: 1.0229581561177996
#             normalized_rmse: 1.4087376190511571
#             rmse: 1.441079637238351
#             rows: 256
#             withheld: false
#           blue/mean:
#             baseline_rmse: 0.8995328184909653
#             normalized_rmse: 1.4517195445964062
#             rmse: 1.305869373609226
#             rows: 256
#             withheld: false
#           blue/min:
#             baseline_rmse: 0.9387242747363832
#             normalized_rmse: 1.458662944787219
#             rmse: 1.3692823149302191
#             rows: 256
#             withheld: true
#           coral/max:
#             baseline_rmse: 0.9858002380723858
#             normalized_rmse: 1.377705745709013
#             rmse: 1.3581426521136388
#             rows: 256
#             withheld: true
#           coral/mean:
#             baseline_rmse: 0.972345983430609
#             normalized_rmse: 1.3876274467440237
#             rmse: 1.3492539743396228
#             rows: 256
#             withheld: false
#           coral/min:
#             baseline_rmse: 0.9469336156041438
#             normalized_rmse: 1.410604921579083
#             rmse: 1.3357492185798807
#             rows: 256
#             withheld: false
#         gap: 1.3561654638636111
#         normalized_rmse: 1.4231899286139746
#       normalized_rmse: 0.06702446475036346
#       steps: 600
#     noise_rmse: 0.02
#     restricted:
#       cells:
#         amber/max:
#           baseline_rmse: 0.882264630289068
#           normalized_rmse: 0.04194823599316676
#           rmse: 0.037009444919789844
#           rows: 256
#           withheld: false
#         amber/mean:
#           baseline_rmse: 1.0020940218603624
#           normalized_rmse: 0.12115123014192883
#           rmse: 0.12140492346625582
#           rows: 256
#           withheld: true
#         amber/min:
#           baseline_rmse: 0.9397980374865507
#           normalized_rmse: 0.03474775061337767
#           rmse: 0.03265586783352442
#           rows: 256
#           withheld: false
#         blue/max:
#           baseline_rmse: 1.0257033728175442
#           normalized_rmse: 0.04361437811077591
#           rmse: 0.04473541473156253
#           rows: 256
#           withheld: false
#         blue/mean:
#           baseline_rmse: 0.8997349524439998
#           normalized_rmse: 0.04838625277527743
#           rmse: 0.04353480283970759
#           rows: 256
#           withheld: false
#         blue/min:
#           baseline_rmse: 0.9383243590679402
#           normalized_rmse: 0.09913623866146913
#           rmse: 0.09302194760242938
#           rows: 256
#           withheld: true
#         coral/max:
#           baseline_rmse: 0.9842903231303742
#           normalized_rmse: 0.2620744381917979
#           rmse: 0.25795733345201605
#           rows: 256
#           withheld: true
#         coral/mean:
#           baseline_rmse: 0.9677662655627186
#           normalized_rmse: 0.042490082917435876
#           rmse: 0.04112046886845718
#           rows: 256
#           withheld: false
#         coral/min:
#           baseline_rmse: 0.9466115169438977
#           normalized_rmse: 0.033320790431420826
#           rmse: 0.03154184397605698
#           rows: 256
#           withheld: false
#       corrupt_operation:
#         cells:
#           amber/max:
#             baseline_rmse: 0.882264630289068
#             normalized_rmse: 0.28817676838749584
#             rmse: 0.2542481700192924
#             rows: 256
#             withheld: false
#           amber/mean:
#             baseline_rmse: 1.0020940218603624
#             normalized_rmse: 0.35618379281641915
#             rmse: 0.35692964946488354
#             rows: 256
#             withheld: true
#           amber/min:
#             baseline_rmse: 0.9397980374865507
#             normalized_rmse: 0.6600755182235716
#             rmse: 0.6203376766194305
#             rows: 256
#             withheld: false
#           blue/max:
#             baseline_rmse: 1.0257033728175442
#             normalized_rmse: 0.33555463472543073
#             rmse: 0.34417952060243334
#             rows: 256
#             withheld: false
#           blue/mean:
#             baseline_rmse: 0.8997349524439998
#             normalized_rmse: 0.28777138184310275
#             rmse: 0.25891797055734816
#             rows: 256
#             withheld: false
#           blue/min:
#             baseline_rmse: 0.9383243590679402
#             normalized_rmse: 0.6685266180938474
#             rmse: 0.627294810442767
#             rows: 256
#             withheld: true
#           coral/max:
#             baseline_rmse: 0.9842903231303742
#             normalized_rmse: 0.3581884551035688
#             rmse: 0.35256143021546127
#             rows: 256
#             withheld: true
#           coral/mean:
#             baseline_rmse: 0.9677662655627186
#             normalized_rmse: 0.341478398411272
#             rmse: 0.33047127440081486
#             rows: 256
#             withheld: false
#           coral/min:
#             baseline_rmse: 0.9466115169438977
#             normalized_rmse: 0.4347428873235019
#             rmse: 0.4115326240498701
#             rows: 256
#             withheld: false
#         gap: 0.32595464662847873
#         normalized_rmse: 0.43574397793498826
#       corrupt_selected_group:
#         cells:
#           amber/max:
#             baseline_rmse: 0.882264630289068
#             normalized_rmse: 1.4673258126217945
#             rmse: 1.2945696655863739
#             rows: 256
#             withheld: false
#           amber/mean:
#             baseline_rmse: 1.0020940218603624
#             normalized_rmse: 1.4156009693991256
#             rmse: 1.4185652687745975
#             rows: 256
#             withheld: true
#           amber/min:
#             baseline_rmse: 0.9397980374865507
#             normalized_rmse: 1.431552900408219
#             rmse: 1.3453706063618238
#             rows: 256
#             withheld: false
#           blue/max:
#             baseline_rmse: 1.0257033728175442
#             normalized_rmse: 1.3693223261364482
#             rmse: 1.4045185283925201
#             rows: 256
#             withheld: false
#           blue/mean:
#             baseline_rmse: 0.8997349524439998
#             normalized_rmse: 1.4345344461773675
#             rmse: 1.2907007817106733
#             rows: 256
#             withheld: false
#           blue/min:
#             baseline_rmse: 0.9383243590679402
#             normalized_rmse: 1.4462461593033642
#             rmse: 1.3570480004827994
#             rows: 256
#             withheld: true
#           coral/max:
#             baseline_rmse: 0.9842903231303742
#             normalized_rmse: 1.3720766905865482
#             rmse: 1.350521809137088
#             rows: 256
#             withheld: true
#           coral/mean:
#             baseline_rmse: 0.9677662655627186
#             normalized_rmse: 1.3470430564001135
#             rmse: 1.3036228282445284
#             rows: 256
#             withheld: false
#           coral/min:
#             baseline_rmse: 0.9466115169438977
#             normalized_rmse: 1.3928611804667714
#             rmse: 1.3184984349339186
#             rows: 256
#             withheld: false
#         gap: 1.296718387596959
#         normalized_rmse: 1.4065077189034685
#       normalized_rmse: 0.10978933130650954
#       steps: 600
#     test_rows: 2304
#     train_rows: 4608
#   mode: full
#   outcome: met
#   provenance:
#     code_sha256: 1e1d421e0ee2492b916cb92b989876fd5f91dc3d76a2a1494c85f6ff0c413c32
#     cuda: null
#     evidence_revision: e00d336e3dc4a3c3352ab3fe5535ba69d6ac682f
#     gpu: null
#     host: 32cf3e650bff
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
#     script_sha256: 6e9224c92e02572a17c6fe8ac09293953ae7f9ffd4f9eb594d59204b3ddc5bfd
#     source_sha256: 3f3ac2c17b6ec9d18857604a204753e8d48e6dc225cf8d9e48c47b8bd80bbf65
#     wheel_sha256: d9796de8429bb57408422b49ea3f47765fc5a40548ecb4dacf165fc922bded09
#   seed: 7101
#   started_at: '2026-11-01T01:48:58.232454+00:00'
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
# selected_group: amber
# operation: mean
# items:
#   - {group: blue, value: -0.4}
#   - {group: amber, value: 0.2}
#   - {group: coral, value: 1.1}
#   - {group: amber, value: 0.8}
#   - {group: blue, value: 0.6}
#   - {group: coral, value: 1.5}
# answer: 0.51
# ```
#
# The answer is shown here for explanation; it is hidden from the model when scored.
#
# ```{typst}
# //| label: fig-evidence-held-out-requests
# //| fig-cap: "Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# //| fig-alt: "Model tree with request, selected_group, operation, items, group, value, answer. Amber cards are hidden prediction targets. Repeated collections keep their fields together."
# #tree(node("request", kind: "root", children: (
#   node("selected_group", type: "Enum"),
#   node("operation", type: "Enum"),
#   node("items", kind: "branch", repeated: true, children: (
#     node("group", type: "Enum"), node("value", type: "Number"),
#   )),
#   node("answer", kind: "target", type: "Number"),
# )))
# ```
#
# ## Comparison
#
# Test the missing combinations and compare with a model trained on all combinations. Change either request to check that it matters.
#
# ## Result
#
# {{< evidence P071 status >}}
#
# Only two of three seeds meet every check. Familiar requests learn, but one withheld combination can fail even when both instructions are used.
#
# <details class="evidence-details">
# <summary>Experiment details and code</summary>
#
# {{< evidence P071 evidence >}}
#
# ### Run this experiment
#
# {{< evidence P071 script >}}
#
# ### Synthetic process and split
#
# Each fresh bag contains two numbers per group. Each group's center is drawn
# independently from Uniform(-1.5, 1.5), with two independent Uniform(-0.8, 0.8)
# deviations. The visible request selects one group's mean, minimum, or maximum.
# Target noise is Normal(0, 0.02). Items are randomly interleaved in every row.
# The model never receives group centers or the generator's selected values.
#
# Training excludes amber/mean, blue/min, and coral/max. Every group and every
# operation still occurs in two observed combinations. Independently generated
# bags supply 4,608 training rows, 576 validation rows, and 2,304 test rows;
# test has 256 rows per cell. No bag is reused across splits or requests.
# Validation for the restricted model also excludes withheld combinations.
#
# ### Training, controls, and gates
#
# Both arms use the xs preset, retaining all item-branch tokens, batch size 128,
# and 600 AdamW updates at learning rate 0.002. Initialization is paired. The
# complete arm has the same row count and update budget, spread over all nine
# combinations. It is a task-learnability control, not a matched cell-frequency
# control. No architecture, checkpoint, or threshold is chosen on test results.
#
# The positive control must reach normalized RMSE below 0.35 in every cell.
# The restricted arm must reach 0.35 on every observed cell and 0.50 on every
# withheld cell. Each denominator uses a training-only operation mean pooled
# across exchangeable groups; no withheld target supplies a baseline estimate.
# Group corruption must give overall normalized RMSE above 0.80 and exceed
# intact error by 0.25. Operation corruption must add at least 0.10 to overall
# normalized RMSE. Both control predictions are compared with the original
# answers, so these controls measure reliance on the requests, not accuracy on
# the altered instructions. Report their cell scores as well as aggregate gaps.
#
# ### Complete experiment code
#

# %%
"""P071: test composition on group-operation pairs excluded from training."""

from collections.abc import Iterator
from functools import partial

import lightning.pytorch as lit
import numpy as np
import pyarrow as pa
import relflow as rf
from reporting import report

PROOF_ID = "P071"
BUDGET = 600
GROUPS = ("amber", "blue", "coral")
OPERATIONS = ("mean", "min", "max")
CELLS = tuple((group, operation) for group in GROUPS for operation in OPERATIONS)
WITHHELD = tuple(zip(GROUPS, OPERATIONS, strict=True))
TRAIN_ROWS = 4608
TEST_ROWS = 2304


def records(*, rows: int, seed: int, complete: bool = False) -> Iterator[dict]:
    rng = np.random.default_rng(seed)
    cells = CELLS if complete else tuple(cell for cell in CELLS if cell not in WITHHELD)
    requests = np.resize(np.arange(len(cells)), rows)
    rng.shuffle(requests)
    for index in requests:
        group, operation = cells[index]
        values = rng.uniform(-1.5, 1.5, (3, 1)) + rng.uniform(-0.8, 0.8, (3, 2))
        selected = values[GROUPS.index(group)]
        answer = {"mean": np.mean, "min": np.min, "max": np.max}[operation](selected)
        order = rng.permutation(6)
        yield {
            "selected_group": group,
            "operation": operation,
            "items": [{"group": GROUPS[i // 2], "value": float(values.flat[i])} for i in order],
            "answer": float(answer + rng.normal(0, 0.02)),
        }


def build() -> rf.Model:
    model = rf.Model.xs(
        selected_group=rf.Enum(values=GROUPS, p_unavailable=0.0),
        operation=rf.Enum(values=OPERATIONS, p_unavailable=0.0),
        items=rf.Branch(
            length=6,
            reduction=None,
            group=rf.Enum(values=GROUPS, p_unavailable=0.0),
            value=rf.Number,
        ),
        answer=rf.Number(mask=True, objective="mse"),
    )
    return model


def predict(model: rf.Model, rows: list[dict]) -> np.ndarray:
    visible = [{key: value for key, value in row.items() if key != "answer"} for row in rows]
    output = model.predict(pa.Table.from_pylist(visible))["predictions"].to_pylist()
    return np.asarray([row["answer"]["content"] for row in output])


def corrupt(rows: list[dict], field: str) -> list[dict]:
    """Cycle one request to a different declared value without changing its bag or target."""
    values = GROUPS if field == "selected_group" else OPERATIONS
    return [{**row, field: values[(values.index(row[field]) + 1) % len(values)]} for row in rows]


def scores(train: list[dict], test: list[dict], predicted: np.ndarray) -> dict:
    """Use the observed-operation training mean; groups are exchangeable by construction."""
    means = {
        operation: np.mean([row["answer"] for row in train if row["operation"] == operation])
        for operation in OPERATIONS
    }
    result = {}
    for group, operation in CELLS:
        indices = [
            index for index, row in enumerate(test) if (row["selected_group"], row["operation"]) == (group, operation)
        ]
        target = np.asarray([test[index]["answer"] for index in indices])
        rmse = float(np.sqrt(np.mean((predicted[indices] - target) ** 2)))
        baseline = float(np.sqrt(np.mean((means[operation] - target) ** 2)))
        result[f"{group}/{operation}"] = {
            "rows": len(indices),
            "withheld": (group, operation) in WITHHELD,
            "rmse": rmse,
            "baseline_rmse": baseline,
            "normalized_rmse": rmse / baseline,
        }
    return result


def run(seed: int, steps: int | None, accelerator: str) -> tuple[dict, dict]:
    test = list(records(rows=TEST_ROWS, seed=seed * 100 + 3, complete=True))
    metrics = {"train_rows": TRAIN_ROWS, "test_rows": TEST_ROWS, "noise_rmse": 0.02}
    checks = {}
    for complete in (False, True):
        lit.seed_everything(seed, workers=True)
        model = build()
        data = rf.SyntheticDataModule(
            train=partial(records, rows=TRAIN_ROWS, seed=seed * 100 + 1, complete=complete),
            validate=partial(records, rows=576, seed=seed * 100 + 2, complete=complete),
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
            deterministic=True,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=False,
            enable_model_summary=False,
            num_sanity_val_steps=0,
        )
        trainer = model.trainer
        model.eval()
        train = list(records(rows=TRAIN_ROWS, seed=seed * 100 + 1, complete=complete))
        output = predict(model, test)
        cells = scores(train, test, output)
        arm = "complete" if complete else "restricted"
        measured = {"steps": trainer.global_step, "cells": cells}
        for cell, score in cells.items():
            gate = 0.50 if score["withheld"] and not complete else 0.35
            checks[f"{arm}: {cell} normalized RMSE below {gate:.2f}"] = score["normalized_rmse"] < gate
        baseline = float(np.sqrt(np.mean([score["baseline_rmse"] ** 2 for score in cells.values()])))
        target = np.asarray([row["answer"] for row in test])
        intact = float(np.sqrt(np.mean((output - target) ** 2)) / baseline)
        measured["normalized_rmse"] = intact
        for field in ("selected_group", "operation"):
            control = predict(model, corrupt(test, field))
            error = float(np.sqrt(np.mean((control - target) ** 2)) / baseline)
            measured[f"corrupt_{field}"] = {
                "normalized_rmse": error,
                "gap": error - intact,
                "cells": scores(train, test, control),
            }
            checks[f"{arm}: corrupting {field} increases normalized RMSE"] = error - intact > (
                0.25 if field == "selected_group" else 0.10
            )
            if field == "selected_group":
                checks[f"{arm}: corrupted group normalized RMSE exceeds 0.80"] = error > 0.80
            checks[f"{arm}: {field} control predictions are finite"] = bool(np.isfinite(control).all())
        checks[f"{arm}: intact predictions are finite"] = bool(np.isfinite(output).all())
        metrics[arm] = measured
    return metrics, checks


if __name__ == "__main__":
    report(PROOF_ID, run, seed=7101)

# %% [markdown]
# </details>
