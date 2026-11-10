# Group evidence studies, 2026-11-09

Six standalone optional research-lane experiments examine five questions about the unreleased Group datatype. All results here are unpaired local lab observations, not required CI results or published evidence. Each full run uses an exact-source capture, three predeclared seeds, independent held-out observations, 2,400 updates per training stage, batch size 128, AdamW learning rate 0.003, 32-bit CPU execution, and EMA decay 0.99. Except for the unseen-label study, evaluation uses identities that appeared in training. Scores below are normalized RMSE: 1 means matching the training-mean predictor and lower is better. Denominators are calculated separately for each held-out population.

Seed-level plots: [rare identities](plots/P102.png), [slot counts](plots/P103.png), [unseen identities](plots/P104.png), [assignment intervention](plots/P105.png), [drift](plots/P106.png), and [fresh-seed drift confirmation](plots/P107.png). The canonical Evidence renderer also produced record-bound single-seed diagnostics for all six studies, including the unmet confirmation.

## Rare versus frequent identities — P102

A fixed training set has 120 distinct observations for each of 32 frequent identities and 8 for each of 32 rare identities; those observations repeat across epochs. The equally weighted held-out population has new observations of those same identities. Group, Category, no identity, and a shuffled-identity Group use the same model preset and update budget. They are not exactly parameter matched: Group has 411,282 trainable parameters, Category 414,862, and the no-identity model 410,446.

| Seed | Group rare | Category rare | No identity rare | Shuffled rare | Group frequent | Category frequent |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 10201 | .257 | .123 | 1.032 | 1.069 | .024 | .021 |
| 10202 | .210 | .027 | 1.011 | 1.044 | .026 | .025 |
| 10203 | .368 | .028 | 1.009 | 1.034 | .026 | .023 |

Group used the identity signal, but Category performed better on rare identities in all three seeds. This does not prove Category always wins at low exposure, and neither model was tuned across architectures or budgets.

## Slot counts — P103

| Seed | Two slots nRMSE / ARI | Four slots nRMSE / ARI | Eight slots nRMSE / ARI | Eight slots occupied |
| --- | ---: | ---: | ---: | ---: |
| 10301 | .562 / .361 | .023 / 1.000 | .023 / .809 | 7 |
| 10302 | .470 / .488 | .026 / 1.000 | .024 / .742 | 8 |
| 10303 | .470 / .488 | .024 / 1.000 | .024 / .737 | 8 |

Two slots cannot represent all four regimes and lose predictive accuracy. Eight slots predict well in this setting while splitting regimes, so a lower ARI does not necessarily mean worse predictions. The four-slot shuffled-identity controls had nRMSE 1.038–1.057.

## Unseen identities — P104

| Seed | Dropout | Known nRMSE | Unseen nRMSE | Null nRMSE |
| --- | ---: | ---: | ---: | ---: |
| 10401 | 0 | .310 | .985 | 1.251 |
| 10401 | .10 | .109 | 1.113 | 1.218 |
| 10402 | 0 | .022 | 1.617 | 1.571 |
| 10402 | .10 | .152 | .990 | 1.061 |
| 10403 | 0 | .024 | 1.760 | 1.387 |
| 10403 | .10 | .201 | 1.068 | 1.092 |

Unknown names produced the same prediction at the same numeric input and evaluation did not change the vocabulary. Unavailable-content dropout improved unseen error in two of three seeds but increased known-identity error in those seeds. This is a fallback test, not recovery of new identity-specific rules.

## Assignment intervention — P105

For seeds 10501–10503, temporarily rotating assignments increased normalized test error from .111/.023/.023 to 1.878/1.551/1.893. Every intervention prediction exactly matched a representative of its destination slot at the same numeric setting, and the original predictions and assignment state were restored exactly after the override. Group numbers have no intrinsic order.

## Behavior drift — P106

A quarter of the identities change rule between two training stages; a matched control continues on stable behavior. All original checks passed, but they only require partial improvement over the pre-drift and stable-stream error. They are too weak to establish successful adaptation.

| Seed | Changed identities before | Changed after drift | Changed after stable control | Unchanged before | Unchanged after drift | Changed identities moved | Unchanged identities moved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10601 | 1.584 | 1.259 | 1.589 | .024 | .424 | 0 / 16 | 0 / 48 |
| 10602 | 1.517 | 1.213 | 1.521 | .024 | .366 | 0 / 16 | 0 / 48 |
| 10603 | 1.573 | 1.245 | 1.576 | .025 | .413 | 0 / 16 | 0 / 48 |

The model reduced error on changed identities somewhat without changing any hard assignment, while degrading predictions for unchanged identities. Both stages start a new optimizer and EMA, as documented in the experiment. Continued training on stable behavior kept unchanged-identity error at .023–.024. This is a specific synthetic drift pattern, not a universal adaptation result.

## Fresh-seed drift confirmation — P107

After observing P106, we defined a separate confirmation before running seeds 10701–10703. Its additional criteria require at least 8 of the 16 changed identities to move groups and unchanged-identity normalized RMSE to remain at most .20. **Both criteria were unmet in all three seeds.**

| Seed | Changed identities before | Changed after drift | Changed after stable control | Unchanged before | Unchanged after drift | Changed identities moved | Unchanged identities moved |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 10701 | 1.415 | 1.062 | 1.416 | .023 | .425 | 0 / 16 | 0 / 48 |
| 10702 | 1.738 | 1.282 | 1.740 | .022 | .443 | 0 / 16 | 0 / 48 |
| 10703 | 1.594 | 1.141 | 1.600 | .025 | .488 | 0 / 16 | 0 / 48 |

The confirmation's three other checks passed, including partial improvement on changed identities against the stable-stream control. They do not offset the two unmet adaptation criteria. Stable-stream unchanged errors were .022–.024. These results test known identities and one abrupt rule change with the declared fixed budget.

## Provenance and reproduction

The exact candidate wheel SHA-256 is `cabe120b55c0e552e64469ecf337171650c3fd087375e7d56dd629ddccde5e8b`; the Docker image ID is `sha256:4782e1e9f2813256d24f81870d40c292723863ee2620342698c2e1ab7ed036a5`. Containers had no network, a two-CPU ceiling, 4 GiB memory, one Torch thread, and ran concurrently on `lab.tail815717.ts.net`. Timings therefore are not isolated benchmarks. The record intentionally says `uncommitted-working-tree` for the evidence revision. The exact proof source, code, wheel, observation hashes, record, and logs travel with each capture.

`attempts/initial/` preserves the first source version; `attempts/final/` adds an explicit synthetic iterator; `attempts/revised/` adds the destination-slot and drift-population readouts; `attempts/rare/` clarifies that the fixed rare observations repeat. The selected exact-source records for P102, P103–P104, and P105–P106 come respectively from rare, final, and revised attempts. All attempts and their full panels are retained; no run was selected because its checks passed. `output/` contains artifacts unpacked with checksum validation from those exact captures. The independent P107 confirmation is kept in its own attempt.

The shell harnesses refuse to overwrite a prior result and expect the preserved lab `input/` tree and image. From a new run directory, `bash input/run.sh smoke` uses only 32 updates and is an execution check, not a learning result; `bash input/run.sh full` uses the declared budgets and seed panel. `plots.py` renders the seed-level comparison from the captured records. See each experiment source for its full generator, hypotheses, controls, and criteria.

The captured proof sources predate the move to Polars prediction outputs. Current experiment scripts adapt that output interface; their source hashes therefore differ from these original captures, whose records and bundled sources remain unchanged.
