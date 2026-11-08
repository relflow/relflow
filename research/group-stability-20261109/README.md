# Group stability exploration, 2026-11-09

These are local research runs on `lab.tail815717.ts.net`, not paired CI evidence. They use the candidate `relflow` wheel with SHA-256 `cabe120b55c0e552e64469ecf337171650c3fd087375e7d56dd629ddccde5e8b`, Python 3.12.16 and CPU Torch 2.12.0. The containers ran without network access, with a two-CPU and 4 GiB limit and one Torch thread. Runs in a panel were concurrent, so their elapsed times are comparative observations, not isolated benchmarks. The image identity and input checksums are in `input/`. Raw per-run JSON, logs, exit statuses, and the captured P101 observations are in `output/`.

## Validation-selected checkpoints

The P078 Group proof already uses EMA with decay 0.99 and learning rate 0.003. We compared its fixed 4,800 updates per arm with a policy using the existing relflow early stopping and rollback callbacks: the same 4,800-update ceiling, validation-loss patience of 20 epochs, and the selected checkpoint restored before evaluating. Both the stable and no-regime arms use the policy. Thus the control may train for fewer steps, and this is a different experimental policy, not a shorter execution of an identical fixed-budget proof. The existing proof and its checks were not changed.

| Seed | Fixed seconds | Selected seconds | Fixed test ARI | Selected test ARI | Fixed test nRMSE | Selected test nRMSE | Selected steps, stable / control |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 9901 | 341.5 | 127.9 | .958 | 1.000 | .125 | .024 | 2304 / 160 |
| 9902 | 333.8 | 182.4 | .958 | 1.000 | .331 | .023 | 3552 / 192 |
| 9903 | 339.0 | 124.4 | .958 | .958 | .114 | .119 | 2400 / 160 |
| 10901 | 294.1 | 133.0 | 1.000 | 1.000 | .027 | .024 | 2304 / 224 |
| 10902 | 278.5 | 124.9 | 1.000 | 1.000 | .023 | .023 | 2240 / 224 |
| 10903 | 293.9 | 145.2 | 1.000 | 1.000 | .023 | .023 | 2592 / 320 |

All original P078 gates passed in each run. The validation policy was fixed after the first three seeds and before the final three. Across both panels, mean elapsed time was 313.5 seconds fixed and 139.6 seconds selected, about 55% less. This is promising but too narrow to establish a general convergence or CI-runtime guarantee. Recorded `stopped_at` values include the patience interval and exceed the selected step. Each restored model tensor was checked against its checkpoint; all twelve selected checkpoint files were subsequently verified against the recorded SHA-256. They remain on the lab under `~/relflow-research/runs/group-stability-20261109-v1/output/checkpoints/` rather than being copied here.

## Straight-through gradient and soft-routing explorations

The temperature prototype keeps the hard train/evaluation routes and changes only the softmax used by the straight-through gradient. These are validation-only search results; they are not independently held-out proof outcomes. At 1,800 updates, temperatures 0.5, 1, and 2 had mixed results across six seeds, and no temperature reached ARI above .95 consistently. At 2,400 updates, the comparison was:

| Seed | Temperature 0.5 ARI / validation nRMSE | Temperature 1 ARI / validation nRMSE |
| --- | ---: | ---: |
| 7801 | .958 / .118 | 1.000 / .059 |
| 7802 | 1.000 / .024 | 1.000 / .024 |
| 7803 | 1.000 / .023 | 1.000 / .024 |
| 8801 | 1.000 / .024 | .958 / .123 |
| 8802 | .918 / .152 | 1.000 / .025 |
| 8803 | 1.000 / .025 | .769 / .248 |

Temperature 0.5 improves some seeds and worsens others. Soft mixtures during training, with hard evaluation, also performed poorly on this search panel (ARI .121–.348 at 1,800 updates). Neither prototype has been promoted to the Group implementation.

## Broader regimes

With the original 4,800-update policy and seeds 9901–9903, P078's downstream prediction and control checks all passed for two additional synthetic cases: four crossing rules with balanced identity counts, and original-style separated rules with identity counts 24/8/8/24. Each crossed run had ARI 1.000 and normalized RMSE .078–.083; each imbalanced run had ARI 1.000 and normalized RMSE .020–.022. These are exploratory runs on known identities, not tests of unseen identities or of arbitrary imbalance and rule geometry.

A separate, standalone research-lane proof, P101, was then defined with balanced crossing rules and fresh seed 10101. Its full-budget capture met all 12 checks: stable test ARI 1.000 and normalized RMSE .083, no-regime ARI -.020 and normalized RMSE 1.087, and shuffled-identity normalized RMSE 1.420. It uses the same 64 identities across splits with fresh observations, so it measures recovery for known identities. The result is a single unpaired research run, not published or required CI evidence. The capture's evidence revision is explicitly `uncommitted-working-tree`; its source, code, wheel, and observation hashes are included in `output/p101-full.zip`.

We also applied the same validation-selection policy to P101 on seed 10101. It stopped the stable arm at step 2432 and restored step 1792; the control stopped at 800 and restored step 160. It passed all 12 gates, but stable test ARI was .958 and normalized RMSE was .149, worse than the fixed-budget result's 1.000 and .083. The captured trajectory shows that after an early peak the partition degraded and then recovered with more training. This single case illustrates a real quality tradeoff: passing the gates earlier does not necessarily recover the best final model. The fixed capture and selected run have different instrumentation, so their timings should not be compared. The two selected checkpoint files remain on the lab and were verified against their recorded SHA-256.

- [Learning trajectories](plots/P101.png)
- [Composition of learned groups](plots/P101-groups.png)
- [Predictions and controls](plots/P101-predictions.png)
- [Captured result](plots/P101-record.json)

## Reproducing and provenance

The scripts in this directory are the recorded lab harnesses. They expect the preserved `input/` tree, wheel-backed Docker image, and a fresh `output/` directory inside a copy of the lab run directory. The scripts intentionally refuse to overwrite existing receipts. For example, from a fresh lab run directory with those inputs, run `bash input/research/group-stability-20261109/checkpoint.sh`, `bash input/research/group-stability-20261109/checkpoint-confirm.sh`, or `bash input/research/group-stability-20261109/stress.sh`. `checkpoint-crossing.sh`, `p101.sh`, `temperature.sh`, `temperature_extended.sh`, and `soft.sh` capture the other runs. The image digest, exact input hashes, parameters, commands, and exit statuses are retained alongside the results.

The initial two-step temperature smoke used a seed that did not expose all identities and failed before producing measurements; a different seed completed the smoke. The first P101 smoke lacked an explicit revision in a container without Git and also failed before measurement; a second smoke with an honest working-tree revision succeeded. Both logs are preserved.
