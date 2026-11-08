# Group Evidence runtime and EMA exploration

This is a local, unpaired lab investigation, not a replacement for P078 or
paired Evidence CI. It uses the exact candidate wheel and P078 source identified
in `input/SHA256SUMS`, the recorded container image in `input/image-id.txt`, and
the CPU in `input/cpu.txt`. Raw JSON, logs, status files, and smoke runs are in
`output/`. Every planned trial exited successfully. The first timing probe
failed before producing a measurement because its name shadowed Python's
`profile` module; `profile-attempt1.log` preserves that failure. Its identical
source was renamed to `timing_probe.py` before the recorded measurement.

## What was tested

P078 keeps four Group slots, 64 known identities, 4,096 training observations,
1,024 validation observations, and 2,048 independent test observations per arm.
The stable, no-regime, and shuffled-test-ID controls and all P078 named checks
are retained where full checks are reported. Discovery uses only the stable
arm's validation data and seeds 7801–7803. Confirmation uses fresh seeds
8801–8803 and runs both arms and all original checks. The 1,800-update,
learning-rate 0.006 setting was chosen from discovery; the 1,800-update original
rate and EMA 0.9 were added before inspecting the confirmation outcomes.

At 1,200 updates, discovery tried learning rates 0.003, 0.006, 0.01, and 0.02;
none recovered all three partitions. At 1,800 updates, 0.006 produced ARI 1.0
on all three discovery seeds, whereas 0.01 missed one. This was a selection
on validation, not evidence of held-out reliability.

## Independent-seed confirmation

Each row runs all P078 checks on each of the three new seeds. Times include both
arms and evaluation, are measured on the lab with concurrent jobs, and are not
hosted CI timing estimates. The 4,800-update setting itself is exploratory;
P078's committed budget is 600 updates.

| Configuration | Mean elapsed | Seeds passing every check | Stable-arm test ARI by seed (8801, 8802, 8803) | Stable-arm test normalized RMSE |
| --- | ---: | ---: | --- | --- |
| 4,800 steps, LR 0.003, EMA 0.99 | 318.3 s | 2 / 3 | 1.000, 1.000, 0.542 | 0.028, 0.029, 0.384 |
| 1,800 steps, LR 0.003, EMA 0.99 | 119.6 s | 2 / 3 | 0.958, 0.958, 0.597 | 0.115, 0.125, 0.353 |
| 1,800 steps, LR 0.006, EMA 0.99 | 126.8 s | 1 / 3 | 0.882, 0.918, 1.000 | 0.203, 0.157, 0.025 |
| 1,800 steps, LR 0.003, EMA 0.9 | 111.0 s | 2 / 3 | 0.958, 0.958, 0.769 | 0.115, 0.140, 0.254 |

Both shorter original-rate configurations fail ARI on seed 8803. LR 0.006 fails
ARI on 8801 and 8802, despite its discovery results. Even the longer baseline
misses ARI and prediction error on 8803. These checks require ARI ≥ 0.95 and
normalized RMSE ≤ 0.35; the controls and their exact results remain in each
record. No tested setting supports a reliable shorter budget yet.

The EMA confirmation adapter changes only the constructed EMA decay, while
the original P078 function still writes its unchanged `EMA_DECAY` constant
inside each arm's `metrics.*.ema_decay`. Consequently that nested field reads
0.99 even in the EMA 0.9 trials. The actual decay is 0.9, as recorded in the
top-level `ema_decay`, and the adapter source and checksum document the change.
The original records have not been edited to hide this metadata mismatch.

For seed 8803 in the 4,800-step original-rate run, the best recorded validation
RMSE was 0.517 at step 2,112, where assignment ARI was 0.958. At the end of
training, test ARI was 0.542 and normalized RMSE was 0.384. This makes
validation-based checkpoint selection worth investigating. The step-2,112
checkpoint and its test predictions were not saved, so these observations do
not establish that checkpoint's held-out result or a safe stopping policy.

## EMA discovery

At LR 0.003 and 1,800 updates, the same three discovery seeds were evaluated
on validation observations, with no EMA or one of three decay values:

| EMA decay | Validation ARI (7801, 7802, 7803) | Validation normalized RMSE |
| --- | --- | --- |
| none | 1.000, 1.000, 1.000 | 0.053, 0.037, 0.028 |
| 0.9 | 1.000, 1.000, 1.000 | 0.039, 0.025, 0.025 |
| 0.99 | 1.000, 1.000, 0.958 | 0.029, 0.023, 0.119 |
| 0.999 | 0.446, 0.508, 0.362 | 0.434, 0.465, 0.510 |

EMA 0.9 improved validation error over the live weights on these three seeds;
0.999 lagged the training at this budget. EMA 0.9 improved the difficult
confirmation seed over EMA 0.99 but still failed its ARI criterion, so this is
not a reliable convergence fix. relflow updates live parameters with AdamW;
EMA supplies averaged parameters at validation and final evaluation.

## Runtime-only options

- Wrapping the unchanged 600-update proof's trajectory callback measured
  36 calls totaling 1.67 seconds out of 36.64 seconds (4.6%) on one seed. The
  instrumented run reproduced its previous ARI and normalized RMSE exactly.
  Removing all trajectory work would have limited upside and would reduce plot
  detail; this profile does not measure hosted CI.
- Materializing the same fixed generator output once as Arrow gave a paired
  mean of 38.30 seconds versus 38.76 seconds for the original synthetic
  source, around 1.2% faster over three seeds. All returned metrics and checks
  matched exactly. On seed 7801, all 39 existing prediction calls also matched
  exactly by input and output hash in a separate paired verification. The
  extra path is unlikely to be worthwhile for this workload.
- Two Torch/OMP threads averaged 36.67 seconds versus 38.70 with one, around
  5.2% faster over three concurrent pairs. The outputs did not match exactly:
  one seed's final ARI changed from 0.189 to 0.204, and prediction errors also
  changed. This is a different numerical execution policy; it cannot silently
  replace the pinned one-thread Evidence runtime.

Trials ran in network-disabled containers on the lab, with a 2-CPU and 4-GiB
limit per task. Concurrent workloads differed between the validation, EMA,
confirmation, ingress, and thread batches. Compare timings within a batch,
not across batches as a controlled speed benchmark. These isolated local
timings do not predict overall CI completion time, which depends on sharding,
the other proofs, and hosted hardware.

## Reproduce

The original lab directory is `~/relflow-research/runs/group-evidence-runtime-20261109-v1`.
The shell scripts run from a fresh directory with `input/package` extracted
from the recorded wheel; `input/scripts/group_lab.py`, P078 and its reporting
sources, and this directory's scripts must retain their recorded paths under
`input/`. Create `output/` before running the desired script. Each output name
is exclusive and existing results are never overwritten. The Python smoke modes
perform two updates only and cannot establish learning behavior.

No Evidence budget, check, scientific result, or production implementation
was changed. A credible faster proof needs a more reliable optimization or
stopping strategy, followed by additional independent seeds and the unchanged
paired CI protocol; selecting another setting on these results would require a
new confirmation panel.
