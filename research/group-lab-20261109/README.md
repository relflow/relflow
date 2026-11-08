# Group optimization search on the lab

This is an exploratory validation search, not a canonical Evidence run. The
candidate wheel was built from merge commit `e5d6f42` (the code is Reid's PR
head merged with main). The wheel, scripts, P078 source, and reporting inputs
are identified by `input/SHA256SUMS`; `input/image-id.txt` records the Docker
image. The lab CPU is recorded in `input/cpu.txt`. All 18 trials completed
successfully; `output/` retains each JSON measurement, log, and exit status.
The separate two-step smoke result is not a learning measurement.

Each trial uses P078's stable-regime generator, 4,096 training and 1,024
independent validation observations of the same 64 identities, 600 AdamW
updates, batch size 128, EMA 0.99, and no unavailable-input corruption. It
compares 4 and 8 group slots and learning rates 0.001, 0.003, and 0.01 on
seeds 7801, 7802, and 7803. Two network-disabled CPU containers ran concurrently,
each limited to 2 CPUs and 4 GiB. The script reports assignment ARI and RMSE
normalized by the training-mean baseline on validation observations. This
search uses no P078 test observations and has no no-regime control; it cannot
establish held-out test performance or generalization to new identities.

| Groups | Learning rate | Mean validation ARI | Mean validation normalized RMSE |
| ---: | ---: | ---: | ---: |
| 4 | 0.001 | 0.089 | 0.834 |
| 4 | 0.003 | 0.232 | 0.605 |
| 4 | 0.010 | 0.504 | 0.479 |
| 8 | 0.001 | 0.079 | 0.778 |
| 8 | 0.003 | 0.224 | 0.525 |
| 8 | 0.010 | 0.587 | 0.261 |

These means summarize the selected validation grid, not a prespecified test.
For eight groups at 0.01, ARI was 0.657, 0.447, and 0.659; normalized RMSE was
0.193, 0.362, and 0.228. More groups make perfect four-way partition agreement
harder if one true group is split across slots. Neither the chosen capacity nor
learning rate establishes a general improvement. Next, evaluate selected
configurations on untouched test data with matched no-regime and shuffled-ID
controls, independent seeds, and a parameter-matched Category comparison.

The runner and setup are in [group_lab.py](../../scripts/group_lab.py) and
[group_lab.sh](../../scripts/group_lab.sh). On the lab the original run is
`~/relflow-research/runs/group-lab-20261109-v1`. To replay, create a fresh
directory with `input/package` populated by the recorded wheel, place those
scripts and `evidence/experiments/{reporting.py,source.py,group/regime_recovery.py}`
under `input/`, create `output/`, and run `bash input/scripts/group_lab.sh`
from that directory with the recorded image. Existing trial output is never
overwritten.
