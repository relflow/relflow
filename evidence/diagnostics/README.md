# Proof diagnostics

Every full paired Evidence CI run captures the observations needed for its plots
while the original baseline and candidate proofs execute. No second learning or
inference run is added. Seeds, budgets, checks, optimizers and canonical result
records remain unchanged. Return-event monitoring retains already-computed
readouts without random draws; its overhead makes these runs unsuitable as
performance benchmarks. Spawned trials retain their returned metrics but do not
inherit monitoring.

Each proof has an authored visual brief in `views.json`: the question to make
visible, comparison and interpretation limits. The plots use the candidate
role of the paired run. Optional research pages retain generator illustrations
until their declared resources and learning records are available. Unmet checks
remain visible; figures never turn an illustration into a measured outcome.

P099 and P100 accept measured views when a matching Duration/Ordinal wheel has
been explicitly provisioned. P099 compares elapsed-time predictions, error
distributions and a shuffled-training control. P100 separates familiar and
withheld ranks with per-rank accuracy heatmaps. The same captured executions
supply their example viewers. Without those records, keep the generator
illustrations. These studies remain in research while the paired baseline
predates the datatypes.

Twelve CPU shards execute both exact wheels and retain separate, checksum-bound
observation archives. The collector validates the complete paired catalog. A
following job verifies every source, wheel, task, record and observation receipt,
then renders static SVG/JSON inside a bounded offline container. Selected plotting
code never executes on the host. A rendering failure fails Evidence; learning
regressions can still produce plots showing the failed comparison. Execution or
provenance errors prevent publication. Never retry an unmet check to get a
passing result.

The plots job uploads `evidence-plots-attempt-<n>` for 90 days. Raw per-task Arrow
observations are separate `evidence-observations-shard-<i>-attempt-<n>` artifacts
retained for 14 days, outside the report size limit. The public bundle includes
reviewed light/dark SVGs, example data, exact records and proof sources, and
compressed observation manifests. Only authored CSS/JavaScript are tracked in
`docs/assets/evidence-plots/`; generated files and ZIPs stay outside Git.

After CI completes, the privileged publication workflow runs a frozen verifier,
authenticates the original internal CI attempt and helper/executor pins, and
retains the plots, publication selection, reports, wheels and checksums in an
immutable `evidence-ci-<run-id>-attempt-<n>` GitHub Release. It creates no source
commits or publication PRs. Forks cannot publish. Each attempt has its own assets;
reruns never overwrite earlier measurements. Once this workflow is on `main`,
its `workflow_run` trigger is automatic. Before merge it can be exercised with:

```sh
gh workflow run publish-evidence.yml --ref <branch> \
  -f preview=true -f library_run_id=<complete-ci-run-id>
```

Set the repository's `READTHEDOCS_TOKEN` Actions secret to a Read the Docs API
token with build access for the `relflow` project. The publisher requests a build
of the current PR version or `latest` after publishing the assets. If absent,
asset publication finishes and the workflow reports the missing configuration.
It skips rebuilds for superseded heads. Tokens never enter proof containers,
plot containers, bundles or documentation builds.

Read the Docs branch and PR builds select the newest CI run and attempt for the
exact checkout commit, including unsuccessful attempts. Pending or unavailable
plots show an explicit status with no fallback to an older passing result.
Package release builds freeze the matching plot bundle in `evidence-v<version>`
alongside their publication JSON; old releases remain unchanged. Static builds
verify GitHub asset digests, bundle inventories and exact record equality. They
never install or import relflow, Torch, Lightning, Matplotlib or Arrow.

For local review of an exact commit after its assets are published:

```sh
uv run --project docs python -c 'import subprocess; from pathlib import Path; from evidence.assets import ci; print(ci(Path.cwd(), subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()))'
RELFLOW_EVIDENCE_PUBLICATION="$PWD/.cache/evidence/ci-preview.json" make dev
```

Review populations, units, matched scales, control labels and both light/dark
layouts when changing visual briefs or renderers. Keep projections fitted to
training/reference data and state their scope. A figure renders only beside its
identical candidate record, with exact downloads and the original CI link.
