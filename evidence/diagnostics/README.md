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
compressed observation manifests. Generated files and ZIPs stay outside Git.
The trusted publication reader stages verified bundles under
`docs/assets/evidence-plots/`; this is a download location rather than an authored
website asset directory.

Ordinary CI keeps these reports and plots in Actions. It creates no
`evidence-ci-*` GitHub releases and does not trigger an Evidence-specific Read
the Docs rebuild. Download an attempt's `evidence-plots-attempt-<n>` artifact
from its Actions run to inspect candidate results before release.

Only the formal Release workflow publishes frozen Evidence. It authenticates
the exact successful main CI attempt and helper/executor pins, then retains
plots, reports, inputs, provenance, and checksums in `evidence-v<version>`.
The title is **relflow <version> evidence**. New public downloads use
`relflow-<version>-evidence*` names, including
`relflow-0.5.0-evidence-plots.zip` and `relflow-0.5.0-evidence.json`.
Already published assets retain their original names and bytes.

Read the Docs branch and PR builds show the latest formal release's results;
tagged builds require the corresponding version's exact frozen selection.
Static builds verify asset digests, bundle inventories, frozen proof sources,
and exact record equality. They never install or import relflow, Torch,
Lightning, Matplotlib, or Arrow. Plot sources are checked against their own
recorded execution, so an updated PR proof cannot change the meaning of a
previous release's measurements.

No Read the Docs token is needed for Evidence publication. Versioned hosting
continues through the repository's normal release/tag integration; CI does not
request per-commit preview rebuilds.

Review populations, units, matched scales, control labels and both light/dark
layouts when changing visual briefs or renderers. Keep projections fitted to
training/reference data and state their scope. A figure renders only beside its
identical candidate record, with exact downloads and the original CI link.
