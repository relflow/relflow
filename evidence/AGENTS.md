# Canonical relflow evidence

Preserve stable P identifiers, generators, seeds, budgets, optimizers, capacities,
checks, and current result provenance. Read SPEC.md before changing semantics. Never
rewrite measurements to make a gate pass. The root AGENTS.md also applies.

Read prediction output with Polars DataFrames and Series. Use
`pl.DataFrame(model.predict(...))` to normalize the older Arrow-returning
baseline into Polars while retaining the candidate's native frame; do not
coerce predictions back to Arrow for metric checks. IPC diagnostic artifacts
remain a separate persistence boundary.

Keep fixed field and branch contracts visible in experiment-owned `Model` and
`Branch` classes. Put declarations at module scope when they do not depend on a
local experimental arm; instantiate the neural model only during execution.
Generated field mappings remain appropriate for parameterized schemas.

`experiments/policies.py` is the narrow shared-configuration exception: its
immutable `CONTROL` policy freezes the historical XS architecture under the
internal name `evidence-control-v1`. It contains configuration and registration,
not model factories, field definitions, generators, data, or training helpers.
Every experiment retains its own root/branch overrides, masks, fit options,
budgets, and checks. The policy has no fit recipe; never replace these protocols
with a fit preset. Do not track changing SDK architecture defaults.
The internal name replaces the previous `xs` policy metadata in new executions;
historical records retain their original names, sources, and measurements.
Runtime input identities and isolated worker views include this shared module.
The reporter hashes all top-level shared runtime Python files without importing
them. Static discovery and documentation builds must not import `policies.py`.

From the repository root, `make check` validates static docs and evidence tooling.
Runtime tests under evidence/tests/runtime are explicit and require a separately
supplied RelFlow wheel; use `uv run --project docs --with /path/to/matching-relflow.whl`
to keep the intended wheel separate from the root editable SDK. Runtime tests
are excluded from default checks. Builds must never
import RelFlow, Torch, or Lightning. Execution writes independent immutable JSON
records. Each `@experiment("P001", seed=123)` declaration owns the stable ID and
default seed; preserve the original callable and inspect declarations statically.
Experiment frontmatter owns presentation and prose; measured results live
in publication JSON assets. Never write results back into experiment sources.

The required core catalog lives in `experiments/<category>/`. Optional studies
live in `research/<category>/`, declare `evidence-requires`, and run only with an
explicit `--lane research` selection. IDs are global across both directories.
Hosted execution keeps every core experiment; frontmatter cannot change a lane.
Static docs and authoring checks cover both lanes without importing resources.

Starlight reads the declarations, authored comment cells, and selected publication
through `docs/starlight/src/evidence/data.ts`. Keep visible behavioral criteria,
structured nested metrics, the full raw record, reproduction details, provenance,
and exact source downloads. Source commentary must remain distinct from measured
checks. The site never generates a second Python page or executes diagram source.
Preserve retained diagnostic plots alongside their readouts and tables. Retain actual
paired predictions, scores, masks, coordinates, and other needed observations
during the proof; do not limit figures to the latest released scalar metrics.
Bind every measured figure to its exact source, full-budget record, and artifact
checksums. Keep missing values explicit and generator illustrations distinct
from model outcomes. Static builds consume prepared SVG/JSON only; paired
Evidence CI runs own model compute and capture already-computed readouts. See [diagnostics/README.md](diagnostics/README.md).
Presentation changes never add measurements or alter scientific budgets and checks.

Use `experiments/run.py --new <category>/<name>` for a new authoring file. The
template lives in `templates/experiment.py.txt`, outside canonical discovery.
Creation must remain static, reject unsafe paths and overwrites, allocate the
next ID without reusing gaps, and preserve explicit ID/seed overrides. Keep the
generated scientific body visibly incomplete with `NotImplementedError` until
the author supplies independent splits, a model, budgets, matched controls,
metrics, and checks. Never generate a placeholder passing result or add a real
experiment merely to test scaffolding. Preserve existing experiment semantics.

GitHub-hosted CPU jobs use exact supplied baseline and candidate wheels, with
one bounded, network-disabled, read-only Docker container per task. Twelve shards
keep each experiment's baseline and candidate on the same runner; a separate
collector verifies the entire original catalog. Mount neither the host home nor
Docker socket into workers. Keep credentials outside worker containers.
Shared hosted Docker operations belong in `runtime.py`; `worker.py` owns the
container entrypoint. Historical readers verify archived reports without
restoring obsolete execution backends.
Smoke, partial coverage, invalid provenance, execution failures, and regressions
of baseline-passing checks cannot satisfy the required gate. Reports use Actions
artifacts and immutable Releases in relflow/relflow, with no AWS dependency.
Freeze canonical versioned candidate measurements only with a package version release, from its
complete passing paired CI run in relflow/relflow. Freeze the publication JSON,
reports, and actual execution provenance in an immutable `evidence-v<version>`
Release before publishing the package tag. Ordinary CI retains per-attempt
reports and plots only as GitHub Actions artifacts. It creates no public
Evidence releases, source commits, branches, or pull requests. Hosted branch/PR
docs show the latest formal release's frozen results. New release downloads use
`relflow-<version>-evidence*` names; existing published names remain immutable. Keep the recorded bootstrap snapshot in
`results.json` for offline builds; it is historical data, not an automatically
updated selection. Hosted builds verify the public release asset's digest and
source revision, without model dependencies or credentials.
Equivalent requests may reuse that execution only through the trusted verifier
and a separate receipt binding the current source, wheel payloads, comparison,
and actual execution runtime. Hosted shards execute both roles freshly;
dependency caches do not authorize measurement reuse. Historical baseline
references retain their original immutable records and execution provenance.
The required gate verifies every logical slot and its actual shard host, image,
and resource policy. A missing or failed shard cannot become a successful gate.
A publication update must not relabel an older result as a fresh run.

Retry only bounded infrastructure failures before a worker has produced a
measurement. Preserve launch attempts and logs. Never retry a failed learning
check, worker exception, OOM, cancellation, or learning timeout until one happens
to pass. See [hosted.md](hosted.md) for execution and workflow trust.
