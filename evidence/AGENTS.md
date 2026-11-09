# Canonical relflow evidence

Preserve stable P identifiers, generators, seeds, budgets, optimizers, capacities,
checks, and current result provenance. Read SPEC.md before changing semantics. Never
rewrite measurements to make a gate pass. The root AGENTS.md also applies.

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

The renderer adds visible behavioral criteria before the Example in generated
pages, keeping the canonical scripts' authored prose and scientific code intact.
Use structured groups and tables for nested metrics, with the full raw record
in its own disclosure. Generated details use Measurements, Method, Reproduce,
and Code tabs, with metric groups and provenance folded by default and a bounded
source-code viewer. Preserve the canonical scripts and exact downloads.
Use diagnostic plots alongside readouts and structured tables. Retain actual
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
Release before publishing the package tag. Ordinary CI retains immutable per-attempt preview report/plot assets, with no publication
commits, branches, or pull requests. Keep the recorded bootstrap snapshot in
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
