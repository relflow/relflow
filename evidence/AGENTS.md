# Canonical relflow evidence

Preserve stable P identifiers, generators, seeds, budgets, optimizers, capacities,
checks, and current result provenance. Read SPEC.md before changing semantics. Never
rewrite measurements to make a gate pass. The root AGENTS.md also applies.

Read prediction output with Polars DataFrames and Series. Use
`pl.DataFrame(model.predict(...))` for metric checks; do not
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

Each experiment declares an isolated uv environment using standard PEP 723
`# /// script` metadata, with `requires-python` and its direct dependencies.
Include the reporter's `pyyaml` and `torch` requirements and the relevant relflow
extra for optional features. Keep question metadata in the first Markdown/YAML cell and put the TOML block
in the first code cell. Use plain Markdown sections and standard `typst` fences;
put figure caption/alt text in `evidence-diagram`. Current sources must not
contain Quarto execution options, shortcodes, or disclosure markup. Keep the shared import path
relative to `__file__`, so direct `uv run <script>` works outside the checkout.
Use `uv run evidence/experiments/run.py --wheel /path/to/relflow-<version>-py3-none-any.whl P001`
to exercise an exact SDK wheel. The runner resolves all dependencies together,
preserves extras, clears inherited import paths, and records the wheel digest.
Listing and scaffolding remain static and require no SDK. Hosted paired CI keeps
its locked offline runtime and exact supplied wheels; script metadata must not
enable network resolution there or change the required catalog.

Own classification in each script's `[tool.relflow.evidence]` table: one
`capability`, a `types` list of primary subjects, and a nonempty `conditions` list.
Use the shared vocabulary in `docs/starlight/src/evidence/catalog.ts`; that module
contains labels, never lists of experiment IDs. Tags describe the protocol, not
the selected outcome. Static docs reject unknown, duplicate, or missing tags.
Replace the scaffold's `unreviewed` placeholders before completing an experiment.
Never rewrite frozen release sources to add tags or infer current datatype tags
for retired proofs. Current documentation shows the active script catalog.

Each experiment is one removable unit: its file owns the environment, schemas,
generator, training protocol, checks, review, diagnostic plan, optional custom
renderers, publication highlights, and scheduling estimates. Adding or removing
it must require no edit to a shared per-ID table. Shared code owns vocabulary,
validation, execution, and reusable plotting primitives only.

YAML frontmatter owns `evidence-review` (question, modeling, comparison paths,
units, scope note, and optional criterion labels), `evidence-plot`, optional
`evidence-highlight`, and optional per-role `evidence-cost` estimates. The latter
are advisory seconds, not measurements or training budgets. There are no
central review, view, highlight, or timing catalogs and no authoring compatibility
paths for those catalogs.

A diagnostic plan selects either a shared `mode` or a `renderer` function in
this same script, with optional named `supplements` using the same contract.
Custom renderers declare `(bundle, spec)` and put their imports and helpers
inside the function. Return a Matplotlib figure, caption string, and portable
interactive-data mapping (an empty mapping is valid). Only this function is loaded in the bounded offline plot
container; module imports, schemas, generators, and training do not execute.
Render captured observations only. Declare plotting dependencies in the script's
uv block. Static documentation never loads or executes a renderer.

All canonical studies live in `experiments/<category>/` and belong to the same
required catalog. There are no execution lanes, opt-in study lists, or directory
classifications. Every script declares its uv dependencies and owns its tags;
IDs are discovered from declarations without a central list. Static docs and
authoring checks inspect the complete catalog without importing model resources.

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
The active suite supports relflow >=0.5.0. Pre-0.5 event bases and authenticated
rewritten pushes use the fixed 0.5 anchor in `releases.py`, with the original
event base retained in `comparison.json` and the manifest. Do not add SDK
compatibility shims, run old wheels, or change learning thresholds to bridge
the breaking release. See [hosted.md](hosted.md) for selection and trust rules.
Smoke, partial coverage, invalid provenance, execution failures, and regressions
of baseline-passing checks cannot satisfy the required gate. Reports use Actions
artifacts and immutable Releases in relflow/relflow, with no AWS dependency.
Freeze canonical versioned candidate measurements only with a package version release, from its
complete passing paired CI run in relflow/relflow. Freeze the publication JSON,
reports, and actual execution provenance in an immutable `evidence-v<version>`
Release before publishing the package tag. Ordinary CI retains per-attempt
reports and plots only as GitHub Actions artifacts. It creates no public
Evidence releases, source commits, branches, or pull requests. Hosted branch/PR
docs show the latest supported formal release's frozen results, or pending
measurements until the first >=0.5.0 release. New release downloads use
`relflow-<version>-evidence*` names; existing published names remain immutable.
Keep generated publications and executed source bundles outside Git. Local builds without a selected
publication show pending measurements. Hosted builds verify the public release
asset's digest and source revision, without model dependencies or credentials.
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
