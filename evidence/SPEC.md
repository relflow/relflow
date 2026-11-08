# relflow evidence protocol

This repository owns the relflow package, website, and canonical controlled experiments.
Each script declares its stable P identifier and default seed with
`@experiment("P001", seed=123)` on its `run(seed, steps, accelerator)` function.
Frontmatter owns the title, group, and presentation options. CI discovers
declarations as syntax, without importing or executing experiment code.
Measured runs and report provenance are separate JSON publication assets; they
never rewrite the scripts. Website builds read static sources and result data
without executing models.

## Experimental contract

Each experiment is a self-contained annotated script under
`experiments/<category>/`. It owns its generator, model, training budget,
held-out evaluation, matched controls, measured metrics, and named Boolean
checks. Preserve those semantics when changing packaging or presentation.
Share reporting and CLI plumbing, not generators, datasets, or model helpers.
Never reuse a stable identifier for a different experiment.

Use independent training, validation, and test observations. Exclude target
answers from inputs, control leakage, and test the declared generalization
boundary. Include a matched negative control where signal, context, alignment,
or relevant structure is removed. Low accuracy can be the correct outcome for
an information boundary. A falling training loss alone is not evidence of
held-out inference.

Declare seeds, sample counts, parameter budgets, optimizer settings, precision,
and training steps in the script. Keep the same data/model seeds and budgets
when comparing baseline and candidate. Do not tune on the test split, lower a
threshold to pass a candidate, or retry new seeds until a run happens to pass.
A single-seed experiment is provisional; promote a mature core claim using
three paired seeds and its declared held-out metrics and controls. Preserve any
internal seed panels already implemented by an experiment.

## Authoring scaffold

`experiments/run.py --new <category>/<name>` creates a new annotated Python file
from `templates/experiment.py.txt`. It allocates an ID above the current maximum
without reusing retired gaps and writes a literal default seed of `0`. Optional
title, group, ID, and seed arguments configure the declaration and presentation.
Discovery remains static, and creation imports no experiments or model packages.
Existing files, unsafe paths, and duplicate explicit IDs are rejected.

The scaffold supplies the header, example/comparison/result sections, model-tree
illustration, decorator, and reporting entrypoint. Its scientific body raises
`NotImplementedError`; it must never return fabricated passing measurements or
checks. Authors own the generator, independent splits, model, budgets, controls,
evaluation, and numerical criteria. A scaffold is not a recorded experiment.
Once created under `experiments/`, it joins the catalog and must be completed
before authoring and full paired checks can pass.

## Execution

The required hosted catalog is the **core** lane under `experiments/<category>/`.
It always includes every core declaration. Studies requiring additional model
artifacts, optional solvers, or an unreleased datatype wheel live separately in
`research/<category>/` and declare an `evidence-requires` list in frontmatter.
Their stable IDs share the same namespace. Directory placement owns the lane;
frontmatter cannot reclassify a required core experiment. Both lanes use the same
standalone protocol, immutable records, and static documentation renderer.

The local runner defaults to core. Select research explicitly with
`--lane research`; `--lane all --list` lists both. Documentation lists both lanes
and marks research pages, including their provisioning requirements under
Reproduce. A missing research result remains Missing; excluding optional
resources from hosted core execution never creates a passing result.

Each script exposes a decorated `run(seed, steps, accelerator)` and calls
`report(run)` under a main guard. Registration attaches immutable metadata and
returns the original function; it never runs the experiment. There is no mutable
process-wide catalog. Static discovery rejects duplicate IDs, multiple decorated
entrypoints, nonliteral IDs or seeds, and unsupported function signatures.
Historical undecorated suites retain their frontmatter and reporter-call reader
for archived execution and publication validation.
Execution produces a new immutable JSON record,
including failures. It never edits sources or a selected publication. A `--steps`
cap always identifies a smoke run and cannot establish learning capability.

Run these commands from the repository root with the intended wheel:

```sh
uv run --project docs python evidence/experiments/run.py --list
uv run --project docs python evidence/experiments/run.py --lane all --list
uv run --project docs --with /path/to/matching-relflow.whl python evidence/experiments/run.py P001 --steps 2
uv run --project docs --with /path/to/matching-relflow.whl python evidence/experiments/run.py P001
uv run --project docs --with /path/to/matching-relflow.whl pytest -c docs/pyproject.toml evidence/tests/runtime
```

The isolated documentation project keeps the supplied wheel separate from the
root editable SDK. The ID runner supplies the experiment import path. Local records go under
`evidence/runs/local/`; `--output-dir` selects another directory. Runtime tests
are explicit and excluded from the default static checks.

## Required merge gate

The library workflow builds wheels from the exact PR merge candidate and base,
exports the candidate's locked requirements, and calls the reusable hosted
workflow at a reviewed full commit SHA. The trusted host executor is pinned independently of the experiment suite.
The suite defaults to the candidate commit in this repository, with an explicit
full SHA override available for manual validation. Host code reads it as data and
runs experiment code only in isolated containers. Require `CI / required` and an up-to-date base so a changed
main cannot reuse stale validation.

Validate the complete catalog against both wheels in the same CPU environment,
using every experiment's declared default seed, full budget, and internal seed
panel. Every check that passes on the baseline must pass on the candidate.
Baseline-unmet checks remain visible, and improvements are allowed. An error on
either side, changed check sets, missing or duplicate records, altered tasks,
invalid provenance, cancellation, timeout, partial coverage, or a smoke budget
must prevent a successful gate. Preserve the named numerical checks unchanged.

Historical single-host reports may contain authenticated reuse after proving
source, actual wheel payloads, comparison, and execution runtime equivalent.
Their original records and measured revisions remain unchanged, with a separate
receipt for the request and explicit role mapping. Retained baseline references
carry the original immutable execution. These historical proofs remain
verifiable; they do not authorize hosted shards to skip fresh measurements.
Every logical slot still requires a validated full-budget execution, and
publication must exactly match the archived candidate results and provenance.

The hosted workflow assigns whole baseline/candidate experiment pairs to twelve
GitHub-hosted CPU shards. Each shard runs at most two task containers at a time;
each container retains its two-CPU ceiling, 4 GiB memory limit, 512 PIDs, and
original experiment timeout. Baseline and candidate for a proof share the same
actual runner and may execute concurrently. Schedule individual tasks longest
first while retaining the two-container limit. Different shards may have
different CPUs or kernels, so retain
every shard's measured identity rather than describing the catalog as one host.
The collector requires the exact planned union, unchanged task identities,
matching within-pair execution policy, and a complete passing comparison.
Duplicate, missing, unexpected, or altered shards fail closed.

Scheduling uses the per-role costs in `timings.json`, with a reference to their
verified source report. Freeze their file digest and resolved costs in the plan.
Balance pairs by summed role cost; missing timings must have a positive fallback.
These forecasts affect placement and ordering only. They cannot change the task
catalog, seeds, training steps, checks, or measurement provenance.

Workers have no network, a read-only root, and no host home or Docker socket
mount. Install runtime packages in the image, never on the host. Keep credentials
outside workers. Executor and helper code remain separately pinned; selected
suite code executes only inside task containers. Fork execution fails closed.
Superseded attempts are cancelled per PR. See [hosted.md](hosted.md) for the
GitHub-hosted execution workflow.

Hosted measurements execute freshly for both roles. Dependency-image caches
accelerate setup without substituting for any experiment. Preserve historical
authenticated reports and their original runtime identities; a historical
execution must not be relabeled as a hosted measurement. Any future hosted
result reuse still requires the complete equivalence and authentication proof
above.

Infrastructure setup and a container launch that produced no worker record may
be retried within a fixed bound, retaining every launch receipt and log. A
learning result, failed check, worker exception, OOM, cancellation, or learning
timeout is terminal for that execution. Preserve workflow attempts separately;
rerunning failed jobs cannot silently combine different plans or rewrite an
earlier measurement's attempt identity. This hosted profile rejects a shard or
collector whose current GitHub attempt differs from the prepared input request.
Rerun all jobs so preparation, inputs, shards, and collection share one attempt.

## Records and publication

Every report identifies the executor, evidence suite, and library revisions, exact wheel and
script digests, environment lock, image and hardware, seeds, budgets, metrics,
checks, errors, and timing. The collector writes independent per-task records
and logs, then checks completeness and provenance before producing the JSON
summary, HTML report, and file checksums. Preserve separate attempts.

Routine reports are GitHub Actions artifacts for 90 days; input wheels and
requirements are retained for seven days. Every main push still validates the
complete paired suite against its pre-push base. Publication happens only for
package version releases. `release.yml` calls `publish-evidence.yml` with the
exact successful main CI run, source revision, and package version. The
publisher verifies coverage, checks, provenance, input hashes, and report
integrity before rendering a review site from the selected data.

The immutable `evidence-v<version>` Release retains `publication.json`, a
generated `measurements.md` table, complete report and input archives,
`github-run.json`, checksums, and the reuse receipt when applicable. The
publication must exist before the package's `v<version>` tag can be published.
Already published package versions keep their original evidence selection.
Publishing creates no source commits or follow-up PRs. No AWS storage or new
credential is required.

Publication JSON uses schema 1 with `version`, `revision`, `report`, and
`results` keyed by stable P identifiers. Keep metrics, checks, source
fingerprints, package versions, and execution provenance bound to the selected
execution. A reused report keeps its original measured revision and separate
current-request receipt; `revision` identifies the verified release request.
Smoke, incomplete, or regressing reports cannot be selected. A source edit or
output-reader update does not establish a fresh learning measurement.

## Documentation

Author prose in `# %% [markdown]` comment cells and Python in `# %%` cells.
The header declares `code-fold: true` and
`execute: {enabled: false, eval: false}`. Use the Markdown engine, with no Jupyter
or model dependency in the documentation build. Static discovery supplies
`evidence-id` from the decorator when generating documentation. An explicitly
supplied frontmatter ID must agree with the decorator.

Each page asks one question and shows an Example, Comparison, and Result, with
one YAML observation and a schema-faithful shared Typst tree. Generated pages
show predeclared behavioral criteria before the Example, with recorded met/unmet
statuses. The renderer injects the checks shortcode only into the generated
page; canonical scripts keep their authored prose and unchanged scientific code.
Select one to three numerical readouts using explicit metric paths and formats.
Explain the measured limit; AUC is not accuracy and learned arithmetic is not
exact accounting.

Write the overview for a reader who has not seen the generator. Explain what
each visible input and hidden target means, then walk through the example's
answer using the actual rule or label definition. Use concrete values within
the declared ranges and preserve the schema's real field names. Name what each
control changes and why that change should affect the answer or measurement.
Explain metric direction and reference values before interpreting the selected
result. Keep illustrative examples distinct from recorded predictions, and
tie claims to the displayed record rather than an unlinked historical campaign.
State the evaluation boundary, such as new observations of familiar identities
versus identities absent from training. Keep detailed settings folded while
leaving the question, worked example, comparison, and interpretation readable.

Present nested metric dictionaries and arrays as labeled disclosure groups and
scalar tables, with metric paths preserved in captions. Keep the complete raw
record available in its own disclosure, and keep training settings, measured
provenance, reproduction commands, and source code folded. Readable presentation
must retain all recorded values and named checks.

Use diagnostic plots alongside readouts and structured tables. During explicit
proof execution, retain the predictions, scores, visibility conditions, target
coordinates, reference vectors, and controls needed to explain the result.
Figures must identify their population, units, references, and evaluation scope;
use matched scales for compared conditions and distinguish a fitted projection
from original-space scores. Keep all unmet checks and full raw records visible.
Missing results remain explicit; generator illustrations contain no model
predictions. Bind measured figures to the exact captured source, record, and
observation checksums. See [diagnostics/README.md](diagnostics/README.md).
Presentation changes never alter scientific budgets or checks or establish a
fresh learning result.

`evidence/render.py` generates ignored `docs/evidence/` pages from canonical
scripts and the selected publication JSON. The checked-in `results.json` is an
unchanged bootstrap snapshot for offline builds. Hosted release versions fetch
their matching public asset; unreleased branch and PR builds use the most
recent released version, or the bootstrap snapshot before the first asset-based
release. An explicitly prepared PR diagnostic review instead selects its exact
full-budget captured records and figures together. These unpaired runs do not
replace the required paired gate or update release publications. Missing or invalid assets for an actual release fail the build.
Downloaded data is cached outside experiment sources and checked against
GitHub's asset digest and the package source revision. Never edit generated
pages as a second source. `make check` validates static sources and tooling;
`make check-docs` renders and checks the full site. Neither imports RelFlow,
Torch, or Lightning. `make clean` removes generated output and caches while
retaining canonical sources and current measurements.

## Execution equivalence

The complete source snapshot retains its raw integrity hash. Separate runtime
fingerprints bind each experiment and shared runtime files under `experiments/`;
publication assets, documentation, and tests are outside this execution identity.
The installed wheel payload, dependency export, seed, budget, trusted executor,
and actual resource/thread policy must also match before baseline reuse.

Scheduling order and cached image layers never reduce budgets or remove seeds.
Independent seed trials may execute in isolated processes when their original
ordered aggregation and checks remain identical. The collector still requires
every canonical baseline/candidate result. Timing data cannot satisfy a learning
check. GPU calibration remains deferred and cannot replace the required CPU gate.
