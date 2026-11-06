# relflow evidence protocol

This repository owns the relflow package, website, and canonical controlled experiments.
Each script's frontmatter declares its stable P identifier, title, group, and
presentation options. CI discovers experiments directly from those headers.
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

## Execution

Each script exposes `run(seed, steps, accelerator)` and calls the shared
reporter under a main guard. Execution produces a new immutable JSON record,
including failures. It never edits sources or a selected publication. A `--steps`
cap always identifies a smoke run and cannot establish learning capability.

Run these commands from the repository root with the intended wheel:

```sh
uv run --project docs python evidence/experiments/run.py --list
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

The hosted workflow assigns whole baseline/candidate experiment pairs to eight
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

Workers have no network, a read-only root, and no host home or Docker socket
mount. Install runtime packages in the image, never on the host. Keep credentials
outside workers. Executor and helper code remain separately pinned; selected
suite code executes only inside task containers. Fork execution fails closed.
Superseded attempts are cancelled per PR. No tailnet or lab service participates
in the required workflow. See [hosted.md](hosted.md).

Hosted measurements execute freshly for both roles. Dependency-image caches
accelerate setup without substituting for any experiment. Preserve historical
authenticated reports and their original runtime identities; an earlier lab
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
The header declares `evidence-id`, `code-fold: true`, and
`execute: {enabled: false, eval: false}`. Use the Markdown engine, with no Jupyter
or model dependency in the documentation build.

Each page asks one question and shows an Example, Comparison, and Result, with
one YAML observation and a schema-faithful shared Typst tree. Keep training
settings, raw checks, measured provenance, and reproduction commands inside the
closed details foldout. Select one to three numerical readouts using explicit
metric paths and formats. Explain the measured limit; AUC is not accuracy and
learned arithmetic is not exact accounting.

`evidence/render.py` generates ignored `docs/evidence/` pages from canonical
scripts and the selected publication JSON. The checked-in `results.json` is an
unchanged bootstrap snapshot for offline builds. Hosted release versions fetch
their matching public asset; unreleased branch and PR builds use the most
recent released version, or the bootstrap snapshot before the first asset-based
release. Missing or invalid assets for an actual release fail the build.
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
