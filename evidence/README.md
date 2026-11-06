# relflow evidence

This directory owns the controlled synthetic experiments, their protocol,
hosted CPU runner, runtime tests, and current publication results.
Stable P identifiers identify experiments throughout the runner, reports, and
website. Scripts declare their ID, group, and presentation options. Measured
runs, checks, seeds, environment, source provenance, and report links live in
separate publication JSON assets.

Run from the repository root:

```sh
make check
uv run --project docs python evidence/experiments/run.py --list
uv run --project docs --with /path/to/matching-relflow.whl python evidence/experiments/run.py P001 --steps 2
uv run --project docs --with /path/to/matching-relflow.whl pytest -c docs/pyproject.toml evidence/tests/runtime
```

Supply an actual wheel. Use the isolated documentation project so the root
package environment cannot substitute the editable SDK for that wheel.
A two-step run checks execution only. Each execution
writes a new JSON file under evidence/runs/local; --output-dir selects another
directory. Execution never edits the reviewed publication selection.

## Required hosted execution

The library's workflow shares one candidate wheel across tests, Evidence, and
release, and builds an exact baseline wheel when needed. It
calls `.github/workflows/evidence-hosted.yml` at a reviewed full commit SHA,
passing a separate reviewed suite SHA. The helper pins the trusted executor.
Eight GitHub-hosted CPU shards run one experiment per Docker container, with
both roles for each experiment assigned to the same runner. Each shard runs at
most two containers concurrently. Containers have bounded CPU/memory, no network, a read-only
root, and no host home or Docker socket. The full catalog runs for both wheels
with the original seeds and budgets. Every baseline-passing check must pass on
the candidate; baseline misses remain reported. A separate collector verifies
all shards and their actual execution identities. CI cancels superseded attempts
within each PR. Errors, missing records, invalid
provenance, smoke execution, and partial coverage fail CI / required.

The combined CI gate also requires the library test matrix and static
documentation validation. The suite normally comes from the candidate commit
in this repository. A new suite can be validated explicitly with
`gh workflow run ci.yml --repo relflow/relflow --ref main -f evidence_ref=<full-suite-commit>`.
This tests the selected suite without changing the trusted executor.

The dependency-only runtime image is prepared once and delivered with its
verified identity. Each shard records its actual CPU/kernel, image, and
resource/thread settings; baseline and candidate retain those measurements.
Both roles run freshly. A dependency-cache hit never supplies a learning result
or bypasses the required collector. Historical authenticated reports retain
their original measurements, revisions, and runtime identities.

Only infrastructure operations that have not produced a worker measurement
receive bounded retries. Failed checks, worker exceptions, OOMs, cancellations,
and learning timeouts remain failures, with their logs and attempts retained.

See [hosted.md](hosted.md) for workflow trust, parallelism, and storage. No cloud
account, tailnet configuration, or self-hosted GitHub runner is required.

## Reports and publication

Every main push still validates the complete paired suite against the pre-push
revision. PRs and merge groups use their actual comparison base; manual runs
use the current commit's first parent by default; `evidence_base=<full-sha>`
selects an explicit ancestor for repeatable manual comparisons. Routine
reports remain Actions artifacts for 90 days, with tested input wheels and
requirements retained for seven days.

**Only package version releases publish new evidence selections.** The release
workflow passes its exact successful main CI run, source revision, and version
to `publish-evidence.yml`. After validating complete coverage, behavioral
checks, provenance, source/wheel digests, the runtime lock, and report integrity,
the publisher renders a review site from the selected records and freezes
`evidence-v<version>` before the package's `v<version>` tag is published.

The immutable evidence Release contains:

- `publication.json`: the versioned selection and original candidate results.
- `measurements.md`: generated comparison figures for publication.
- `evidence-reports.tar.gz`: records, logs, summary, and HTML report.
- `evidence-inputs.tar.gz`: exact tested wheels and runtime requirements.
- `github-run.json`, `SHA256SUMS`, and the reuse receipt when applicable.

Publication never edits experiment headers or the source README, commits a
bot branch, or opens a follow-up PR. An already published package version
keeps its evidence selection frozen. New measurements for publication require
a new version. Report Releases do not replace the latest package Release.

The checked-in `results.json` preserves the existing historical snapshot for
offline builds. `evidence/render.py` combines it, or an explicitly selected
publication JSON, with experiment definitions. Hosted docs read their matching
public release asset after verifying its digest and source revision. Unreleased
previews use the last released selection; before the first asset-based release
they use the bootstrap snapshot. No model import, cloud account, or new API
credential is required.

To execute another complete CI run without publishing a new selection:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main
```

The release workflow normally publishes evidence automatically. To prepare
assets explicitly for an unreleased version using its verified main CI run:

```sh
gh workflow run publish-evidence.yml --repo relflow/relflow \
  -f version=0.4.3 -f library_run_id=<successful-main-ci-run-id>
```

Keep actual metrics, checks, seeds, environment, tested revisions, and source
digests bound to the measured execution. A verified reused selection retains
its original identity and separate current-request receipt. Source edits and
reader changes do not establish fresh learning results.

## Runtime efficiency

Eight shards distribute whole proof pairs using recorded durations as scheduling
hints, with two containers per runner. Wheels are installed in isolated image
layers rather than in every task. P025 keeps all three seeds and its existing
CPU behavior. Every report retains timing and CPU/memory telemetry.

Dependency caches reduce setup; they do not remove seeds, reduce budgets, or
substitute historical measurements. Shards execute both roles freshly. Complete
coverage and the paired regression gate remain required. See
[hosted.md](hosted.md#execution-cost-and-timing).
