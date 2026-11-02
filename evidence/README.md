# relflow evidence

This directory owns the 79 controlled synthetic experiments, their protocol,
lab runner, runtime tests, and current publication results.
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

## Required lab execution

The library's workflow shares one candidate wheel across tests, Evidence, and
release, and builds an exact baseline wheel when needed. It
calls .github/workflows/evidence-lab.yml at a reviewed full commit SHA, passing
a separate reviewed suite SHA. The helper pins the trusted executor. GitHub
joins Tailscale using short-lived OIDC and dispatches one experiment per Docker
container on lab. Containers have bounded CPU/memory, no network, a read-only
root, and no host home or Docker socket. The full catalog runs for both wheels
with the original seeds and budgets, using verified references for unchanged
baselines. Every baseline-passing check must pass on
the candidate; baseline misses remain reported. CI queues lab work across PRs
and cancels superseded attempts within each PR. Errors, missing records, invalid
provenance, smoke execution, and partial coverage fail CI / required.

The combined CI gate also requires the library test matrix and static
documentation validation. The suite normally comes from the candidate commit
in this repository. A new suite can be validated explicitly with
`gh workflow run ci.yml --repo relflow/relflow --ref main -f evidence_ref=<full-suite-commit>`.
This tests the selected suite without changing the trusted executor or
Tailscale trust.

Before training, the pinned helper probes the actual cached container image,
host CPU/kernel, and resource/thread settings. It authenticates complete
successful CI reports and proves their source and wheel-payload equivalence
to the current request. Report-only updates must exactly match archived
candidate measurements and their provenance. A reuse receipt records the
current request, original run/artifacts, explicit role mapping, and all hashes;
it does not change or clone an original execution record. A miss runs the
suite. The required Evidence job always performs validation.

Baseline execution reuse is local to the trusted lab controller. Its cache is
outside worker mounts, limited to eight source reports and 1 GiB. Retained
reports carry self-contained source proofs, bounded to two levels and 256 MiB.
Missing, invalid, changed, or overly deep proofs cause fresh execution.

See [lab.md](lab.md) for trust and repository variables. No cloud account, stored
Tailscale secret, or self-hosted GitHub runner is required.

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

To execute every task without report or baseline reuse:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main \
  -f force_fresh_evidence=true
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

Lab defaults to 16 workers, with the longest recorded experiments scheduled
first. Wheels are installed in cached, isolated image layers rather than in
every task. P025 keeps all three seeds and runs two trials concurrently within
its existing CPU limit. Every report retains timing and CPU/memory telemetry.

Verified reuse compares execution inputs. Publication, documentation, and test
changes do not retrain unchanged models. Runtime, dependency, seed, budget, and
policy changes invalidate the affected measurements. Complete coverage and the
paired regression gate remain required. See [lab.md](lab.md#execution-cost-and-timing).
