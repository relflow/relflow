# relflow evidence

This directory owns the 79 controlled synthetic experiments, their protocol,
lab runner, runtime tests, and current publication results.
Stable P identifiers identify experiments throughout the runner, reports, and
website. Each script's frontmatter records its ID, group, selected full run,
metrics, checks, seed, environment, source provenance, and retained report link.

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

The library's workflow builds exact baseline and merge-candidate wheels and
calls .github/workflows/evidence-lab.yml at a reviewed full commit SHA, passing
a separate reviewed suite SHA. The helper pins the trusted executor. GitHub
joins Tailscale using short-lived OIDC and dispatches one experiment per Docker
container on lab. Containers have bounded CPU/memory, no network, a read-only
root, and no host home or Docker socket. The full catalog runs for both wheels
with the original seeds and budgets. Every baseline-passing check must pass on
the candidate; baseline misses remain reported. CI queues lab work across PRs
and cancels superseded attempts within each PR. Errors, missing records, invalid
provenance, smoke execution, and partial coverage fail CI / required.

The combined CI gate also requires the library test matrix and static
documentation validation. The suite normally comes from the candidate commit
in this repository. A new suite can be validated explicitly with
`gh workflow run ci.yml --repo relflow/relflow --ref main -f evidence_ref=<full-suite-commit>`.
This tests the selected suite without changing the trusted executor or
Tailscale trust.

See [lab.md](lab.md) for trust and repository variables. No cloud account, stored
Tailscale secret, or self-hosted GitHub runner is required.

## Reports and publication

Every main push runs the complete paired suite. After the complete CI run
succeeds, the evidence publisher automatically validates and retains its
reports in an immutable Release in `relflow/relflow`. It opens a publication
PR updating experiment results and the README measurements. Merge that PR
through the normal checks and review to update the Read the Docs build.
Publication-only changes do not recursively open another publication PR.

Routine CI reports are Actions artifacts retained for 90 days; input wheels
and requirements are retained for seven days. To refresh the current
measurements explicitly, start another full run on `main`:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main
```

Successful manual main runs also feed automatic publication. To retain a
completed run explicitly and obtain a publication patch for review, use:

```sh
gh workflow run publish-evidence.yml --repo relflow/relflow \
  -f tag=<unique-release-tag> -f library_run_id=<library-actions-run-id>
```

The publisher verifies complete coverage, checks, provenance, source and wheel
digests, and the runtime lock. Failed, stale, partial, or regressing runs cannot
be selected. The site's normal build reads committed records and canonical
sources. Report Releases do not replace the latest package release.

Select the candidate records from that completed full run, preserving their
actual metrics, checks, seeds, environment, tested revisions, and source digests.
Updating source or output readers does not establish a fresh learning result;
replacing the current selection requires new complete execution.
