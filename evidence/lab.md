# Lab evidence execution over Tailscale

The library workflow builds candidate and baseline wheels on GitHub-hosted
workers. A pinned reusable workflow joins the tailnet using GitHub OIDC,
dispatches the suite to lab over Tailscale SSH, retrieves the report, and leaves
the ephemeral node. Lab runs experiments in separate bounded Docker containers.
There is no general GitHub runner installed on lab and no cloud compute account
or stored Tailscale client secret is required.

## Lab prerequisites

- Docker Engine available to the chosen SSH user.
- Python 3.10 or newer with PyYAML for the trusted coordinator.
- Tailscale SSH enabled and sufficient disk space for the cached runtime image.
- Sixteen workers use at most 32 logical CPUs and 64 GiB of container memory. Each
  experiment keeps its original seed, budget, and CPU behavior profile. The
  host GPU is not used for the CPU behavioral gate.

The dispatcher writes only under `~/relflow-evidence/runs/<run-id>/`, with unique
run directories. It never installs RelFlow or its runtime dependencies on the
host. Each exact wheel is installed once in a cached image layer with network access
disabled. Fresh containers mount the pinned executor and suite read-only, plus a
per-task output directory. They have no network, dropped capabilities,
no new privileges, a read-only root, and explicit CPU, memory, and PID limits.
Interruptions clean up containers belonging to that run and retain incomplete
reports; incomplete execution cannot pass the required gate.

## Tailscale trust configuration

Create a federated identity in the Tailscale admin console's Trust credentials
page. Configure:

| Setting | Value |
| --- | --- |
| Issuer | GitHub Actions, `https://token.actions.githubusercontent.com` |
| Subject | `repo:relflow@312307812/relflow@1217201723:*` |
| Custom claim `repository` | `relflow/relflow` |
| Custom claim `job_workflow_ref` | `relflow/relflow/.github/workflows/evidence-lab.yml@101c70f623f6bd75c481431db84d269f61097996` |
| Scope | `auth_keys` write, restricted to `tag:relflow-evidence-ci` |
| Device tag | `tag:relflow-evidence-ci` |

The subject above matches the ID-qualified subject observed from GitHub for
this repository. Use the issuer-reported format; a repository-name-only subject
does not match it.

The helper commit is the complete SHA in the library workflow's `uses:` line.
Keep the custom workflow claim exact; do not replace it with a wildcard. The
immutable helper validates the calling repository and PR source, then checks
out a separately pinned, tested evidence implementation. A fork cannot gain
tailnet access by changing its own workflow.

The trusted executor is pinned to
`041e8f1f09c5e3d0238487379a86e716e43af039`; the helper is pinned to
`101c70f623f6bd75c481431db84d269f61097996`. Keep these immutable commits
reachable. Routine experiment and publication changes do not change the pins.
A trusted-mechanics update requires a separately reviewed executor/helper
revision and the corresponding exact Tailscale workflow claim before validation.

Grant the CI tag access to lab's SSH service and allow noninteractive Tailscale
SSH for the selected account using an `accept` rule. Retain existing human
access rules. Review the existing policy when limiting destinations: an older
allow-all rule can grant broader network access than a new narrow rule.
Interactive SSH `check` rules require a person and cannot serve unattended CI.

Copy the generated Client ID and audience. These identifiers are public
configuration, not credentials. The workflow uses the generated audience
`api.tailscale.com/<client-id>`.

References: [Tailscale workload identity federation](https://tailscale.com/docs/features/workload-identity-federation),
[the official GitHub Action](https://github.com/tailscale/github-action).

## Library repository configuration

Set these Actions variables in `relflow/relflow`:

- `TAILSCALE_EVIDENCE_OIDC_CLIENT_ID`: the federated identity's Client ID.
- `EVIDENCE_LAB_TARGET`: the permitted SSH account and machine, such as
  `evidence@lab`.
- `EVIDENCE_LAB_HOST`: optional connectivity probe target; defaults to `lab`.

The library owns `ci.yml`, its single entry point for PRs, merge groups, main
pushes, and explicit manual validation. Reusable `tests.yml`, `docs.yml`, and `evidence.yml`
run the full library matrix, static documentation checks, and paired evidence.
The latter pins the exact helper commit in this repository; the helper separately
pins its trusted executor. `ci.yml` supplies the candidate SHA as the suite, or
the explicit manual override. The combined
`CI / required` job always evaluates their results. Main pushes build package
distributions alongside tests and documentation checks; release and coverage workflows consume the
successful main CI run without repeating its test matrix. Require
`CI / required` and an up-to-date base or merge queue in branch protection. Preserve the
repository's existing requirements and bypass policy.

Every event validates complete paired evidence. PRs and merge groups compare
their actual candidate and base revisions; main pushes use the pre-push base,
including multi-commit pushes. Manual validation uses the first parent or an
explicit full ancestor SHA supplied as `evidence_base`. The
helper checks the retained input request against the actual GitHub event.
Reports preserve the original measured revisions even when reused.
Version releases select their successful main CI run and retain verified reports
plus frozen publication JSON assets before the package tag is published. Ordinary
main CI keeps its gate and artifacts without opening publication PRs.

The shared package build supplies the same candidate wheel to SDK tests,
Evidence, and release. Before training, the trusted helper probes the actual
pinned Docker image, host CPU/kernel, and resource/thread policy, then checks
authenticated complete reports for equivalent source and package payloads.
Full-report reuse uploads a separate receipt and original input artifacts;
the report's records and provenance remain unchanged. The collector may also
reference unchanged baseline executions from its bounded host cache. Workers
cannot mount or modify that cache. Failed or incomplete executions never seed
it, and a missing proof runs fresh tasks.

Missing trust configuration fails the job with a clear setup message. It never
turns a skipped or missing evidence run into a successful required check.

## Manual validation

Build both wheels from the desired Git revisions and export the candidate's
locked runtime requirements. From clean executor and suite checkouts (which may be the same checkout):

```sh
uv run --project docs python evidence/ssh_run.py --suite-root evidence --target evidence@lab \
  --candidate-wheel /path/to/candidate/relflow.whl \
  --baseline-wheel /path/to/baseline/relflow.whl \
  --candidate-sha <candidate-sha> --baseline-sha <base-sha> \
  --requirements /path/to/requirements.txt --output evidence/runs/<unique-run-id>
```

Supply actual wheel filenames from the build. Add `--ids P001 --steps 2` for a
smoke test. It produces execution records but deliberately fails the learning
gate because it is partial and uses a reduced budget. Complete runs produce
independent JSON records, logs, provenance, and an HTML report.

## Updating or validating the suite

Change evidence sources through a normal PR alongside the library. Required
evidence uses that PR's candidate commit as the suite. For an explicit test of
a prospective suite commit against library main, run:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main \
  -f evidence_ref=<full-suite-commit>
```

This performs the complete paired suite and records both the suite and executor
revisions. Manual validation does not replace the PR-triggered required check.
The helper and executor pins change only when their trusted mechanics change;
then update the exact Tailscale workflow claim as part of that review.

A shared `relflow-evidence-lab` concurrency group uses `queue: max` so different
PRs wait for lab capacity. The outer CI workflow cancels superseded runs of the
same PR. No long-lived tailnet credential or self-hosted runner is needed.

## Execution cost and timing

The default is 16 containers on lab's 16 physical cores and 128 GiB of RAM.
Each container keeps its two-CPU ceiling, 4 GiB memory limit, isolation, and
full task timeout. Each training process retains one Torch thread. P025 runs
its original three seed trials in two isolated processes inside that ceiling;
the ordered results feed the same median and worst-case checks. GPU calibration
is deferred; the required profile remains CPU.

Tasks start in descending recorded duration. Publication results supply
scheduling hints only: they cannot establish a passing task or authorize reuse.
Each report includes `timing.json` with preparation and execution wall time,
container durations, summed fresh-task CPU seconds, and peak child-process RSS.
These distinguish less waiting from less computation.

Cache keys bind the exact wheel payload, exported runtime requirements, trusted
executor, resource/thread policy, seed, budget, experiment source, and shared
files under `experiments/`. An edit to one experiment invalidates its baseline
while preserving other unchanged experiments. Documentation, static result
assets, publication tooling, and tests do not invalidate training. The complete
source snapshot and raw fingerprints remain retained for audit. Runtime helper
or dependency changes still invalidate affected execution.
Each worker receives a read-only copy of exactly its declared runtime files:
its experiment and shared helpers/data. Other experiment scripts and publication
files are absent, and the input copy is outside the writable results mount.

Full-report reuse authenticates the original input request and verifies a PR's
synthetic merge parents against its recorded base and GitHub workflow head.
It does not depend on GitHub preserving a PR association after a merge. Reuse
can recover original Actions artifacts or frozen `evidence-v<version>` assets.
A partial CI retry may retain an earlier successful Evidence job only after
the complete CI run succeeds for the same revision and workflow; its measured
attempt remains unchanged.
