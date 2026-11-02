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
- Eight workers use at most 16 CPU cores and 32 GiB of container memory. Each
  experiment keeps its original seed, budget, and CPU behavior profile. The
  host GPU is not used for the CPU behavioral gate.

The dispatcher writes only under `~/relflow-evidence/runs/<run-id>/`, with unique
run directories. It never installs RelFlow or its runtime dependencies on the
host. Containers mount the pinned source and the selected wheel read-only,
plus a per-task output directory. They have no network, dropped capabilities,
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
| Custom claim `job_workflow_ref` | `relflow/relflow/.github/workflows/evidence-lab.yml@7cba2bf2fe3082e643a95aa43c6bad8395569420` |
| Scope | `auth_keys` write, restricted to `tag:relflow-evidence-ci` |
| Device tag | `tag:relflow-evidence-ci` |

The subject above matches the ID-qualified subject observed from GitHub for
this repository. Use the issuer-reported format; a repository-name-only subject
does not match it.

The helper commit is the complete SHA in the library workflow's `uses:` line.
Keep the custom workflow claim exact; do not replace it with a wildcard. The
immutable helper validates the calling repository and PR source, then checks
out a separately pinned, tested evidence implementation. A fork cannot gain
tailnet access by changing its own workflow. Moving the helper from relflow/docs
to this repository requires updating this external Tailscale claim to the new
exact reference before the consolidated workflow can run. Importing the source
does not change the federated identity configuration.

The trusted executor is pinned to import commit
`349b2202e6390a5fe0cbdf09a0c5972e495351ed`; the helper is pinned to
`7cba2bf2fe3082e643a95aa43c6bad8395569420`. Push both commits with the
migration branch and merge while preserving this history so these immutable
references remain reachable. Squashing or rebasing this migration requires
recreating the pins and updating the exact Tailscale claim before validation.

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

- `TAILSCALE_OIDC_CLIENT_ID`: the federated identity's Client ID.
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
successful main CI run without repeating its test matrix. Configure branch protection only after a
real successful run exists, requiring that check and an up-to-date base or
merge queue. Preserve the repository's existing requirements and bypass policy.

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
