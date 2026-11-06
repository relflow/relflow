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
records. Experiment frontmatter owns definitions and prose; measured results live
in publication JSON assets. Never write results back into experiment sources.

GitHub-hosted CPU jobs use exact supplied baseline and candidate wheels, with
one bounded, network-disabled, read-only Docker container per task. Eight shards
keep each experiment's baseline and candidate on the same runner; a separate
collector verifies the entire original catalog. Mount neither the host home nor
Docker socket into workers. Keep credentials outside worker containers.
Smoke, partial coverage, invalid provenance, execution failures, and regressions
of baseline-passing checks cannot satisfy the required gate. Reports use Actions
artifacts and immutable Releases in relflow/relflow, with no AWS dependency.
Publish candidate measurements only with a package version release, from its
complete passing paired CI run in relflow/relflow. Freeze the publication JSON,
reports, and actual execution provenance in an immutable `evidence-v<version>`
Release before publishing the package tag. Ordinary CI must not create publication
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
