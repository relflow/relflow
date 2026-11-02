# Canonical RelFlow evidence

Preserve stable P identifiers, generators, seeds, budgets, optimizers, capacities,
checks, and current result provenance. Read SPEC.md before changing semantics. Never
rewrite measurements to make a gate pass. The root AGENTS.md also applies.

From the repository root, `make check` validates static docs and evidence tooling.
Runtime tests under evidence/tests/runtime are explicit and require a separately
supplied RelFlow wheel; use `uv run --project docs --with /path/to/matching-relflow.whl`
to keep the intended wheel separate from the root editable SDK. Runtime tests
are excluded from default checks. Builds must never
import RelFlow, Torch, or Lightning. Execution writes independent immutable JSON
records; only a reviewed publication change updates experiment frontmatter.

Lab uses exact supplied baseline and candidate wheels, with one bounded,
network-disabled, read-only Docker container per task. Mount neither the host
home nor Docker socket into workers. Keep credentials outside worker containers.
Smoke, partial coverage, invalid provenance, execution failures, and regressions
of baseline-passing checks cannot satisfy the required gate. Reports use Actions
artifacts and immutable Releases in relflow/relflow, with no AWS dependency.
Preserve historical release URLs in selected experiment records.
