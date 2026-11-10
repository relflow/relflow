# Cluster-to-Group Evidence migration design

Status: **design only; not implemented or approved as a passing Evidence run**.
The Group change removes `Cluster` intentionally. The current paired executor
runs the candidate suite against both wheels and requires identical checks.
That cannot compare a Group-only experiment against a Cluster-only baseline.
The existing Group scripts also reuse Cluster proof IDs despite changed methods
and checks. Neither a missing baseline nor a renamed proof is a measurement.

The failed PR #87 attempt `42506090355` retained baseline execution errors and
check-set mismatches for P018, P019, P020, P078 and P095. All five candidate
records also had unmet checks in that attempt. Those are historical results for
that exact run, not measurements of a newer candidate. P065 was removed from the
candidate catalog. Its historical Cluster result and identity must not be reused.

## Contract

1. Allocate new immutable IDs for the five Group studies. Keep P018, P019,
   P020, P065, P078 and P095 as retired Cluster identities with their exact
   original source, seeds, budgets, checks and provenance. Preserve their
   published results and links; never rewrite those measurements as Group.
2. Declare an explicit, reviewable migration manifest: exact baseline Git SHA,
   each retired identity and its baseline source path and source hash, the new
   candidate identity where one exists, and the reason these are not directly
   comparable. P065 has no declared replacement. Refuse unknown retirements,
   duplicate mappings, changed baseline source, or a different baseline SHA.
   A new main SHA requires a newly reviewed declaration, not capability guessing.
3. Run every unaffected proof on both wheels with unchanged paired checks,
   budgets, seeds and regression rules. For each retired proof, run the exact
   baseline source and preserve its full result as a baseline-only record. For
   each new Group proof, run its full declared candidate budget as a distinct
   candidate-only record. The collector must require every declared task,
   validate all checks and provenance, and fail on execution errors or unmet
   candidate-only checks. Do not compare Cluster and Group Boolean check names
   or claim that an unpaired Group outcome matches a Cluster baseline.
4. Keep each related old/new task on one shard with the existing CPU, memory,
   image, network, timeout and fresh-execution policies. Baseline-only and
   candidate-only tasks must be declared in the frozen plan; no task may be
   omitted because it is slow, fails, or has already produced a measurement.
   Report both original records, all unmet checks, and an explicit
   `not directly comparable` relationship. No synthetic passing result is allowed.
5. Make this a versioned executor/manifest policy, including runtime-input
   hashes for both role-specific sources and the migration declaration. Verify
   it in planning, worker isolation, collection, report/reuse authentication,
   plotting, and release publication. Normal future PRs whose baseline already
   contains Group return to the ordinary full paired policy. A forged migration
   manifest, mismatched baseline SHA or omitted task must fail closed.
6. Freeze and validate the executor first, then the reusable hosted helper that
   pins it, and finally the consumer workflow pins. The executor and helper must
   be available at immutable repository revisions before GitHub can check them
   out. Do not point trusted execution at the candidate suite or reuse an old
   run under a new revision.

## Required verification before adoption

- Unit tests for exact task coverage, old/new ID uniqueness, declared baseline
  SHA and hashes, unchanged pair behavior, missing/duplicate/tampered tasks,
  source and image identity, unequal check sets, unmet new Group checks,
  omitted P065, and automatic return to regular pairing after migration.
- Build the exact baseline and candidate wheels. Execute every declared
  migration task on the same bounded host with its original seed and budget;
  retain the original baseline and candidate records. Execute the entire
  unaffected paired catalog, not a partial or smoke run, before claiming CI.
- Exercise all twelve shards and the collector, then verify visualization,
  publication and reuse reject incomplete, tampered, or unproven migration
  records. Smoke runs cannot satisfy the required gate.

This change needs executor, suite and workflow work and may reveal real Group
learning failures. Until those pieces are implemented, published and verified,
`CI / required` should remain failing rather than silently relaxing the gate.
