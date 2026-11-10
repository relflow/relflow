# Cluster-to-Group Evidence migration design

Status: **explicit transition policy; actual CI results are reported by the
required gate**. The Group change removes `Cluster` intentionally. The ordinary paired executor
runs the candidate suite against both wheels and requires identical checks.
That cannot compare a Group-only experiment against a Cluster-only baseline.
The failed runs also reused Cluster proof IDs for Group scripts despite changed
methods and checks. Neither a missing baseline nor a renamed proof is a measurement.

The failed PR #87 attempts `42506090355` and `42729355108` retained baseline
execution errors and check-set mismatches for P018, P019, P020, P078 and P095.
All five candidate records also had unmet checks in each attempt. Those are
historical results for those exact runs, not measurements of a newer candidate.
P065 was removed from the candidate catalog. Its historical Cluster result and
identity must not be reused.

The six baseline proof sources for current main
`c0cae6df38010e232c0f12c966eaa6a06fae4b94` are preserved byte-for-byte in
`migrations/cluster_to_group/`, with source paths and SHA-256 digests in its
manifest. Their source bytes also match the earlier `a912f291` baseline used in
the retained lab investigation; its measurements are not measurements of the
new baseline wheel. The transition planner and collector use this inventory
when the independently pinned executor includes the reviewed policy.

`migration.py` verifies the pinned inventory and distinct replacement IDs and
classifies baseline and candidate failures separately. The planner and
collector integrate that logic and retain their usual task and provenance
checks. An executor or helper pin without this policy cannot perform the migration.
The Group scripts use the distinct IDs P108–P112. P109 is a new proof with
fresh seeds, independent observation splits and final-assignment criteria.
P111 and P112 retain their quality and control thresholds but use extended
budgets and fresh seeds in predeclared lab experiments. P108 and P110 have
new fixed-capacity recovery protocols, independent observations, fresh seeds,
and held-out identity-shuffle controls in predeclared lab experiments.

The initial P108 and P110 Group versions expected an adaptive reduction from
15 slots even though Group has fixed slots and does not promise balanced or
compact use. Their replacements test recovered assignments and held-out
predictions with controls. The former Group P019 checked every epoch
against a final-quality ARI target; the new P109 checks final assignments and
held-out predictions with an identity-shuffle control. The 600-update Group run
now named P111 missed its partition and prediction gates, and the Group
experiment now named P112 also missed its error gate. The new lab experiments
met every check for all three P109 and P112 seeds, but only two of three seeds
for each of P108, P110, and P111. The [lab record](../research/group-ci-20261110/README.md)
retains the unmet checks and controls. These results are not hosted CI records.
Further experiments must
state their intended behavior, budgets and criteria before independent runs. Do
not weaken existing checks or reuse old measurements to make the migration pass.

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

This transition may reveal real Group learning failures. Until the complete
hosted run is validated, `CI / required` must not silently relax the gate.
