# Lessons from relarena

Interpreted from retained research inspected on November 4, 2026. Exact source
files and SHA256 digests are in [source provenance](relarena-sources.json). Paths
are relative to `Desktop/arena`; using this skill does not require that checkout.
Original receipts retain their own SDK, schema, query, and source identities.

1. **Geometry is part of the schema.** Uncapped nested histories failed even at
   tiny batches. Small parameter counts did not predict VRAM or wall time. Bound
   source pools and retained axes independently; inspect nested products before
   allocating batches. Packing does not remove all dense state.
2. **Audit join multiplicity before training.** Repeated attachment keys could
   expand child lists and attach them repeatedly, producing quadratic growth.
   Caps mitigated cost without repairing joins. Distinguish redundant attachment
   expansion from legitimate repeated events; preserve anchor identity.
3. **Provide the required context.** Raw/derived schemas changed available model
   input, so improvements cannot be attributed solely to architecture. Proven
   singleton relations can use row lookups; deeper context must justify its cost.
   Hashed equality does not guarantee learned cross-collection retrieval.
4. **Time is a data contract.** Censor histories, lookups, root attributes, and
   derived ages at the anchor. Preserve calendar structure and recency separately.
   Validation gains did not reliably transfer through temporal shift. Inspect
   validation support and seed variance before expanding search.
5. **Separate correctness and predictive quality.** The final three-task MEDIUM
   attachment-key dedup study showed tiny AUROC changes and 20.21% worse site-success
   MAE in its paired single-seed run. No dedup result replaced the frozen baseline.
   Visits also changed input order after recovery. Structural repair needs its own
   validation; these results neither justify duplicate joins nor prove universal
   predictive benefit from deduplication.
6. **More search is not automatically a gain.** The later three-strategy grid
   increased SMALL mean normalized test error by 1.40%; its incomplete MEDIUM panel
   did not establish an overall ranking gain. Early JEPA/contextual/hybrid arms did
   not establish consistent supervised transfer under tested budgets. Start with a
   supervised baseline when appropriate; do not rule out self-supervision generally.
7. **Mark development exposure honestly.** Shared-tier presets used development-task
   test feedback. The frozen 21-task eight-pass, seed-0 baseline and later comparisons
   retain that exposure. Do not select historical test winners per task, substitute
   later controls, or describe these research scores as untouched holdout estimates.
8. **Performance needs matched measurements.** Earlier text caching helped exact
   replay but hurt fresh-row throughput. Later text stores and worker changes have
   different implementations and scopes. Inspect reuse, tokenization, transport, and
   ownership. A short preflight does not prove full-fit capacity or all-task speedup.
9. **Preserve failures and cumulative work.** Optimizer/EMA restoration did not
   restore iterable cursors. Replay and reset clocks complicate budget eligibility.
   Keep exact data/schema/query/SDK identities, seeds, budgets, coverage, failures,
   and timing scope. Complete updates alone do not certify official eligibility.

## Historical recipes are priors

The singleton policy used one recipe per tier, finite history caps, random contiguous
training crops, deterministic inference tails, and supervised objectives. SMALL and
MEDIUM/LARGE differed in width, context, regularization, learning rate, and batch
policy. The source `PRESETS.md` records settings and test exposure. Do not promote
them to universal defaults or copy them silently into unrelated tasks.

Current relflow has packed execution with nonzero dropout, `Period` as the public
calendar field, updated category admission, and different numeric representations.
Do not ban current dropout or prescribe old numeric lanes from historical reports.
Validate the installed SDK; keep measurements attached to their executed versions.
