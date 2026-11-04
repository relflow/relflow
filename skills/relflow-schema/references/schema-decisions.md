# Schema decisions

Checked against relflow 0.4.4 source. Recheck the installed version for other
releases. A loadable declaration does not prove useful predictions or sufficient
context. Construction resolves declarations; training learns model state.

| Observation | Investigate | Candidate |
| --- | --- | --- |
| Numeric quantity | Units, offsets, nonfinite values, regression objective | `rf.Number` |
| Small descriptive vocabulary | Unseen labels, capacity, case/whitespace semantics | `rf.Category` |
| Large identifier space | Equality, namespaces, unseen identities | `rf.Hash` |
| Fixed label universe | Permitted values and unknown behavior | `rf.Enum(values=...)` |
| Free text | Language, encoder, tokenization and storage cost | `rf.Text` |
| Numeric array | Fixed coordinates versus repeated measurements | `rf.Vector` or preprocessing into records |
| Unordered labels | Whether order or multiplicity matters | `rf.Set` |
| Related object records | Ownership, cardinality, ordering, information path | `rf.Branch` |
| Calendar timestamp | Parts and timezone semantics | `rf.Period(...)` |
| Recency or elapsed time | Anchor, units, censoring | Derived value read by `rf.Number` |

These are candidates, not an exhaustive registry. Extensions own atom compatibility,
semantic validation, resources, and tensorization. Arrow compatibility alone does
not convey the task-specific meaning preserved by an extension.

Numeric-looking identifiers may need strings to preserve leading zeros. Record
preprocessing explicitly: changing representation changes model input. Distinguish
null from zero, null collections from empty collections, and null items from both.

The anonymous root at `/` is singleton. Parents supply names; fields and branches
have no public `name` argument. Use bare classes without options. Generated names,
including configuration collisions, belong in `fields` mappings. Do not generate
`Struct(...)` or the unimplemented `Model.from_datamodule(...)`.

`mask=True` hides a field from input and makes it a supervised reconstruction
target. `embed=True` requests representations. Do not reconstruct every field
by default when the user has one target.

## Relationships and time

Determine prediction grain before grouping. Audit keys, foreign keys, join/output
row counts, and per-anchor identity over the declared scope; a prefix cannot prove
uniqueness. Deduplicating attachment keys differs from deleting repeated events.
Preserve real event multiplicity and outer anchor rows. Censor every lookup,
history, and time-varying root attribute at the prediction cutoff.

Attach needed referenced context to the row that uses it when that matches the
task. Equality-carrying identifiers do not guarantee learned multi-hop retrieval
between independent collections. Use singleton lookups for proven unique relations;
do not guess cardinality from names or erase genuinely repeated relations.

`query` is explicit node-relative structural selection. It does not implement joins,
sorting, cutoff filtering, or derived values. Those belong in preprocessing.

## Geometry and objectives

Report candidate-pool limits separately from `Branch.length`. Omitted length is
unbounded, not an automatic safe memory setting. Nested finite axes multiply;
inspect width distributions and extreme examples, not just average lengths or
parameter count. Mean reduction can erase cardinality; unreduced context can
multiply downstream work. Verify the reduction against the particular task.

For oldest-to-newest histories, `overflow="tail"` keeps recent records. Finite
`overflow="random"` uses contiguous windows during training and tails in validation,
test, and prediction. Variation requires a candidate pool larger than the cap.
Ordering and a separate resource limit on that pool are part of the choice.

`Period` preserves selected calendar parts; derive elapsed age separately. Current
`Number` preserves float64 source values except on MPS, rejects integers beyond
+/-2^53, and treats NaN as `other`. Current category admission preserves zero content
rows until a later training batch starts them. Historical notes about numeric log
lanes or immediate admission shifts do not override current contracts.

Choose reconstruction objectives against the external metric. Raw losses across
different field types or objectives may be incomparable. Keep metric, population,
split, and budget fixed when comparing schemas. Training, downloads, and HPO are
separate from schema compatibility validation.
