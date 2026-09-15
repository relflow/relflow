# relflow Modeling Proof Specification

Status: active. Recorded results live in `results.yaml`, with implemented
coverage tracked in [`README.md`](README.md). Sections marked planned describe
hypotheses awaiting implementation and measurement.

Scope: empirical regression proofs for learned model behavior

## Expected User-Facing Architecture

The record-reasoning experiments in this suite are intended to improve model
behavior without requiring users to wire relational Q/K/V, joins, or reference
paths into the model. The schema already identifies structural coordinates and
reconstruction targets; relflow should use that information internally. This
does not replace structural ``query=`` paths or preprocessors: those still own
source selection, renaming, filtering, joining, sorting, windows, and derived
source values when observations do not already match the schema.

The prospective public contract is deliberately small:

- ``mask=True`` remains exactly
  ``Mask(skip=True, dropout=False, reconstruct=True)``. The example's target
  content remains available as supervision but produces no encoder parcel.
  Existing decoders already own learned target-shaped queries; any future
  encoder-visible or data-conditioned target slot must exclude that example's
  hidden answer from its forward context.
- Fields belonging to the same structural coordinate receive automatic record
  context before attention. This is an internal encoding behavior, not another
  schema option.
- A masked target may use a schema-owned latent query at its own coordinate to
  retrieve visible evidence. The learned query is conditioned on visible
  sibling fields. With ``decoder_position=None``, scalar decoders are
  position-free and coordinate-shaped targets retain positional routing
  capacity; ``True`` or ``False`` explicitly overrides that default. Users
  never supply target content or wire Q/K/V; the architecture derives routing
  from the declared geometry.
- ``reduction=None`` performs no pooling. It preserves one routed slot for each
  encoded child field-coordinate slot, in deterministic concatenation order,
  plus the corresponding presence mask. Branch attention may contextualize the
  payload first; pass-through does not preserve raw values, attach semantic
  provenance, fuse sibling fields, or perform a join.
- ``Attention(n_outputs=...)`` produces a fixed number of anonymous
  learned-query summaries augmented with masked first-moment and count lanes.
  It is a compression only when its output width is narrower than its encoded
  input, and output count does not assign item, group, rank, or identity
  semantics. The additive lane improves learned sum algebra but is not an exact
  raw-value reducer. ``Attention(position=False)`` disables rotary position in
  this learned-query reduction only; strict unordered controls must also set
  branch ``attention=None`` so an earlier branch-attention layer does not
  encode order.
- ``Mean()`` produces one presence-masked arithmetic mean of encoded tokens,
  not raw Number values. It does not promise a domain reduction merely by name.
- Number embeddings retain both Fourier features and a monotone standardized
  scalar lane so additive and multiplicative paths can extrapolate scale.
- Branch attention remains rotary. Reducer and decoder positional controls do
  not disable position in that earlier layer. Scalar decoder retrieval is
  position-free by default; repeated decoders use coordinate position by
  default. Every unordered claim still needs a metamorphic permutation gate.

These behaviors are implemented architectural contracts exercised by the
current proof suite. Their numerical thresholds remain provisional until each
family's multi-seed promotion matrix passes.

## Objective

Protect the modeling claims that distinguish relflow from a schema parser or a
collection of datatype codecs. A proof trains a model on a controlled synthetic
process and verifies that held-out predictions recover a relationship the public
schema makes learnable.

The suite should answer four questions:

1. Can relflow recover typed relationships from flat and nested records?
2. Does the schema's structure create the intended inductive behavior?
3. Does the model fail in predictable ways when information or training signal
   is absent?
4. Can bounded branch representations perform conditional reductions and
   entity-keyed information transfer without preprocessing the answer?

These are empirical regression proofs, not mathematical proofs and not claims
that one architecture or optimizer solves every dataset.

## The Inference Boundary

relflow should infer a relationship when all of the following are true:

- the relevant values survive preprocessing, query selection, and branch
  overflow;
- the values meet in the same branch context or in the target decoder's
  ancestor heritage;
- their tensorfield represents the meaning needed by the relationship;
- the training objectives provide gradient to the relevant modules; and
- the training distribution identifies the relationship and the held-out split
  tests the intended kind of generalization.

Within that boundary, the public package should support these claims:

| Capability | Modeling claim |
| --- | --- |
| Typed scalar context | Continuous, Boolean, and bounded categorical inputs can jointly predict typed targets. |
| Value state | `valued`, `null`, `padded`, and `masked` are distinct signals; a real zero need not collapse into missingness. |
| Repeated branches | Aligned child fields can interact at an item coordinate and repeated items can be summarized for an ancestor target. |
| Conditional aggregation | A visible operation and optional item category can select a reduction over repeated numeric values. |
| Weighted and distributional aggregation | Same-coordinate numeric fields can form learned contributions, moments, and paired statistics before branch reduction. |
| Ordered branches | Retained order can carry predictive signal; histories with the same members in a different order need not be equivalent. |
| Nested branches | Local relationships can be learned before their configured reduction routes a representation to a parent context. |
| Entity-keyed transfer | A target item can retrieve information attached to the same logical entity in another branch, even when branch order differs. |
| Set | Unordered membership and co-occurrence can predict outcomes, invariant to member order and duplicates. |
| Category | Repeated labels in a bounded learned vocabulary can acquire persistent label-specific behavior. |
| Hash | Equality can generalize to identifiers never seen during training while equal tokens remain in one encoded context. |
| DateParts | Recurring calendar phase can generalize across dates without treating absolute timestamp magnitude as the signal. |
| DatePart composition | Multiple visible calendar coordinates can jointly identify periodic behavior that no one coordinate determines. |
| Vector | Geometry supplied by a fixed-width continuous representation can drive prediction and reconstruction. |
| Text | A frozen text encoder can provide semantic features; relflow learns around those features but does not generate text. |
| Masked reconstruction | Correlated visible context can reconstruct selected state and content without a separate learning path. |
| Contextual embeddings | A reconstruction-trained root or branch can expose a useful normalized representation of the synthetic latent process. |
| Cluster | Repeated high-cardinality labels can organize into latent groups when the cluster head receives an engaged objective. |

The suite must also protect the limits of those claims. relflow should not be
expected to:

- infer causality from predictive association;
- invent deterministic source shaping—a filter, join, sort, time cutoff, or
  derived value—that should have been made explicit in a preprocessor. This is
  distinct from learned target-conditioned selection among visible, colocated
  inputs, which category-conditioned reduction deliberately tests. A learned
  aggregation proof is also not a substitute for exact accounting when the
  deterministic answer is already known;
- assume that the current sibling routing and fixed decoder queries implement
  arbitrary entity-level mappings; sibling entity transfer makes that a
  desired capability and distinguishes it from same-branch associative recall;
- recover records removed by branch overflow;
- assign semantic behavior to a `Category` label absent from its training
  vocabulary;
- invert a `Hash`, treat its training representation as persistent identity, or
  preserve equality after the relevant tokens no longer share a context;
- learn adaptive cluster capacity from a `Cluster` used only as a plain input;
- infer elapsed time or recency from calendar parts that intentionally omit
  absolute time;
- obtain a semantic embedding merely because `embed=True` was set without a
  useful training objective;
- generate source text from the `Text` reconstruction objective; or
- promise calibrated confidence, out-of-distribution detection, or causal field
  importance without a separate evaluation designed for that claim.

Every positive proof should have a matched negative control for at least one of
these boundaries.

## What Counts As A Proof

Every proof must make the following reviewable in its annotated Python script,
which is also the source of its documentation page:

1. **Claim** — one sentence describing the behavior protected.
2. **Synthetic process** — the latent variables and equations used to produce
   visible fields and targets.
3. **Observability** — why the configured schema contains enough information to
   identify the target.
4. **Split unit** — row, entity, sequence, or time, chosen to rule out the
   memorization mode the proof is meant to exclude.
5. **Baselines and controls** — a constant baseline plus a shuffled, omitted,
   incorrectly typed, or incorrectly structured comparison.
6. **Training budget** — fixed architecture, optimizer, batch size, and either
   maximum optimizer steps or a fixed number of complete data passes.
7. **Primary metric and gate** — measured on held-out predictions in the target's
   natural unit.
8. **Diagnostics** — enough recorded state to distinguish no
   learning, overfitting, boundary saturation, and seed instability.
9. **Example** — one nested YAML record per example in a Markdown cell;
   explain its target and add a matched control when it clarifies the
   information boundary or causal gate.

Each standalone script contains its complete generator, schema, training, and
evaluation, interleaved with Markdown explanations, a Typst tree, and YAML
examples. `proofs/results.yaml` owns historical reports and new measurements,
indexed by stable proof IDs. Quarto renders the script and recorded evidence
without executing the experiment. Only reporting and command-line handling
are shared between experiments.

A falling training loss is not proof of inference. The primary measurement must
use held-out data or, for a deliberately mechanistic proof, explicitly say that
it tests optimization state rather than generalization.

## Common Experimental Protocol

### Data

- Yield nested Python records from a locally seeded `numpy.random.Generator`;
  do not download or persist a dataset.
- Give train, validation, and test generation independent random streams.
- Prefer at least 4,096 training observations and 2,048 test observations for
  scalar tasks. Use fewer only when one observation contains a large repeated
  context.
- Balance classification targets unless imbalance is the subject of the proof.
- Add irreducible noise where a perfectly deterministic target would make the
  test unrealistically easy, but keep the Bayes limit known.
- Never expose the synthetic latent variable as a model field. It may be kept in
  a separate evaluation table keyed by `Batch` identity.
- Split by entity for identity claims, by complete sequence for history claims,
  and by time for temporal claims. A random row split is acceptable only when
  rows are the independent generative unit.

### Training

- Exercise `rf.Model`, built-in public tensorfield constructors, and
  `rf.SyntheticDataModule` through a normal Lightning training loop.
- Use `lit.seed_everything(seed, workers=True)` and
  `Trainer(deterministic=True)`.
- Prefer fixed optimizer steps. Fixed epochs are acceptable when complete data
  passes are part of the protocol; report the data size and epoch count so the
  effective update budget cannot change silently.
- Disable loggers, progress bars, summaries, and checkpointing unless they are
  part of the claim.
- Run on CPU by default. An accelerator-specific proof must record its device and environment
  and must not replace the CPU behavioral gate.
- Compare alternative schemas or configurations with the same observations,
  optimizer family, parameter budget where practical, and number of updates.
- Do not tune a proof on its test split. Select a checkpoint or threshold from
  validation data, then evaluate the test split once.

### Seeds and stability

Use a panel of three paired data/model seeds for a mature core proof. A core
claim passes when:

- the median result clears the primary gate with the declared safety margin;
- every seed beats the naive baseline; and
- no seed exhibits a declared catastrophic failure such as NaN, chance-level
  performance, or convergence to a forbidden capacity boundary.

An extended stability run should use at least ten seeds before a new threshold
is frozen. A single-seed proof may be introduced while a hypothesis is being
calibrated, but its interpretation must label it as a mechanistic or provisional
check.

### Metrics

Prefer metrics with interpretable baselines:

| Target or representation | Primary metric |
| --- | --- |
| Balanced Boolean | ROC AUC and balanced accuracy |
| Category | Accuracy or macro accuracy; top-k only when candidates are part of the claim |
| Number | Normalized RMSE: model RMSE divided by train-mean baseline RMSE |
| Set | Macro F1 over populated labels plus exact-set accuracy when appropriate |
| Value state | Balanced state accuracy, reported separately from content |
| Cluster partition | Adjusted Rand index or another permutation-invariant partition score |
| Embedding | Held-out nearest-centroid or k-nearest-neighbor accuracy using labels only in the evaluator |
| Invariance | Maximum prediction delta and embedding cosine similarity over equivalent inputs |

Implement small metrics in NumPy or Torch rather than adding a large evaluation
dependency. Always report absolute performance and the matched baseline. A
relative improvement alone can make two bad models look successful.

### Acceptance thresholds

The numeric gates below are provisional starting points. For each new proof:

1. run at least ten seeds for the intended model and its negative control;
2. inspect the result distributions without changing the generator;
3. place the gate in the gap between them with room for numerical variation;
4. record the raw summary and rationale in `proofs/results.yaml`; and
5. treat a later threshold change as a modeling-contract change requiring an
   explanation, not routine test maintenance.

Avoid exact loss values, exact learned parameters, and a requirement that every
epoch improve. Assert a terminal window or best validation checkpoint when
optimization trajectories are expected to oscillate.

## Core Proof Matrix

### Proof-harness calibration

**Claim:** the harness detects real signal without reporting leakage as model
skill.

Generate a balanced Boolean target independently from several Number, Category,
and Set inputs. A no-signal model must obtain test AUC between `0.42` and `0.58`
on at least 4,096 test rows. Add a positive-control field equal to the target;
the otherwise identical model must obtain AUC of at least `0.98`.

This proof should land before broadening the suite. Every later generator can
reuse its split and baseline checks, but not necessarily a shared abstraction.

### Mixed flat supervision

**Claim:** typed scalar fields can jointly recover a nonlinear conditional
relationship.

Generate:

```text
x ~ Uniform(-2, 2)
segment ~ {a, b, c, d}
flag ~ Bernoulli(0.5)
y = 1.4*x + 0.6*sin(2*x) + segment_effect[segment] + 1.1*flag + epsilon
converted = y > 0
```

Use visible `Number`, `Category`, and `Boolean` fields with masked `Number` and
`Boolean` targets. Hold out rows while retaining the training segments. Require
normalized RMSE at most `0.35` and Boolean AUC at least `0.95`. Independently
permuting both targets must remove the advantage over their constant baselines.

### Missingness is not zero

**Claim:** value state is predictive independently from content.

Create a Number field that is exactly `0.0` in half the rows and `null` in the
other half. The masked Boolean target says whether the source was null. No other
field carries signal. Require held-out accuracy of at least `0.98`. Both a
matched model with target-independent validity and the trained signal model
evaluated after every null is prefilled with zero must be at chance.

This protects the learned state embedding and guards against a refactor that
asks neutral numeric content to represent both real zero and absence.

### Item-aligned reconstruction

**Claim:** a repeated target is reconstructed at the coordinate of its sibling
item fields.

Generate variable-length orders. Each item contains visible `quantity` and
`unit_price` plus a masked `subtotal = quantity * unit_price + epsilon`. Split by
order, include empty and partially padded orders, and omit `subtotal` from the
prediction request while retaining the item structs.

Require normalized RMSE at most `0.25` on valued retained items and correct
prediction geometry for every item and padded slot. A control that independently
shuffles subtotals among items must lose most of the predictive skill.

### Order-dependent history

**Claim:** a `Branch` can distinguish histories with identical members in a
different order.

Place one `A`, one `B`, and random distractor events in each variable-length
history. The target is whether `A` occurs before `B`. Construct paired positive
and negative records with identical multisets. Split by complete history and
require test accuracy of at least `0.95`.

Two controls are required:

- randomly permuting test event order must reduce accuracy near chance; and
- representing the same labels as an unordered `Set` must not solve the task.

### Nested context adds information

**Claim:** local branch structure preserves a relationship destroyed by
flattening.

Generate customers with several sessions and the same global multiset of event
types. In positive records, marker events `A` and `B` occur in the same session;
in negative records they occur in different sessions. A nested
`sessions -> events` schema must reach accuracy of at least `0.90`. A matched
model given only one flattened event collection should remain below `0.65`.

This proves the value of the schema hierarchy, not merely the capacity of a
larger model.

### Unordered Set interaction

**Claim:** Set membership and co-occurrence are learnable and invariant to order
and duplicates.

Generate signal labels `a` and `b` plus random nuisance labels. Define the target
as `a XOR b`, then randomize member order and insert duplicates. Require held-out
accuracy of at least `0.95`.

For the same trained model, canonical, permuted, and duplicate-augmented forms
of one logical set must produce equal probabilities within `1e-6`. Removing one
signal member should change the appropriate predictions by a material margin.

### Unseen identity equality with Hash

**Claim:** Hash permits equality reasoning for identities absent from training.

Generate balanced pairs `(left_id, right_id)` with a Boolean equality target.
Use disjoint identifier universes for train, validation, and test. Both IDs must
remain as separate `Hash` leaves in the same root context. Require test AUC of at
least `0.95`.

A matched `Category` model must be unable to solve the disjoint-ID test because
both values are unavailable vocabulary content; its AUC should remain below
`0.65`. Keep both Hash tokens in the same root context for the Hash equality
proof. Sibling entity transfer separately tests whether entity correspondence
survives sibling routing and reaches a
key-conditioned decoder; failure there must not be misdiagnosed as failure of
Hash equality itself.

### Persistent Category identity and its OOV boundary

**Claim:** Category learns behavior for repeated known labels but does not invent
behavior for unseen labels.

Assign each of 64 entity labels a stable binary propensity. Use separate rows
for training and known-entity testing, then a second test set containing only new
entities with independently assigned propensities. The known-entity test should
reach AUC of at least `0.95`; the new-entity test should stay below `0.65`.

This is one paired proof. The negative result is part of the public datatype
contract and prevents accidental claims that categorical strings carry lexical
semantics.

### Calendar recurrence with DateParts

**Claim:** cyclical calendar coordinates generalize a recurring phase rule.

Generate timestamps across many dates. Define a target from an hour-of-day arc
that crosses midnight. Train on even hours and test on held-out odd hours and
new dates. A `DateParts(dateparts=["hour_of_day"])` model should reach balanced
accuracy of at least `0.85`.

Use an hour label represented as a vocabulary-backed Category for the matched
OOV control. Also generate a target based only on absolute year; the configured
DateParts field must not beat chance because year was not represented.

### Cross-DatePart periodic inference

**Claim:** one periodic calendar coordinate can identify another only to the
extent that the mapping is present and stable in the visible calendar context.

Train across complete years with a visible
`DateParts(dateparts=["day_of_year"])` and predict a masked month-of-year
Category on held-out years. Measure macro accuracy overall and separately near
month and leap-day boundaries. This is not a vocabulary lookup: the continuous
cyclic representation must learn the piecewise calendar partition and transfer
it to dates absent from training.

Pair the positive case with two identifiability boundaries:

- day of year does not determine weekday across years unless year or an
  equivalent anchor is visible; and
- after February, leap and non-leap years can assign the same day-of-year phase
  to different calendar dates, so exact boundary behavior requires leap/year
  context.

The positive month proof must beat a month-frequency baseline materially on
held-out years. A weekday-only target must remain near its seven-class baseline,
and ambiguous leap-boundary rows must be reported rather than hidden in an
overall score. Additional rungs should cover hour-to-shift, week-to-season,
day-of-week plus hour-to-business-window, and combinations where neither
DatePart is sufficient alone but their pair is.

DateParts should ultimately be covered along four separate axes:

| Axis | Examples | Main boundary |
| --- | --- | --- |
| Within-cycle mapping | day-of-year to month; hour to shift | phase boundaries and discontinuities |
| Coordinate composition | weekday plus hour to business window | neither coordinate is sufficient alone |
| Cross-year transfer | seasonal target on held-out years | no absolute-year memorization |
| Calendar ambiguity | day 60, week 53, daylight-saving transitions | omitted year, calendar, or timezone context |

Also test cyclic continuity directly: December/January and 23:00/00:00 should
be close for periodic targets, while a target with a genuine boundary may
separate them. Timezone conversion, locale-specific weeks, fiscal calendars,
holidays, and elapsed duration belong in explicit preprocessing unless their
required context is represented in the schema.

Implementation snapshot: held-out month accuracy from day of year is 0.925,
versus 0.101 after timestamp permutation. Day of year alone reaches the
identifiability limits for weekday (0.143) and the ambiguous leap boundary
(0.500); adding the required day-of-week or week-of-month coordinate reaches
1.000. A balanced business-window target reaches only 0.500 from weekday and
0.750 from hour alone, but 1.000 when both coordinates are visible.

### Supplied vector geometry

**Claim:** Vector preserves useful geometry supplied by an upstream system.

Draw latent classes, assign well-separated prototypes in eight dimensions, and
add isotropic noise. Use a visible `Vector(n_dim=8)` and a masked Category target.
Hold out independently sampled vectors from the same prototypes. Require macro
accuracy of at least `0.95`; permuted prototype labels must remove held-out
skill.

Do not use this proof to claim automatic feature scaling. A separate rescaling
experiment may characterize preprocessing sensitivity but is not a Vector
contract.

### Contextual masked reconstruction

**Claim:** the unified masking path learns cross-field structure without an
external outcome label.

Draw an unobserved latent class that generates three noisy views: a Category, a
Set, and a Vector. Configure sampled reconstructing masks on all three visible
fields. The latent class is retained only by the evaluator.

On stable validation masks, every datatype's content metric must beat its
marginal predictor, and no field may improve only in state accuracy. Independently
permuting the three views across observations must remove the reconstruction
advantage. Compare learned-mask and structural-skip variants under the same
selection stream; both must learn, although their exact scores need not match.

### Contextual embedding utility

**Claim:** a reconstruction-trained root embedding represents the latent process
well enough for held-out retrieval.

Reuse contextual masked reconstruction, set `embed=True` on the root, and
train without exposing the latent
class. Fit nearest centroids from training embeddings using latent labels only
inside the evaluator. Require held-out nearest-centroid accuracy of at least
`0.85` and materially better separation than an independently permuted-view
control.

Also verify that returned embeddings are finite and unit normalized. Do not
compare coordinates across separately initialized models or checkpoints; the
public contract does not align those spaces.

### Adaptive cluster discovery

**Claim:** repeated high-cardinality identities with shared behavior recover a
latent partition and a committed count near the true K.

Use two complementary processes:

1. repeated merchant IDs whose supervised labels are determined by one of K
   latent groups; and
2. repeated IDs assigned to one of K hidden functions `y = f_k(x)` without an
   explicit group label.

The train and test splits should contain different observations of the same IDs.
Require, across the final five epochs or the restored best checkpoint:

- committed K within `[true_K - 1, true_K + 2]` and at neither configured
  boundary;
- usage perplexity within `1.5` of true K;
- permutation-invariant assignment ARI of at least `0.80`; and
- downstream predictive skill materially above a marginal baseline.

Required controls are a plain-input Cluster whose head remains dormant, unique
IDs that cannot pool repeated evidence, and a no-regime generator. The current
tests under [`cluster/cluster_convergence/`](cluster/cluster_convergence/)
establish committed-count behavior and the dormant control with one seed;
partition recovery, held-out predictions, and the seed panel remain to be
added before this proof is complete.

### Signal-field reliance

**Claim:** a model trained to tolerate structural skipping relies more on true
signal fields than on nuisance fields.

Generate two informative fields, two redundant fields, and at least eight
independent nuisance fields. Train candidate inputs with sampled structural skip,
clear those masks for evaluation, then measure held-out loss while deactivating
one field at a time.

Each informative-field ablation must increase primary-target loss by at least
three times the largest nuisance-field increase. The mean nuisance delta must be
within five percent of baseline loss. Report correlated redundant fields as a
group rather than interpreting a small individual delta as irrelevance.

This is a predictive reliance proof, not a causal attribution claim.

## Complex Relational Proofs

The following proofs deliberately exceed ordinary supervised prediction. They
ask whether the schema-generated model can execute a bounded, learned relational
program: select a set of item tokens, combine their numeric content, or move a
value between tokens that share an identity.

They form a dependency ladder. Do not interpret a failure in entity-keyed group
aggregation until unconditional reduction, conditional filtering, and simple
entity copying have been calibrated independently.

### Operation-conditioned numeric reduction

**Claim:** one branch representation can answer a visible reduction request over
its numeric children.

Generate a base bag of between 4 and 32 finite values, then define:

```text
operation ~ {sum, mean, min, max}
answer = reduce(operation, items[*].value)
```

Represent `operation` as a visible root Category, `items` as a Branch containing
one visible Number, and `answer` as a masked root Number. Values, lengths, and
presentation order are random. Values should come from a bounded asymmetric
distribution so all four operations usually produce distinct answers.

For every generated bag, create all four operation variants and keep the whole
four-row family in one split. This paired construction prevents the model from
using a correlation between bag statistics and the requested operation. Split
by base bag, never by expanded query row.

Evaluate fixed length first, then variable length:

| Stage | Length | Purpose |
| --- | --- | --- |
| A | exactly 8 | Establish operator selection without a cardinality variable. |
| B | 4 through 16 | Determine whether the branch preserves enough cardinality for sum as well as mean. |
| C | 4 through 32 | Characterize the useful bounded range and degradation near capacity. |

Report normalized RMSE separately for every operation. Provisional gates are
`<= 0.20` for every operation at fixed length and `<= 0.30` at variable lengths
through 16. No aggregate-only score may hide one failed operator.

Required controls and diagnostics:

- hiding `operation` creates contradictory targets for identical bags and must
  destroy per-operation skill;
- random reordering of a bag should change a prediction by no more than `0.05`
  target standard deviations after training on randomized order;
- report error by operation and length, especially sum error versus length; and
- if variable-length sum fails while mean succeeds, rerun with an explicit
  visible item count. Passing only with count identifies a cardinality-loss
  boundary in pooling rather than a generic optimization failure.

The proof does not require exact arithmetic. It requires uniformly useful
approximation inside the declared value and length range, with clear behavior
outside it.

Implementation snapshot: stage A passes for all four operations, including a
hidden-operation control and a fresh test-time item permutation. Variable-length
stages B and C remain unimplemented.

### Cardinality and length generalization

**Claim:** a reduction route that predicts a sum must retain total mass across
the supported length range and must not confuse configured capacity with learned
extrapolation.

Train variable-length bags only at lengths one through six while configuring
capacity twelve. Evaluate independent in-range bags, exact whole-bag
duplication, equal-valued bags of different lengths, and an out-of-range split
at lengths seven through ten. Mean is the matched positive for averages and
negative for sums: complete-bag duplication must leave its representation and
average prediction unchanged to numerical precision.

Use three sum rungs:

1. learned Attention with structural presence but no explicit count;
2. Mean plus an ordinary visible `item_count` Number as an arithmetic control;
3. a future semantic sum/count reducer whose cardinality behavior is explicit.

The visible count is a useful present-day diagnostic, not the desired default
schema. It separates “the decoder cannot multiply value by count” from “the
reduction erased count.” Both learned routes must be graded in range before an
unseen-length gate is interpreted. A branch length is only an overflow/shape
bound; it is not a promise of behavioral support at lengths absent from
training.

Metamorphic gates require item-permutation invariance, duplication-invariant
mean, duplication-equivariant sum, and linear response to repeating one equal
value `k` times. Empty, null, overflow, and nested cardinality each require a
separate declared semantic rather than sharing this gate.

Implementation snapshot: `Mean()` reaches 0.0373 nRMSE with permutation and
duplication drift below `2.3e-7`. Attention's internal additive mass/count lane
now clears the in-range, unseen-length, duplication, and equal-value gates.
Mean plus a visible count also clears its unseen-length and duplication gates
after Number gained a monotone scalar lane and target queries gained aligned
visible-sibling context. Random-split accuracy alone remains insufficient; the
algebraic interventions are the contract.

### Weighted numeric aggregation

**Claim:** a branch can preserve item alignment through a nonlinear item-wise
transform and then reduce the transformed values.

Generate independent value and weight fields at every item coordinate:

```text
items[*] = {value, weight, optional_group}
weighted_sum = sum(value * weight)
weighted_mean = sum(value * weight) / sum(weight)
```

Use signed asymmetric values and strictly positive non-constant weights. Vary
both fields independently so neither `sum(value)`, `sum(weight)`, the unweighted
mean, nor item count can proxy the answer. Keep whole collections in one split
and randomize item order. Start with a fixed length before testing variable
cardinality and unseen length extrapolation.

Use a ladder that localizes the failed primitive:

1. **Precomputed-contribution control:** expose `contribution=value*weight` at
   each item and predict its sum. This tests only branch reduction.
2. **Learned product:** expose only value and weight, then predict weighted sum.
   The gap from the control measures aligned multiplication plus reduction.
3. **Normalized weighted mean:** expose the same inputs and predict the ratio of
   two sums. Report errors by total weight and cardinality.
4. **Conditional weighted aggregation:** add item groups and a visible selected
   group. Items outside the selected group must not change the answer.

Required metamorphic controls jointly permute complete items, permute weights
independently of values, scale every weight, and duplicate every item. Joint
item permutation must leave both targets unchanged. Independent weight
permutation must change the answer and destroy skill against the original
target. Scaling all weights by positive `a` must scale weighted sum by `a` but
leave weighted mean unchanged. Duplicating all items must double weighted sum
and leave weighted mean unchanged.

This proof should not ask users to precompute the contribution as the final
recipe. That field is a diagnostic control. The intended public behavior is
automatic same-coordinate feature interaction followed by an explicitly chosen
reduction. If the control passes and learned product fails, document that as an
encoder/alignment boundary rather than a pooling failure.

### Grouped weighted composition

**Claim:** an interleaved item Category and root selected-group request can
compose with raw value/weight binding and summation without prefiltering or a
derived contribution field.

Emit every base bag once for each selected group. Rotate group labels, swap
weights only within groups, and jointly permute complete items as three separate
interventions. The first tests conditional routing, the second item-local
multiplication, and the third unordered semantics. A low held-out error is not
sufficient unless all three behave in the expected direction.

Implementation snapshot: the natural six-item raw schema reaches 0.0859 nRMSE;
rotated labels reach 1.4263, within-group weight swaps 0.6020, and joint-item
permutation drift 0.0674 target standard deviations. The supplied-contribution
diagnostic now also clears its accuracy, label-rotation, and permutation gates
after coordinate-local mixing was added. The engineered variant is not
automatically the preferred user recipe: it proves filtering and summation but
cannot prove learned multiplication. The promoted example must remain the
natural raw schema.

### Distribution moments and paired statistics

**Claim:** fixed-width summaries preserve more than a center when the requested
target depends on dispersion or paired variation.

Generate fixed-length bags first and independently request variance, standard
deviation, range, mean absolute deviation, and root-mean-square. Then attach two
aligned Numbers and request covariance, correlation, and least-squares slope.
Matched bags must share means while differing in spread, or share both marginal
distributions while differing only in pair alignment. Otherwise a model can
pass without learning the claimed statistic.

Controls expose precomputed `x**2`, `x*y`, or centered contributions one rung at
a time. These are diagnostic sufficient-statistic controls, not the recommended
user schema. Independently permuting `y` must destroy covariance and correlation
while leaving both marginals fixed. Positive affine rescaling has known
metamorphic effects and must be checked. After fixed length passes, vary
cardinality, skew, outliers, duplicated values, and nearly zero variance.

Implementation snapshot: raw fixed-length variance reaches 0.0739 nRMSE and
maps a matched-mean low/high-spread pair from true variances 0.04/1.21 to
0.0424/1.2075. Supplied squared deviations reach 0.0927. Supplied centered
cross-products reach 0.1542 for covariance. With `reduction=None` preserving
the aligned records and same-coordinate mixing preserving each pair, raw
`x`/`y` covariance reaches 0.0961 nRMSE and rises to 1.3684 after only `y` is
shuffled. Early learned compression remains the documented footgun because it
can erase the pairing before the sufficient statistic is formed.

### Numeric aggregation coverage matrix

Treat numeric aggregation as a lattice of independently reviewable proofs. A
green cell in one row never implies the rows below it:

| Family | Core target | Required confounder or control |
| --- | --- | --- |
| Cardinality-free | mean, min, max | duplicate all items; answers stay fixed |
| Cardinality-sensitive | count, sum | equal-valued bags with different lengths |
| Item transform | sum of squares, product sum | precomputed-contribution control |
| Weighted | weighted sum and mean | independently permuted weights |
| Conditional | group-filtered reductions | outside-group perturbation |
| Paired | covariance, correlation, slope | same marginals with shuffled pairing |
| Distributional | variance, quantile, histogram bin | matched mean and extrema |
| Robust | median, trimmed mean, winsorized mean | sparse extreme outliers |
| Hierarchical | sum of session sums, max session sum | same flattening, new grouping |
| Multi-output | one statistic per group or operation | every cell graded separately |
| Missing-data | skip-null versus null-propagating semantics | identical values, new validity |
| Empty-set | declared fallback, null, or error | zero selected items |
| Extrapolation | unseen length and magnitude | in-range result reported alongside |

Before promotion, every numeric proof must declare and test its empty, null,
overflow, and duplicate semantics. A provisional proof may instead identify
an untested case explicitly in its remaining work; silence is not a semantic
contract. These cases are part of the synthetic program, not cleanup details
to infer after a failure. For ratios, report denominator range and guard
near-zero cases. For all grouped tasks, perturb non-selected groups as a
metamorphic invariance test.

Implementation snapshot: at six items, the public schema learns raw weighted
sum below 0.30 nRMSE and a supplied-contribution control below 0.25. Permuting
weights independently removes skill, while joint item permutation and positive
weight scaling obey their expected transformations. Raw weighted mean passes
through variable lengths two to six and remains invariant to joint permutation,
uniform weight scaling, and duplicating all items. A matched `Mean()` route
learns contribution average but produces identical downstream sum predictions
for one and six equal contributions, making the count-erasure footgun explicit.

### Category-conditioned filtered and grouped reduction

**Claim:** a branch can select values by an associated item Category and reduce
only the selected subset.

Extend each operation-conditioned reduction item with:

```text
items[*].group in {A, B, C}
selected_group in {A, B, C}
operation in {sum, mean, min, max}
answer = reduce(operation, value where group == selected_group)
```

Ensure every group has at least two items so empty-set semantics do not enter
the first proof. Randomly interleave groups in the branch. For each base bag,
create all 12 `(selected_group, operation)` requests and keep them in one split.
The model sees `items.group` as Category, `items.value` as Number, and the two
request fields as root Categories.

Require normalized RMSE at most `0.30` for every operation and group through a
total branch length of 24. Report a `4 x 3` metric grid; an overall average is
diagnostic only.

Use these controls:

- independently permute group labels relative to values while retaining the
  original targets; filtered skill must disappear;
- hide `selected_group`; the paired requests become contradictory and must not
  pass;
- evaluate identical bags under every request to verify that predictions change
  in the direction required by the selected subset; and
- include a visible precomputed answer as a positive harness control, but never
  in the model used for the real claim.

After the single-answer form passes, add a group-by form. One visible operation
produces three masked root Number targets, `result_A`, `result_B`, and
`result_C`, from the same item branch. All three targets must pass their
per-group gates simultaneously. This distinguishes one conditional query from a
branch representation that retains several group summaries at once.

An extended compositional split may hold out selected `(group, operation)` pairs
while exposing each group and operation elsewhere. Record the result as
characterization until repeated runs establish that this kind of compositional
generalization is a relflow contract.

Implementation snapshot: selected-group mean passes for three interleaved
groups and fails after matched group-label permutation. The full 12-cell
group-by-operation program now passes every per-cell 0.30 nRMSE gate when item
records use `reduction=None` and the root uses twelve Attention outputs. The
gain comes from coordinate-local record interaction, data-conditioned decoder
queries, and additive evidence; output count alone still must not be treated as
conditional routing.

### Entity-keyed value transfer across sibling branches

**Claim:** a masked item value in one branch can retrieve the value attached to
the same logical entity in a sibling branch, independently of item order and
persistent entity memorization.

`Entity` here describes the synthetic concept. The current public package has
no `rf.Entity` tensorfield. Use `rf.Hash` for the identity leaves because the
proof requires equality for previously unseen IDs; use Category only in a
separate closed-vocabulary characterization.

Generate one observation as:

```text
source = [{entity_id: e_i, value: v_i} for i in entities]
target = [{entity_id: e_i, value: MASKED} for i in permutation(entities)]
target_value(e_i) = source_value(e_i) + epsilon
```

Draw `v_i` independently for every observation. An entity must not have a
stable global value. Use disjoint identifier universes across train, validation,
and test, and independently permute the two branch orders. These constraints
make entity matching the only usable signal.

The intended direct schema is:

```python
model = rf.Model(
    reduction=None,
    source=rf.Branch(
        length=4,
        reduction=None,
        entity_id=rf.Hash(n_hashes=4),
        value=rf.Number,
    ),
    target=rf.Branch(
        length=4,
        reduction=None,
        entity_id=rf.Hash(n_hashes=4),
        value=rf.Number(mask=True),
    ),
    # ordinary model width, layers, heads, batch size, and optimizer omitted
)
```

Start with two entities, then sweep branch length through 4, 8, and 16. Score
normalized RMSE over target coordinates and report performance by length.
Provisional acceptance is `<= 0.25` at length 4, with no seed above `0.40`.

Three comparisons localize failures:

1. **Direct sibling branches:** the desired public capability.
2. **Shared-context prerequisite:** a preprocessor stacks source and target
   items into one Branch and adds a `role` Category. Same-branch associative
   recall should reach normalized RMSE `<= 0.15` before the direct form is
   judged as a sibling-specific failure.
3. **Broken-identity control:** permute target IDs independently from their
   answers while leaving value distributions unchanged. Performance must return
   close to the train-mean baseline.

Also compare aligned and independently permuted branch order. Similar
performance is required; otherwise the model learned positional copying rather
than identity matching.

This proof is now an implemented provisional contract. `reduction=None`
retains sibling candidate tokens through the root, visible fields at each
target coordinate condition its learned decoder query, and decoder attention
matches content without rotary position. The broken-key control distinguishes
that behavior from generic value transport.

The implemented design follows the first two requirements below; the remaining
alternatives should still be evaluated against the proof rather than assumed:

- expose sibling token memory and structural provenance to the decoder;
- derive target-conditioned retrieval internally from the visible target key
  and aligned source key/value coordinates;
- preserve several learned branch latents or every encoded token; or
- evaluate an explicit `Reference` or shared-context preprocessing contract as
  a separate, more demanding user-API alternative rather than silently making
  it the common solution.

Any new design must make identity matching auditable, preserve branch geometry
and `Batch` identity, and remain late-extensible across tensorfields. Q/K/V used
inside a prototype describes implementation mechanics, not fields the ordinary
user should have to wire.

Implementation snapshot: the two-entity shuffled same-branch `Hash` recall and
both sibling Category/Hash transfer routes pass their accuracy, broken-key, and
gap gates. Persistent Category also succeeds through fixed-width compression
at 0.0198 nRMSE in the calibrated run, while preserved Category reaches 0.0046;
fresh Hash identities still use pass-through as the safe recipe. These are
learned routing results, not an exact join guarantee.

### Entity-keyed group reduction across branches

**Claim:** a target entity in one branch can request an aggregate over all
matching source items in another branch.

This composes operation-conditioned reduction, category-conditioned filtering,
and sibling entity transfer:

```text
source[*] = {entity_id, value}
requests[*] = {entity_id, operation, answer=MASKED}
answer(e, op) = reduce(op, source.value where source.entity_id == e)
```

Generate 2 to 8 entities per observation and 2 to 6 source items per entity.
Shuffle source items and request entities independently. Draw unseen Hash IDs,
item values, and requested operations for each observation. Keep entity
frequency balanced and require every requested entity to have at least two
source items in the core proof.

Entity-keyed group reduction begins only after fixed/variable reduction,
category filtering, and a length-4 sibling copying pass. Use the same three-way
comparison as sibling entity transfer: direct sibling branches, one stacked
shared context, and broken identity. Report normalized RMSE for every operation,
source multiplicity, entity count, and branch length.

The initial gate is normalized RMSE `<= 0.35` for mean, min, and max with at most
four entities. Sum receives a separate cardinality diagnostic and the same
explicit-count control as operation-conditioned reduction. Expand the
acceptance range only after the error surface is measured across seeds.

Extended variants should add unmatched entities, repeated requests for one
entity, and noisy or conflicting source values. Each variant must define its
semantics—null prediction, marginal fallback, mean, latest, or error—before it
becomes a proof. The model must not be graded against an unstated join policy.

## Further Complex Interaction Atlas

The next candidates cover distinct forms of relational computation rather than
only increasing the size of the reduction-to-transfer ladder. Each should
first be implemented at the
smallest bounded size that identifies the operation, then swept until the useful
capacity boundary is visible.

Some candidates deliberately hide an exactly computable operation in order to
test learned interaction. That is a benchmark constraint, not production
advice: known contractual arithmetic, arg-selection, joins, set operations,
calendar conversion, and graph traversal belong in a preprocessor or the
application. A learned version is appropriate only when approximation itself
is the capability under study or the real target is noisy and predictive.

Use these expectation labels:

- **native candidate** — all required tokens meet in one current branch or
  decoder heritage, so a calibrated proof should become a core contract;
- **bottleneck probe** — the information exists but current pooling may erase
  the correspondence needed by the target; and
- **relational objective** — the task likely motivates a new schema/routing
  primitive if the matched shared-context control succeeds.

### A grammar for generating interaction hypotheses

The examples can be generalized as a small synthetic relational language:

```text
answer = emit(
    reduce(
        transform(
            select(context, query)
        )
    ),
    destination,
)
```

Vary one axis at a time before sampling full programs:

| Axis | Candidate operations |
| --- | --- |
| Context | Flat fields, one branch, nested branches, sibling branches, or several views. |
| Select | All items, Category equality, numeric predicate, Hash-key equality, time window, nearest vector, or graph path. |
| Transform | Identity, sign by role, difference, product of aligned fields, elapsed time, or peer-relative deviation. |
| Reduce | Any, all, count, distinct count, sum, mean, min, max, quantile, arg-extreme, or weighted average. |
| Destination | One root target, one result per group, one result per item, a sibling target item, or an exported embedding. |
| Generalize | New rows, new values, unseen IDs, new order, longer cardinality, held-out operation combinations, or extra distractors. |

The generator may randomly select programs only after each selected primitive
has an independent positive proof. Every operation token and argument needed to
identify the answer must be visible to the model. The evaluator executes the
same program outside relflow to produce ground truth. Keep all requests derived
from one base context in the same split.

### Learned quantifiers and conditional counts

**Expectation:** native candidate.

Generate items with a group Category, a Number, and a Boolean property. Provide
a visible query containing a selected group, threshold, and quantifier:

```text
quantifier in {any, all, none, at_least_two}
predicate(item) = item.group == selected_group and item.value > threshold
answer = quantifier(predicate(items))
```

Use a masked Boolean answer and, in a companion task, a masked Number count.
Build paired query variants over the same base collection. Balance cases close
to each decision boundary so majority heuristics cannot pass. Report accuracy
by quantifier and collection length. The count target diagnoses whether a
Boolean failure is logical selection or lost cardinality.

### Arg-extreme attribute retrieval

**Expectation:** native candidate.

Each item contains `score`, `payload`, and an optional group. The request names
`argmax` or `argmin` and optionally a selected group. The masked answer is the
payload attached to the winning item, not the extreme score itself:

```text
winner = argmax(item.score where item.group == selected_group)
answer = winner.payload
```

Use Number payloads first, then bounded Category payloads. Make scores unique,
randomize item order, and split by complete collection. Shuffling payloads
relative to scores while preserving both marginals must destroy skill. This
tests item alignment, conditional selection, and retrieval in one branch.

Implementation snapshot: at branch length three, full pass-through reaches
about 0.16 nRMSE. A one-output child followed by three root Attention outputs
reaches about 0.30. Rotating payloads relative to scores must remove skill in
both routes. The three learned outputs jointly retain bounded candidate
evidence; they do not promise that output slot *i* represents candidate *i*.

### Order statistics and learned ranking

**Expectation:** native candidate with a strong pooling-capacity diagnostic.

Provide a requested rank or quantile and predict the corresponding numeric
value from one branch:

```text
rank in {minimum, q25, median, q75, maximum}
answer = order_statistic(rank, items[*].value)
```

Operation-conditioned min/max are the endpoints; median and interior quantiles
require a summary of the distribution rather than one salient token. Report error by rank,
length, duplicates, and skew. A histogram or several learned branch latents may
be necessary if single-vector pooling cannot retain the distribution.

Implementation snapshot: one pass-through length-nine branch answers five
visible rank requests with 0.0823 overall nRMSE after the representation
changes. Per-rank nRMSE is 0.0423, 0.0975, 0.0882, 0.1243, and 0.0550 from
minimum through maximum. Duplicate, symmetric, and skewed shape cells all pass;
cycling rank labels worsens nRMSE to 1.438, hiding rank collapses each bag's
five predictions to a `2.384e-7` spread, and complete-item permutation drift is
0.0367 target standard deviations. The hidden-request gate requires both
endpoints and at least three of five rank cells to exceed 0.65 nRMSE; central
unconditional estimates need not all be worse than that arbitrary threshold.

### Same-branch associative recall

**Expectation:** native candidate and required positive control for sibling
entity transfer.

Stack source and query records into one branch. Each source record has a Hash
key and visible random value; each query record repeats a key and has a masked
value. A role Category distinguishes source from query. Keys are unseen across
splits and positions are independently shuffled.

Require each query coordinate to recover its paired source value. Use two,
four, eight, and sixteen key-value pairs. Broken keys must remove skill, and a
joint permutation of records must permute outputs without changing matched
answers. This directly measures local attention as content-addressable memory.

Implementation snapshot: the two-pair shuffled-key form now passes its 0.25
nRMSE recall gate, rises above 0.80 after keys are broken, and preserves at
least a 0.50 causal gap. Its matched fixed-position compressed control remains
useful because it passes even when keys are broken, confirming positional
reconstruction rather than associative recall.

### Peer-relative per-item inference

**Expectation:** native candidate.

Ask every item to compare itself with peers rather than only predicting one root
target. Useful processes include:

```text
above_group_mean_i = value_i > mean(value_j where group_j == group_i)
deviation_i = value_i - mean(value_j where group_j == group_i)
is_outlier_i = abs(deviation_i) > threshold
```

Mask the Boolean or Number answer at each item coordinate. Keep all groups
represented and randomize their interleaving. Require permutation-equivariant
predictions: reordering complete items must reorder outputs the same way.
Shuffling group labels independently from values is the negative control.

This tests whether branch interaction can compute a group statistic and route
it back to each member, rather than only compressing the collection for an
ancestor target.

Implementation snapshot: with item and root `reduction=None`, raw ungrouped
deviation reaches 0.091 nRMSE and grouped deviation reaches 0.054. Independent
group translation remains accurate at 0.358, corrupting group labels worsens
the grouped result to 1.424, and complete-item permutation drift is 0.056. A
supplied peer-mean diagnostic reaches 0.030. A one-output learned item summary
also reaches 0.074 because each repeated target retains its own aligned visible
siblings as query context; corrupting group labels worsens that route to 1.561.
This is evidence for shared-context compression, not lossless one-vector
encoding of an arbitrary collection.

### Temporal interpolation and local smoothing

**Expectation:** native candidate.

Generate irregularly sampled trajectories from a mixture of linear trends,
sinusoids, and smooth random components. Each event contains an explicit elapsed
time Number and a value. Mask one or more interior values with a query-backed
policy and reconstruct them from surrounding events.

Score by gap width, distance to the nearest visible neighbors, and whether one
or both sides are visible. Shuffling timestamps while retaining values must
destroy interpolation skill. Extrapolation beyond the last event is a separate
forecasting proof and should not share the interpolation gate.

### Sequence grammar and motif detection

**Expectation:** native candidate.

Move beyond the two-marker order-dependent history check. Generate event
sequences from small hidden state machines with identical unigram and bigram frequencies but
different higher-order motifs. Ask for the next event, the hidden regime, or
whether a motif occurred within a bounded window.

Controls should preserve event counts while destroying the relevant order—for
example, block permutation or reversal. This distinguishes genuine sequence
composition from bag-of-events prediction and measures how much order survives
branch reduction.

### Hierarchical sufficient statistics

**Expectation:** native candidate with a multi-level reduction diagnostic.

Generate customers containing sessions containing transactions. Use targets
whose answers depend on different hierarchy levels:

```text
global_total = sum(all transaction amounts)
largest_session_total = max(sum(amounts within each session))
largest_session_average = max(mean(amounts within each session))
```

Construct paired records with identical flattened transactions but different
session assignments. The global total should be invariant to regrouping, while
the latter two answers should change. This tests whether local branches preserve
the sufficient statistic required by their parent rather than merely producing
an undifferentiated embedding.

Implementation snapshot: a one-level branch sums six independently sampled
values. Mean learns a scalar average but gives the same downstream input for
one versus six copies of an equal value, documenting its cardinality boundary.
In the nested paired case, one-output Attention's additive mass lane recovers
about 84% of the true regrouping delta and beats the best hierarchy-erased
oracle by the required margin.

### Collection overlap and anti-join reasoning

**Expectation:** bottleneck probe.

Give two collections of unseen Hash identities and predict several set
properties:

```text
has_overlap = any(left.id == right.id)
intersection_size = count(distinct left.id intersect right.id)
left_is_subset = all(left.id in right.id)
```

Balance results while matching branch lengths and ID frequencies. Simultaneous
ID renaming and within-side permutation must leave the intended answer
unchanged. The test asks whether pass-through sibling branch representations
retain enough identity information and whether the scalar decoder can compare
their members.

Implementation snapshot: the colocated two-Hash primitive reaches 0.9890 AUC
on unseen IDs, but the ordinary direct sibling-collection route remains at
chance: 0.4972 intact AUC, 0.5153 after rename/permutation, and 0.4940 after
breaking all overlaps. This is an explicit unresolved capability, not a public
feature claim. A possible bounded, opt-in mechanism is deferred to
[`RELATIONS.md`](../RELATIONS.md).

### Query-to-candidate nearest-neighbor retrieval

**Expectation:** bottleneck probe that may become a relational objective.

Provide one query Vector and a branch of candidate Vectors with attached Number
or Category payloads. The target is the payload of the candidate with largest
cosine similarity to the query. Generate fresh prototypes and payload bindings
per observation so neither position nor payload can be memorized.

The current candidate branch is reduced before interacting with a root-level
query. Compare that direct form with a shared-context form that repeats the query
inside each candidate record. Success only in shared context motivates
query-conditioned branch reduction or decoder access to candidate memory.

### Cross-branch temporal alignment

**Expectation:** relational objective.

Generate two independently ordered event streams with timestamps. A masked
target in stream B depends on the nearest, same-time, or fixed-lag event in
stream A:

```text
target_B(t) = value_A(nearest timestamp to t - lag)
```

Use elapsed-time Numbers, disjoint observation timelines, unequal stream
lengths, and distractor events. Compare exact-time matching before nearest or
as-of semantics. Every variant must declare tie-breaking and missing-match
behavior. A preprocessing join is the oracle/control, not the tested model
input.

### Multi-hop identity chains

**Expectation:** relational objective.

Create an edge collection, an entity-attribute collection, and a query entity:

```text
edges = [{source_id, destination_id}]
attributes = [{entity_id, value}]
answer(q, 1) = attribute(destination(q))
answer(q, 2) = attribute(destination(destination(q)))
```

Use unseen Hash IDs and fresh graphs in every observation. Start with one unique
path and no cycles, then add distractor edges, two hops, branching, and cycles.
One-hop and two-hop results must be reported separately. A shared-context graph
encoding is the positive control. This is the clearest proof for whether relflow
needs iterative message passing or an explicit relation node.

### Conservation and reconciliation

**Expectation:** learned-relation benchmark for one branch; bottleneck probe
across branches.

Generate ledgers whose entries have account, debit/credit role, and amount.
Tasks include predicting whether the ledger balances and reconstructing one
masked amount required to make the signed total zero. Match balanced and
unbalanced examples on entry count and amount marginals.

Extend to two branches—such as invoices and payments—where a masked residual is
the difference of grouped sums. This combines conditional aggregation with an
invariant that provides unusually strong metamorphic checks.

Omitting those deterministic totals is justified here only when the experiment
is intended to test whether the model can learn the signed or cross-branch
relationship, or when the production target is genuinely noisy and predictive.
Exact ledger balance, reconciliation residuals, and missing values determined
by an accounting identity belong in a preprocessor or the application and
should be validated as exact calculations, not delegated to a learned
approximation.

### Redundant-view recovery under compound masking

**Expectation:** native candidate within one heritage; bottleneck probe when
views live in sibling branches.

Generate several noisy measurements of one latent value using different
datatypes or branches. Evaluate every mask pattern: one view absent, two absent,
and all informative views absent. Reconstruction should degrade monotonically
with information removal and reach the marginal baseline when no informative
view remains.

This tests graceful degradation, multi-policy masking, and whether the model is
using complementary context rather than one accidentally dominant field.

## Mutation Proofs

**Status: P046–P050 implement five experiments; the remaining
scenarios below are planned.** Recorded outcomes live in `results.yaml`.

The original 45 scenarios measure models with fixed schemas. Mutation has unit
coverage for selection, validation, graph rebuilding, compatible state,
vocabulary retention, rollback, and loop locks. The missing evidence is what
happens to an already learned function after a schema edit, and whether the
changed model can learn its new task. Executable coverage is:

| ID | Script | Scope |
| --- | --- | --- |
| P046 | [Neutral edits](mutations/neutral_edits_preserve_learning.py) | Metadata, inactive extension/deletion, equivalent source rebinding, rejected edit, and save/load |
| P047 | [Added target](mutations/extend_prediction_target.py) | Hidden output extension, adaptation with rehearsal, continuation/scratch controls, and save/load |
| P048 | [Selective reset](mutations/reset_prediction_target.py) | One hidden head, unchanged/complete-reset controls, relearning, and save/load |
| P049 | [Deactivate and restore](mutations/deactivate_and_restore_input.py) | Trained pure input, repeated active toggles, normal/exceptional override exits, and inactive checkpoint reactivation |
| P050 | [Delete and adapt](mutations/delete_input_and_adapt.py) | Trained input deletion, inactive/scratch/continuation controls, conditional-mean adaptation, re-addition, and save/load |

These are empirical hypotheses with provisional gates. A failed preservation
gate remains visible even when adaptation succeeds. P047 additionally isolates
the branch pool's schema-derived capacity as a diagnostic control; this internal
intervention is restored before fitting and is not a public mutation recipe.

The first three GPU seed repeats pass every P046 and P048 gate. P047 learns
the added target and retains useful old-task accuracy after adaptation, but
fails immediate prediction preservation in all three seeds. Its pooling-scale
control restores the original predictions. Keep that failure visible; it is
evidence of a mutation boundary to resolve, not a reason to relax the gate.

The public operations are `update`, `extend`, `delete`, `reset`, and `override`.
`select` identifies their scope. The contracts come from the
[mutation guide](../docs/guides/schema-mutation.qmd),
[schema editor](../src/relflow/architecture/mutations.py), and
[graph rebuild](../src/relflow/architecture/graph.py). Rebuilding copies state
entries with matching names and tensor shapes, or matching Python types for
non-tensor state. It is not a general migration of semantic identities:
renaming a node or resizing its tensors can initialize learned state anew.

Separate three claims in every experiment:

1. **Preservation:** an edit that leaves the computation equivalent should
   retain an already learned prediction.
2. **Immediate effect:** removing information or resetting learned components
   should change behavior in a specified way before any more training.
3. **Adaptation:** the edited model should learn the new relationship within a
   declared budget. A transfer advantage over a fresh model is an additional
   hypothesis, not a condition for the mutation API to be correct.

### Shared protocol

Use this sequence for each paired seed:

```text
train source -> save source checkpoint -> fork experiment and controls
             -> mutate -> evaluate before further training
             -> adapt with a new optimizer -> evaluate -> save/load and evaluate
```

- Use independent, restartable synthetic generators for source training,
  adaptation, validation, and test. Keep a fixed held-out source panel to
  measure forgetting. Counterfactual pairs share the same latent draw and
  belong to the same split.
- Start arms from separate loads of the same trained checkpoint. Reset the
  initialization RNG before each edit or fresh-model construction. Record the
  actual mutation order and selected addresses; module object identities are
  expected to change during rebuilding.
- Require the source model to clear its learning gate first. If it does not,
  record that failure and mark downstream evidence uninterpretable. A mutation
  cannot demonstrate forgetting when the source never learned the task.
- Measure predictions immediately after the edit and after adaptation.
  Preserved state values alone do not demonstrate preserved behavior. Equally,
  retaining weights does not require identical predictions when the edit
  changes visible context, target meaning, or reduction.
- Mutate between completed loops. Set `model.optimizer = rf.adamw(...)` and use
  a new Trainer for each fitting phase. Check that the optimizer covers the
  current trainable parameters, including newly added nodes. Never continue
  through an optimizer instance attached to replaced parameters.
- At evaluation, omit supervised target values from input and disable dropout.
  For leakage controls, also supply incorrect target placeholders and require
  the same predictions. Do not train inside an `override` used to measure
  restoration: that context restores schema attributes, not a full checkpoint.
- Observe learned Number normalization and Category vocabulary state as well
  as weights. Use prediction mode for comparisons so evaluation does not fit
  statistics or grow vocabularies from the test set.
- Hold adaptation examples, batches, updates, and evaluation checkpoints fixed
  across arms. Compare the edited model with a fresh model of the final schema
  at the same adaptation budget. Report a second fresh-model control at the
  full source-plus-adaptation budget when claiming a pretraining advantage.
  Restart the optimizer in the unchanged continuation control too.

Start with small models and explicit budgets, such as 512 source updates and
256 adaptation updates, evaluating at adaptation steps 0, 32, 128, and 256.
These are pilot settings; choose final budgets and thresholds using validation
and separate calibration seeds, never the reported test panel. A `--steps`
override caps each phase and remains a smoke run.

Report source error, immediate post-edit error, final error, old-task retention,
new-task quality, control scores, and updates to a validation threshold.
Normalize regression errors against a constant fitted on the corresponding
training split. Keep that denominator fixed across comparison arms.

For prediction-preserving edits, start with `rtol=1e-5` and
`atol=1e-6 * max(training_target_sd, 1e-8)` in deterministic float32 evaluation;
also report the actual maximum and RMS drift. Require compatible state values
to remain exactly equal when no training has occurred. For simple regression
tasks, an initial learning gate is nRMSE below 0.25. Gate values are provisional
and follow the calibration protocol above. Promote claims with three paired
seeds; use the ten-seed protocol before freezing new thresholds.

### Proposed experiments

Each row becomes its own annotated script under `proofs/mutations/`. Assign
permanent IDs when the scripts and registry entries are added. Design labels
below describe hypotheses, not existing catalog entries.

| Experiment | Edit | Main observation | Essential control |
| --- | --- | --- | --- |
| Neutral rebuild | Update metadata; append and remove an inactive input | Learned predictions and retained state survive repeated rebuilding | Unchanged checkpoint and a reset model |
| Add a target | Extend a trained model with a hidden output | New head learns while old tasks retain skill | Fresh final schema and unchanged continuation |
| Add an informative input | Extend with a formerly unavailable feature | Prediction beats the best estimate from the old inputs | New feature independently shuffled |
| Add a repeated branch | Extend with a nested collection | New subtree contributes to a target after adaptation | Identical old inputs with different collection contents |
| Change a masking role | Update a reconstructing field to `mask=True` | Hidden-target learning continues without target leakage | Missing, correct, and corrupted target placeholders |
| Ablate and restore | Override an input's activity, then delete it in a separate arm | Signal loss is visible; a non-training override restores behavior | Corrupt the signal while preserving the original answer |
| Reset and relearn | Reset one output; separately reset a branch and its descendants | Selected knowledge is lost and can be relearned | No reset and complete model reset |
| Increase branch capacity | Update `length` | Old-size behavior survives and new tail items become usable | Same retained prefix, different tails |
| Resize a vocabulary | Update Category capacity | Existing identity knowledge and new-category learning are measured separately | Fresh larger vocabulary and unchanged-capacity model |
| Compose edits and reload | Extend, adapt, reject an invalid edit, save/load, continue | Mutation sequence and learned behavior survive serialization | In-memory continuation of the same edited checkpoint |

### Neutral rebuild

Train `y = x + offset(code)`, with continuous `x` and a small Category vocabulary
whose offsets are fixed per seed. Held-out rows use new values of `x` with the
same known category identities. This forces both numerical and categorical
state to matter.

Update a field description, append `unused=rf.Number(active=False)` to the
root, and delete that inactive field. Evaluate after each operation and repeat
the cycle to detect cumulative drift. Require output alignment, learned
quality, compatible tensors, normalization state, and vocabulary token-to-index
mapping to survive. A completely reset model must lose the source skill, so
an invariant constant predictor cannot pass this experiment.

Also rebind `x` to an equivalent source key using `update(..., query="renamed_x")`
while keeping its schema address fixed and supplying the same values. Treat
changing the schema node's name as a separate boundary: name-based state
transfer does not promise automatic retention across a new address.

### Add a target

Draw independent `a, b ~ Uniform(-1, 1)`. Train two hidden targets,
`u = a + 2b` and `v = 2a - b`, so the source tasks require both input degrees
of freedom. Extend the root with `w=rf.Number(mask=True)`, where `w = 3a + b`.

Before adaptation, record old-target drift and verify that the new output is
present and correctly aligned. After adaptation, require useful held-out
accuracy for all three targets. Include old targets during adaptation to make
retention with rehearsal explicit. A later no-rehearsal experiment measures
forgetting; it must not silently inherit the same retention claim.

Compare learning curves with the final schema trained fresh, and with the
original model receiving the same continuation data. An optional shuffled-`w`
training arm must not acquire held-out skill. Record whether transfer improves
the adaptation curve; do not describe faster learning unless that comparison
supports it.

### Add an informative input

Draw independent `a, b ~ Uniform(-1, 1)` and set `y = a + b`. The source schema
contains only `a` and hidden `y`. Its best prediction using the available
information is `a`; the missing `b` contributes irreducible variance `1/3`.
Relative to the zero-mean constant baseline, the population nRMSE floor is
`1 / sqrt(2)`. Report empirical oracle error on the actual held-out rows too.

Extend the trained model with `b=rf.Number` and adapt. Require nRMSE below the
provisional learning gate and below the old-information oracle. Pair rows with
the same `a` and opposite `b` to show that predictions respond to the new
feature. Shuffling `b` across held-out rows while keeping their original targets
must destroy much of the improvement. There is no immediate prediction-equality
requirement here: the edit intentionally changes the evidence available.

### Add a repeated branch

Begin with a scalar input `base`, an old hidden target `old = 2 * base`, and
hidden `total = base + sum(event.amount)`. Initially the schema cannot see
events. Extend it with `events=rf.Branch(length=4, amount=rf.Number)`.

Use signed amounts, lengths one through four, and counterfactual rows with the
same `base` but different event totals. Adapt on both targets. Require the new
task to beat its old-information oracle and retain old-task skill relative to
the unchanged continuation control. Alter amounts while holding event count
fixed to rule out a count-only shortcut. Compare with a model that had the
same branch from initialization to separate a mutation failure from difficulty
learning aggregation at all.

### Change a masking role

Train a small reconstruction task with visible `a, b` and a field `z = a - b`
using a nonzero train-time masking rate on `z`. Then use the public update API
to make `z` an always-hidden reconstruction target with `mask=True`.

Continue training and evaluate `z` using only `a, b`. Compare with a model that
used `mask=True` throughout. Supplying true `z`, omitting `z`, or replacing it
with arbitrary incorrect values at prediction time must produce the same
post-mutation predictions. Shuffling visible `b` must worsen prediction against
the original labels. Record state and content quality separately so predicting
that the target is present cannot substitute for learning its value.

Score source-stage conditional reconstruction only with `z` withheld. Accuracy
obtained while its true value is visible is not evidence of predictive skill.

### Ablate, restore, and delete

Implemented separately as **P049** (deactivate and restore) and **P050**
(delete and adapt). Their measured outcomes are recorded in `results.yaml`.

Train `y = a + b` with two independent visible inputs. In a read-only prediction
phase, override `b` with `active=False`; compare against the full model and
against independently shuffled `b`. Require a measurable error increase and
report the best possible error without `b` as the information boundary.

After normal context exit and after an intentional exception inside the
context, require restoration of the source predictions within tolerance.
Choose a pure input whose compatible learned components survive inactivity;
do not generalize this to edits that remove a trained decoder or resize state.

In a separate checkpoint arm, delete `b` permanently and retrain with only
`a`. The model should approach the restricted-information oracle, not recover
the missing independent signal. Deleting and re-extending the same name creates
new node state; checkpoint restoration is the control for actual restoration.

P049 repeats `update(..., active=False/True)` three times, checks normal and
exceptional `override` exits, and saves an inactive checkpoint before loading
and reactivating it. It does not train during inactivity. A shuffled-input
control must destroy learned skill; supplied, omitted, and changed inactive
values must give equivalent predictions. All original state entries and the
restored schema must survive. Output-head and branch deactivation remain outside
this pure-input experiment.

P050 uses 512 source updates on 2,048 rows and 256 adaptation updates on a
separate 4,096-row split. It forks deleted, inactive, unchanged-continuation,
and fresh restricted-schema arms. Each uses the same adaptation rows and batch
seed; fresh optimizers must cover the current parameters. Validation curves
are recorded at steps 0, 32, 128, and 256. The deleted and scratch schemas must
match, including field order. Immediate deletion and deactivation predictions
are compared before either arm adapts.

Both proofs use independent 2,048-row test panels and provisional source
nRMSE below 0.25. Immediate removal must raise error by more than 0.35 nRMSE
and reach at least 90% of the measured `a`-only oracle error. In P050, the
adapted restricted models must finish within oracle nRMSE − 0.05 to + 0.10,
and their prediction distance from `a` must stay below 0.20 baseline RMSE.
Those finite-sample tolerances do not assert that a population information
bound is an absolute lower bound on every test sample. Constant prediction
cannot satisfy the conditional-mean gate.

The re-addition control starts from its own source checkpoint, deletes `b`,
and extends the same name again. It checks that the new node's state and
normalization are fresh. Re-addition also appends the field after the target;
record that order change and do not attribute its prediction difference solely
to fresh state. This control does not claim recovery without retraining.

The first three GPU seeds pass all P049 and P050 checks. Reactivation restores
source predictions exactly. Deleted and inactive models adapt close to the
measured conditional-mean error, with comparable scratch controls; the
unchanged model retains accuracy using both inputs. Gates remain provisional
until the separate calibration runs are complete.

### Reset and relearn

Train two hidden outputs from the same visible inputs. Reset one output node
and require an immediate loss of its learned skill. Its normalizer and other
owned state are included in the reset. The other hidden output should retain
its prediction because neither visible context nor its own state changed.
Require relearning after adaptation and compare its curve with no reset and
complete model reset.

Run a separate branch case for `descendants=False` versus `descendants=True`.
The first resets the branch encoder; the second also resets its descendant
fields, including owned vocabularies. Check the actual reset scope and measure
both local and parent-target quality. State outside the selected scope should
remain intact, but parent predictions can change when their context depends on
the reset subtree. Do not require independence that the schema does not supply.

### Increase branch capacity

Train a sum task on signed item amounts with `length=4` and `overflow="error"`.
Update the branch to `length=8`. First compare predictions on the old lengths
without further training. Then evaluate lengths five through eight both before
and after adaptation, reporting zero-shot generalization separately.

Use pairs with identical first four items and different remaining items. The
adapted model must distinguish their totals and beat a predictor using only the
first four values. An unchanged-capacity model must reject oversized rows;
silent truncation must not be mistaken for a model-quality result. A static
capacity-eight control and the existing cardinality proofs distinguish schema
growth from ordinary length extrapolation. Increasing capacity is not itself
evidence that the added range has been learned.

### Resize a vocabulary

Train a Category input to predict randomly assigned, balanced labels for its
known identities. Use an initial capacity comfortably above the observed
vocabulary, then increase capacity and introduce additional identities during
adaptation. Keep old and new identities balanced in the held-out report.

Inspect the vocabulary mapping and embedding tensors immediately after the
resize, then report old-identity accuracy, new-identity accuracy, and unavailable
rates. The current rebuild skips tensors with changed shapes; it does not copy
the old rows into a larger embedding table. Consequently, preservation of old
accuracy is an open hypothesis, not an existing API guarantee. Record a rebuild
rejection as an execution result, and never silently shrink or remap the data
to make it succeed. Establishing lossless vocabulary growth may require a
separate implementation change if this experiment exposes the gap.

### Compose edits and reload

Start from the trained add-target experiment. Add the hidden target, adapt,
apply an equivalent query rebind, and attempt an invalid duplicate-name
extension. Require the rejected edit to leave schema, learned state, output
alignment, and predictions intact; follow it with a valid edit to test recovery.

Save the edited model and load it through `rf.Model.load`. Require the edited
schema, vocabulary mappings, normalization state, and held-out predictions to
survive. Continue both the loaded and in-memory arms with fresh optimizers,
paired RNG state, and identical data. Compare final quality rather than
requiring serialization to restore an optimizer trajectory it never promised.

Repeat several neutral edit cycles before saving to expose accumulated damage.
Include reuse of an existing data module in one continuation arm and a freshly
constructed data module in its matched control. Both must read the current
schema and include new objectives. Loop locks and rejection mechanics remain
unit-test responsibilities; this experiment measures their consequences for
an actually trained artifact.

### Implementation priority and evidence

**Neutral rebuild, added target, reset/relearn, deactivation/restoration, and
deletion/adaptation** now have executable experiments. They test retention,
extension of capability, and loss of learned information using simple tasks
with matched controls. Next add input and branch growth and masking-role
changes. Finish with capacity, vocabulary, and composed checkpoint lifecycles.

Each script must contain before/after schema illustrations, at least three
contrasting YAML records, the actual mutation calls, all training phases,
matched controls, and named behavioral checks. Reuse the current reporting
format with nested metrics for `source`, `immediate`, `adapted`, and `controls`.
Record the mutation sequence, phase budgets, source checkpoint origin, and
optimizer policy with the result. Register a new script with an empty run
history and explicit unmeasured status; never copy a fixed-schema proof's
passing measurements into a mutation entry.

The original 45 experiments and their results remain fixed-schema evidence.
Success in these new experiments would establish only the tested edits and
data distributions, not arbitrary architecture surgery, address migration,
lossless tensor resizing, or training through an active mutation.

## Metamorphic Modeling Checks

Synthetic ground truth allows stronger checks than one held-out score. Apply
these transformations to paired prediction requests from the same trained
model and assert the expected relationship between outputs.

### Reduction semantics

- Permuting complete items leaves sum, mean, min, and max unchanged.
- Duplicating every item doubles sum but leaves mean, min, and max unchanged.
- Adding a constant `c` adds `n*c` to sum and `c` to mean, min, and max.
- Multiplying all values by positive `a` multiplies every reduction by `a`.
- Adding a value below the current maximum cannot change max; adding one above
  it must change max toward that value. Apply the symmetric rule to min.
- Adding an item outside the selected group cannot change a filtered answer.
- Splitting one group into two and summing their predicted subtotals should
  agree with the predicted total within a declared tolerance.

### Identity and retrieval semantics

- Apply one random bijection to every occurrence of an unseen Hash ID; matched
  outputs must remain unchanged.
- Jointly permuting source and request items must permute item-level outputs but
  preserve key-value answers.
- Permuting only source order must not change target-order answers.
- Replacing one key while holding values fixed must affect only queries whose
  match changed.
- Adding unmatched distractor entities should not materially change answers for
  existing matched entities within the calibrated capacity range.

### Hierarchy and time semantics

- Regrouping items leaves truly global reductions unchanged but changes targets
  defined by group-local extrema or averages.
- Shifting every timestamp by one constant leaves elapsed-time relationships
  unchanged; calendar targets change only when the requested calendar part
  changes.
- Reversing a trajectory preserves its multiset aggregates while reversing the
  sign of a slope target.
- Making an interpolation gap wider must not systematically improve its error.

Metamorphic gates should use error in target-standard-deviation units. Exact
equality is appropriate only when tensorization makes two inputs identical, as
with Set permutation and duplicate collapse. Learned approximate arithmetic and
retrieval require calibrated tolerances.

## Extended And Exploratory Hypotheses

These experiments are valuable, but their expected result is configuration- or
asset-dependent. They should begin as characterization reports and become gates
only after a stable separation is demonstrated.

| Hypothesis | Experiment |
| --- | --- |
| Self-attention adds pairwise capacity | Compare `attention="mha"` with `attention=None` on duplicate detection or matched-pair reasoning inside a branch. |
| GQA and MQA preserve sufficient quality | Run item alignment, order-dependent history, and nested context at matched update and parameter budgets across `mha`, `gqa`, and `mqa`; report quality and throughput together. |
| Query pooling exceeds mean pooling when target slots need distinct context | Compare decoder pooling on item-aligned reconstruction while keeping the rest of the model fixed. |
| Jitter improves measurement robustness | Train mixed flat supervision with and without declared Number jitter, then test several unseen noise magnitudes. Require clean-data non-inferiority before claiming robustness. |
| Longer or deeper schemas buy useful capacity | Sweep branch length, encoder depth, and `d_model` on one fixed generator; plot learning curves against parameters and optimizer steps. |
| Multi-task reconstruction improves a scarce target | Compare a supervised target alone with the same target plus correlated reconstruction objectives, holding total updates fixed. |
| Frozen Text features transfer semantics | With a pinned local encoder, classify held-out paraphrase templates while holding out surface forms. Mark this proof optional because model assets are external. |
| Checkpoint round-trip preserves learned behavior | Save the best model from one core proof, load it, and require identical held-out Arrow predictions within numerical tolerance. |

No architecture alternative should be declared better from its training loss,
parameter count, or throughput alone. Use the same synthetic process and report
the quality/resource tradeoff.

## Suite Layout And Execution

Each experiment is one annotated Python script under `proofs/<category>/`, with
a permanent ID such as `P014`. `results.yaml` maps that ID to its script and staged docs
path; do not reuse IDs when filenames or titles change. Keep descriptive
filenames for readers and use IDs in automation and result references.

```text
proofs/
├── README.md
├── SPEC.md
├── results.yaml
├── reporting.py
├── render.py
├── source.py
├── run.py
├── aggregation/
│   └── raw_value_weight_sum.py
├── calibration/
├── cluster/
├── identity/
├── relational/
│   └── hash_recall.py
├── state/
├── structure/
└── temporal/
```

Choose the directory for the broad behavior being examined; the script's page
metadata names its more specific catalog family. Keep shared tooling and the
results registry at the root. Moving a script updates its registry path without
changing its ID, published URL, or recorded run history.

Each script owns its seeded record generator, model, training, metrics, and
controls. Use `rf.SyntheticDataModule` with restartable generator factories.
Keep Arrow conversion inside relflow. Materialize held-out records only when
it makes paired evaluation clearer. Do not share dataset or model helpers
between experiments: the reader should be able to understand one file on its
own. `reporting.py` owns CLI handling and result recording; `source.py` reads
page metadata and fingerprints Python syntax without importing the experiment.

Each script exposes `run(seed, steps, accelerator)`, returning measured metrics
and named Boolean checks. Run it through an ordinary `__main__` guard. Report
all checks, even when some are not met; a learning outcome is data, not a pytest
assertion. Preserve dataset, corruption, and shape invariants with ordinary
validation errors. Defaults keep the full experiment budget; a `--steps`
override is explicitly recorded as a smoke run.

Author prose in `# %% [markdown]` comment cells and Python in `# %%` code cells.
The first Markdown cell contains the page metadata, including `proof-id`,
`code-fold: true`, and `execute: {enabled: false, eval: false}`. Quarto's native
script rendering needs no Jupyter environment. Both execution flags are also
disabled globally. Four foldable code sections keep setup, data and controls,
training and evaluation, and the reporting entrypoint beside their explanations.

Include a schema-faithful Typst tree, contrasting YAML records, controls,
current insights, and remaining work in that same script. The docs build stages
ignored copies under `docs/proofs/<category>/<slug>.py`, preserving published
URLs. Do not maintain a separate authored page. Shortcodes insert status,
evidence, and reproduction commands with a script download.

Measurements remain in `results.yaml`; gates are criteria, not measured
results. Historical interpretations stay archived there while current insights
are authored in the script. Full file hashes retain provenance, and a Python
AST fingerprint detects code changes without invalidating results for Markdown
or formatting edits. Retain the original recorded hashes when reorganizing
scripts; those hashes identify the source used for each measured run.

The normal unit suite remains `pytest` under `tests/`. Run experiments with:

```bash
uv run python proofs/run.py --list
uv run python proofs/run.py P014 --accelerator gpu
PYTHONPATH=proofs uv run python proofs/aggregation/raw_value_weight_sum.py
make proofs
```

Prefer the ID command. Direct script execution uses `PYTHONPATH=proofs` from the
repository root to import shared reporting; the runner sets this path itself.

Run serially by default. Record seeds, budgets, environment, source fingerprints,
metrics, and every check in YAML. Failed execution is separate from unmet
behavioral gates. Shortened smoke runs never update the capability status.

## Failure Reports

A proof failure should print:

- claim name and seed;
- model and data configuration relevant to the hypothesis;
- train, validation, test, and baseline metrics;
- terminal metric trajectory or selected checkpoint step;
- vocabulary size, unavailable rate, committed K, or branch occupancy when
  relevant; and
- the negative-control result.

Do not dump full prediction tables. Include a small deterministic sample only
when it explains the failure.

## Order Of Implementation

Build the suite in this order:

1. Harness calibration.
2. Flat mixed supervision and explicit value state.
3. Item alignment and order dependence.
4. Unconditional and operation-conditioned reduction, then weighted and
   category-conditioned reduction.
5. Unseen Hash equality, Category's OOV boundary, and same-branch associative
   recall.
6. Shared-context recall, then direct sibling entity transfer.
7. Entity-keyed aggregation, retrieval, peer-relative inference, collection
   overlap, temporal alignment, and multi-hop candidates.
8. Complete cluster partition and held-out gates.
9. Nested context, Set invariance, and hierarchical statistics.
10. Contextual reconstruction and embedding utility.
11. Calendar recurrence, cross-DatePart inference, Vector geometry, and field
    reliance.

This sequence establishes simple learning and leakage controls before using the
same harness to judge more complex structure or representation claims.
