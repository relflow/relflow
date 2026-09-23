# Model size presets

Status: implemented initial configurations, revision 1. This document specifies
five capacity presets for RelFlow: `xs`, `sm`, `md`, `lg`, and `xl`, exposed as
classmethods on `rf.Model`. A shared factory binds each immutable configuration
to its classmethod.
The recommendations target a balance of predictive capacity and compute across
flat and nested schemas. Their numerical values are engineering starting points;
predictive quality and hardware performance still require benchmarks.

A preset supplies defaults for the shared width, root encoder and reduction,
non-root branch encoders and reductions, and common leaf decoder settings.
Recommend `md` as the first model to try when there is no established baseline;
use `sm` for faster iteration. Select a preset by calling its classmethod, such
as `rf.Model.md(...)`; ordinary `rf.Model(...)` retains its current behavior.

**Recommended configurations.** Each column is a complete preset. `branch`
means every non-root branch, at every depth; the generated root stays anonymous
at `/` with `length=1`. Width is global across the model.

| Parameter | `xs` | `sm` | `md` | `lg` | `xl` |
| --- | ---: | ---: | ---: | ---: | ---: |
| `d_model` | 64 | 128 | 256 | 384 | 512 |
| Root `n_layers` | 1 | 2 | 3 | 4 | 6 |
| Branch `n_layers` | 1 | 1 | 2 | 2 | 3 |
| Root and branch `n_heads` | 2 | 4 | 8 | 8 | 8 |
| Derived dimensions per head | 32 | 32 | 32 | 48 | 64 |
| Root and branch `attention` | `"mha"` | `"mha"` | `"mha"` | `"mha"` | `"mha"` |
| Root and branch `dropout` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Root reduction | Attention | Attention | Attention | Attention | Attention |
| Root reduction `n_outputs` | 1 | 2 | 4 | 8 | 16 |
| Branch reduction | Attention | Attention | Attention | Attention | Attention |
| Branch reduction `n_outputs` | 1 | 2 | 4 | 4 | 8 |
| All reduction `n_layers` | 1 | 1 | 1 | 1 | 1 |
| All reduction `n_heads` | inherit owner | inherit owner | inherit owner | inherit owner | inherit owner |
| All reduction `dropout` | inherit owner | inherit owner | inherit owner | inherit owner | inherit owner |
| All reduction `position` | `True` | `True` | `True` | `True` | `True` |
| Leaf `pooling` | `"query"` | `"query"` | `"query"` | `"query"` | `"query"` |
| Leaf `n_heads` | 2 | 4 | 8 | 8 | 8 |
| Leaf `n_linear` | 1 | 1 | 1 | 1 | 1 |
| Leaf `dropout` | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| Leaf `decoder_position` | `None` | `None` | `None` | `None` | `None` |

Reduction heads and dropout use the existing `None` inheritance semantics.
Leaf decoder heads come from the leaf profile, independently of any enclosing
branch override. `decoder_position=None` retains the current automatic policy:
position-free scalar queries and positional queries for repeated targets.

| Size | Intended starting use |
| --- | --- |
| `xs` | Cheap experiments, compact baselines, and constrained deployments. |
| `sm` | Routine iteration and relatively simple predictive relationships. |
| `md` | General-purpose starting point for nested relational tasks. |
| `lg` | More complex interactions or several targets when `md` underfits. |
| `xl` | Larger capacity experiments after measuring a benefit from `lg`. |

These names describe configurations for the same schema. They are not fixed
parameter-count bands, dataset-size thresholds, latency guarantees, or promises
that a larger model will generalize better. Each branch owns separate parameters;
schema breadth, depth, vocabulary cardinality, external encoders, and the number
of decoded fields can dominate total model size.

**Nearby tuning ranges.** The bundled revision-1 presets resolve to the exact
values above. The following are bounded experiments around those defaults, not
random search spaces automatically sampled during construction. Change one
capacity axis at a time before combining changes.

| Size | Candidate widths | Root depth | Branch depth | Root outputs | Branch outputs |
| --- | --- | --- | --- | --- | --- |
| `xs` | 32, 64, 96 | 1–2 | 1 | 1–2 | 1–2 |
| `sm` | 96, 128, 192 | 1–3 | 1–2 | 1–4 | 1–4 |
| `md` | 192, 256, 320 | 2–4 | 1–3 | 2–8 | 2–8 |
| `lg` | 320, 384, 512 | 3–6 | 1–3 | 4–16 | 2–8 |
| `xl` | 512, 640, 768 | 4–8 | 2–4 | 8–32 | 4–16 |

Retain the preset head count when it divides the new width, or explicitly choose
another valid even count. Prefer roughly 32–64 dimensions per head as a tuning
starting point, not a validation rule. All actual attention modules must satisfy
their divisibility and rotary-dimension constraints. A width override never
silently rewrites explicit head counts.

Reduction depth and decoder depth may be tested at 2 after identifying a reason
to increase them. Decoder `n_linear` is the number of query cross-attention
blocks, not the depth of a datatype's final prediction head. Extra decoder depth
is paid separately for each decoded field.

Dropout may be tuned independently, starting with 0.0, 0.05, and 0.1. Zero is the
initial preset default because it retains the current packed encoder training
path for partially present inputs. Nonzero dropout can improve generalization;
measure its quality and throughput together. A larger size alone does not decide
the amount of regularization a dataset needs.

**Why this progression.** Increasing width and root depth increases shared
context capacity. Shallower non-root encoders limit the multiplication of cost
across a large schema. Multiple reduction outputs let larger models retain more
context before passing it upward. The smaller branch output budget limits the
number of tokens presented to ancestor encoders.

Keep learned attention reduction even in `xs`. Selecting `Mean()` or disabling
attention changes the model's operations and inductive behavior; both remain
explicit task or latency choices. Keep reduction depth at one initially so the
size ladder has fewer interacting changes to evaluate.

Keep MHA in every canonical preset. GQA and MQA are efficiency tradeoffs to test
on the target backend and workload. In the current code, GQA uses half as many
K/V heads as query heads; MQA uses one. These modes affect branch sequence and
coordinate encoders. Reduction and decoder cross-attention still use MHA.
The GQA paper motivates a quality/efficiency tradeoff, chiefly in language-model
decoding; it does not establish a RelFlow speedup or justify changing attention
families automatically at a size boundary. [GQA paper](https://arxiv.org/abs/2305.13245)

Learned-query summaries are related to attention pooling in the
[Set Transformer](https://arxiv.org/abs/1810.00825). RelFlow's implementation
has its own positional and additive-evidence behavior. In particular, reductions
run after branch self-attention: fewer output slots reduce ancestor work, but
do not eliminate that branch's own input-attention cost. The five numerical
presets are a RelFlow proposal, not configurations established by that paper.

**Parameter ownership.** The following inventory is the complete scope of a
size preset. `Preset.root` and `.branch` use frozen `BranchDefaults` values;
`.leaf` uses frozen `LeafDefaults`. These configuration groups are not new
`Branch` constructors or a replacement serialized `Schema`.

| Group | Owned parameters | Meaning |
| --- | --- | --- |
| Model | `d_model` | Shared embedding and hidden width. |
| Root encoder | `n_layers`, `n_heads`, `attention`, `dropout` | Encoder at `/`. |
| Branch encoder | `n_layers`, `n_heads`, `attention`, `dropout` | Defaults for each non-root branch. |
| Root reduction | `reduction`; Attention `n_outputs`, `n_layers`, `n_heads`, `dropout`, `position` | Root context representation. |
| Branch reduction | Same reduction inventory | Context routed from a branch to its parent. |
| Leaf decoder | `pooling`, `n_heads`, `n_linear`, `dropout`, `decoder_position` | Common decoder defaults exposed by `RequestBase`. |
| Provenance | Preset name and revision | Identify the source of defaults without replacing the resolved schema. |

Apply common leaf defaults through the extension request contract. Preserve all
extension-specific fields. Do not dispatch on a concrete datatype name in the
preset resolver; a registered third-party request using the common decoder
contract must receive the same behavior.

The following remain independently configured:

| Parameters | Reason |
| --- | --- |
| `Branch.length`, `overflow` | Decide which observations and coordinates survive; size selection must not truncate more data. |
| Fields, queries, preprocessing, field order, descriptions | Define the schema and source meaning. |
| `active`, `mask`, `embed`, `nullable`, objective `weight` | Define visibility, prediction roles, output selection, and objectives. |
| Category vocabulary and `topk`, Number objective/normalization/Fourier settings, jitter, Set limits, Cluster configuration, Text model selection, other extension options | Have datatype-specific modeling, data, or output semantics. |
| Optimizer, learning rate, weight decay, scheduler, training duration, early stopping, calibration | Require a training and evaluation recipe. |
| Batch size, accumulation, precision, accelerator, worker count, packing/compilation policy | Depend on hardware and actual schema geometry. |

For a separate training recipe, tune learning rate and regularization on
validation data and measure each size at comparable budgets. Transformer tabular
benchmarks support evaluating architecture and tuning together; they do not give
a universal model-size rule. [Tabular model study](https://arxiv.org/abs/2106.11959)

Keep FFN expansion (currently 4), activation (GELU), normalization (pre-LayerNorm),
encoder rotary position, initialization, and attention kernels at their current
implementation values. They are not existing public preset knobs. Exposing one
later needs its own schema contract and evidence, rather than an undocumented
attribute injected by a preset.

**Public selection and overrides.** Select a configuration through its classmethod:

```python
import relflow as rf

model = rf.Model.md(
    n_layers=4,  # Explicit root override; child branches retain md defaults.
    balance=rf.Number,
    merchant=rf.Category,
    transactions=rf.Branch(
        length=128,  # Application-selected history length.
        overflow="tail",
        reduction=rf.Attention(n_outputs=8),
        amount=rf.Number,
        merchant=rf.Category,
    ),
    returned=rf.Category(mask=True),
)
```

Expose five preset classmethods:

| Classmethod | Selected configuration | Return type |
| --- | --- | --- |
| `rf.Model.xs(...)` | `xs` | `Self` |
| `rf.Model.sm(...)` | `sm` | `Self` |
| `rf.Model.md(...)` | `md` | `Self` |
| `rf.Model.lg(...)` | `lg` | `Self` |
| `rf.Model.xl(...)` | `xl` | `Self` |

Each factory is a real `classmethod` that creates a new instance of `cls`.
Inherited factories must construct and return the calling subclass. A shared
factory declares the constructor signature once, using a model-bound type
parameter for the subclass return type, and wraps it with `classmethod`.
Keep `cls` positional-only and construction options keyword-only.
They accept the ordinary Model tree-construction keywords, including `fields`,
child definitions, explicit configuration overrides, `batch_size`, `optimizer`,
and `scheduler`. The factories supply no training or runtime defaults themselves.
`d_model`, root `n_layers`, and root `n_heads` become optional at these factory
entry points because the selected profile supplies them. Keep supported options
discoverable in signatures and type checking, including autocomplete for schema
fields and configuration overrides.

All five factories use one shared resolution path with the selected immutable
profile. Do not introduce a `preset=` or `size=` constructor selector, separate
model subclasses for each size, or duplicate the resolution algorithm between
factories. Data fields named `preset`, `size`, or `md` remain ordinary child fields.

**Factory and policy.** The `relflow.presets` package exposes frozen `Preset`,
`BranchDefaults`, and `LeafDefaults` configurations, the `XS`, `SM`, `MD`, `LG`,
and `XL` values, and `factory(preset)`. These types and values live under
`rf.presets`; the package root exports the `presets` module.

The factory accepts a `Preset` and returns a real typed classmethod bound to
that configuration. Assigning it in a `Model` subclass preserves construction
of the calling subclass. `Preset.name` is a nonempty provenance string and
`revision` is a positive integer. Constructing or deserializing a policy is a
pure validation operation. See [Size Presets](../docs/core-concepts/model-tree.qmd#choose-a-size)
for the public examples.

**Module ownership.** The `src/relflow/presets/` package owns the immutable
configuration contracts, five bundled profiles and their revisions, public
classmethod factory, and shared schema resolver. The resolver fills scope
defaults into a fresh schema; the factory captures a validated policy snapshot
for persistence. Keep bundled values in one location; architecture modules
import the public preset package.

The public classmethods live on `Model` in `architecture/root.py` as five
one-line declarations such as `md = presets.factory(presets.MD)`. The shared
factory uses the bound profile and resolver, then constructs `cls` from the
resolved result. Its inferred return type preserves the constructor signature for
static checking. Keep preset policy out of the encoder, decoder, and execution
modules; those consume the resolved schema as usual. Preset configuration
and resolution code operate on schema definitions without importing
the runtime `Model` class; factory typing must not create an import cycle.

Direct `rf.Model(...)` construction continues to require `d_model`, root
`n_layers`, and root `n_heads` when building from fields. Continue to restore a
fully resolved schema through `rf.Model(schema=...)`. Preset classmethods reject
`schema=...` with an error directing the caller to that constructor; a saved
schema already specifies its architecture.

Resolution rules, in precedence order:

1. Preserve explicitly supplied settings on that model, branch, or leaf.
2. Fill omissions from the selected preset's corresponding scope.
3. Retain existing defaults for fields outside the preset inventory.

Root overrides remain local to the root. A global `d_model` override affects all
modules and must validate against every effective attention head count. An
explicit `None` is a value: `attention=None` disables the branch encoders,
`reduction=None` preserves routed tokens, node `dropout=None` means zero, and
leaf `decoder_position=None` keeps automatic behavior. Inside an Attention
configuration, `n_heads=None` and `dropout=None` keep owner inheritance.

An explicitly supplied reduction replaces the preset reduction as one object.
For example, `reduction=rf.Attention()` deliberately requests its one-output,
one-layer defaults, even under `xl`. Do not merge the preset's output count into
it. Explicit `rf.Mean()` and `None` must also survive unchanged.

Resolve into a fresh tree before building modules. Reusing a Branch or leaf
definition across two different presets must not mutate that definition. Run
normal schema and extension validation after resolution; errors should identify
the address, conflicting values, and the override that needs correction.

`Branch.__init__` uses an omission sentinel for preset-owned settings so
Pydantic can distinguish `Branch()` from an explicitly supplied default.
Subtree binding preserves `model_fields_set` while retaining materialized
values, including extension default factories. Comparing values to defaults
would lose this distinction. Explicitly setting a value equal to the ordinary
default still overrides the preset.

**Persistence and mutation.** The resolved schema remains authoritative for
building and restoring an existing model. Save every resolved node setting in
the checkpoint through the existing schema representation. Preset name and
revision are provenance; loading must not re-resolve the model using newer
preset defaults or silently change weights, geometry, or defaults.

Retain an immutable snapshot of the preset default policy for the distinct case
of adding new nodes with `extend(...)`. New nodes receive omitted branch/leaf
settings from that snapshot, including after checkpoint restore. Existing nodes
keep their resolved settings during updates and graph rebuilds. Persist this
construction policy separately from the resolved per-node schema; it does not
override existing nodes. A schema-only model without saved policy uses ordinary
defaults for new nodes. A later change to preset values should receive a new
revision. Restoring the saved policy uses its full configuration independently
of the current bundled values, preserving `extend(...)` behavior.

Selecting another size constructs another architecture. Changing width, depth,
or reduction outputs on an existing trained model is not an automatic preset
switch or a promise of weight compatibility.

**Compute and output consequences.** For branch `b`, define the maximum routed
input width as `T_b = length_b * (active direct leaves + sum(child branch outputs))`.
A completely inactive subtree (`T_b=0`) emits zero structural outputs. Otherwise,
Attention reduction emits its configured `K_b` outputs; Mean emits one; no
reduction emits `T_b`. These counts describe structural slots, with actual work
also affected by presence and packing.

For MHA, a transformer block has a leading parameter term of approximately
`12 * d_model**2` with the current FFN. More heads at a fixed width partition
that width; they do not multiply the MHA projection parameter count. Sequence
attention has a quadratic token-interaction term. A branch can additionally
have one coordinate-attention block when it has multiple active direct leaves;
that block is separate from the declared sequence `n_layers`.

Attention reduction has cross-attention blocks plus an additive evidence
projection with `K_b * d_model**2` weights. Output slots therefore affect model
parameters as well as parent token counts. Very deep or broad schemas should
test smaller branch output counts before increasing every branch's capacity.
Attention with more outputs than inputs is allowed and acts as an expansion;
do not silently clamp it or describe every reduction as compression.

Root outputs contribute shared decoder context. Decoders also receive ancestor
parcels and can use local sibling evidence, so root output count alone does not
describe all available decoder context. A larger preset also changes exported
embedding dimensions: with root `embed=True`, `xs` emits one width-64 vector,
while the other presets emit multiple vectors (for example, four width-256
vectors for `md`). Applications requiring one vector should explicitly use
`reduction=rf.Attention(n_outputs=1)` and evaluate that bottleneck. Leaf prediction
coordinates remain determined by the schema; reduction slots are not new records.

Presets do not promise unordered behavior. Encoder attention remains rotary;
turning off positional information in a reduction alone does not make an entire
branch permutation invariant.

**Validation before promotion.** Configuration checks and performance evidence
have different purposes. The exact matrix should first pass these behavioral
checks, covered by the implementation's automated tests:

- Construct and serialize all five presets on flat and nested schemas, including
  Category/Number targets and a registered custom tensorfield.
- Check that all five entry points are classmethods, construct the calling
  class (including a Model subclass), and expose typed overrides with subclass
  return types. Check that factories reject `schema=...` and ordinary Model
  construction retains its existing dimension requirements.
- Declare a typed subclass method through `rf.presets.factory` and confirm
  that it retains its bound immutable policy.
- Check root, non-root, and decoder settings independently, including explicit
  defaults, explicit `None`, atomic reductions, bare leaf classes, and head/width
  validation with an error at the affected address.
- Confirm reused schema definitions are unchanged and root overrides do not
  leak into branch or leaf defaults.
- Confirm complete checkpoint round-trips, unchanged behavior if bundled
  values change, and identical defaults for new nodes before and after restore.
- Check prediction alignment and exported embedding shapes for each reduction
  output count, including padding and nested repeated contexts.

Then measure flat classification/regression, nested aggregation, ordered
history, identity-based retrieval, multiple targets, and deep/broad schemas.
Include short and long branches, varying padding, and meaningful vocabulary
cardinalities. Record task metrics, actual parameters (separating external
encoders and vocabulary tables), peak memory, examples/second, and warmed batch-1
and production-batch latency on named hardware/software configurations.

Use at least three seeds, shared data splits and semantic schemas, and both
matched training budgets and suitably tuned runs. Include MHA/GQA comparisons,
branch output-count ablations, and zero/nonzero dropout. Promote the ladder only
after documenting the quality/compute tradeoffs; bigger configurations need not
win every task. Until then, call these recommended initial configurations.

**Validation and reference measurements.** All five configurations constructed with
explicit current API arguments and survived a `Schema` JSON roundtrip with
identical settings and `branch_outputs`. The implemented classmethods also have
tests for defaults, overrides, inference, typing, third-party extensions, and
checkpoint/mutation behavior. Accuracy, throughput, and latency comparisons
remain future benchmark work.

The reference schema has root `customer=Number` and `merchant=Category` inputs,
an `events=Branch(length=32)` with `amount=Number` and `merchant=Category`, and
one root `target=Category(mask=True)`. All scopes use the matrix above, with
reduction heads/dropout explicitly set to their effective inherited values.
Counts were taken before observing data or growing categorical vocabularies.
The two encoder columns include each branch's reduction and coordinate block.

| Size | Root encoder parameters | Events encoder parameters | Whole model, initial parameters |
| --- | ---: | ---: | ---: |
| `xs` | 154,304 | 154,304 | 372,998 |
| `sm` | 826,624 | 628,352 | 1,698,438 |
| `md` | 4,213,504 | 3,423,744 | 8,582,918 |
| `lg` | 11,833,344 | 7,691,520 | 21,631,494 |
| `xl` | 29,430,784 | 17,868,288 | 51,025,414 |

These counts illustrate the ladder on one fixed schema and are not advertised
size limits. They also show why keeping leaf/reduction depth at one is useful:
the width and encoder-depth progression already spans substantial capacity.

**Implementation evidence.** The current contracts above were checked against
[preset policies and resolution](../src/relflow/presets/base.py),
[bundled configurations](../src/relflow/presets/builtins.py),
[typed factories](../src/relflow/presets/construction.py),
[Model construction](../src/relflow/architecture/root.py),
[Branch definitions](../src/relflow/structs/structure.py),
[schema binding and output widths](../src/relflow/structs/experiment.py),
[common leaf options](../src/relflow/structs/tree.py),
[reduction configuration](../src/relflow/structs/reduction.py),
[branch encoding](../src/relflow/architecture/encoder.py),
[attention pools](../src/relflow/architecture/pool.py),
[decoder construction](../src/relflow/tensorfields/base.py), and
[checkpoint persistence](../src/relflow/architecture/checkpoint.py).
