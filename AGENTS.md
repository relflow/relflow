# RelFlow Agent Guide

This file is the high-signal context for AI coding agents working in this repo. Prefer it over rediscovering package conventions from scattered tests.

## Package Mental Model

RelFlow builds PyTorch/Lightning models from JSON-like schemas. Users normally import the package as:

```python
import relflow as rf
```

The public surface should stay usable from `relflow` directly: `Model`, `Branch`, built-in tensorfields, data modules, preprocessors, inference writers, deployment helpers, mutation predicates, and extension base classes.

The schema is the architecture. `Model(...)` receives field constructors and branch nodes, then builds the root branch, tensorfield embedders, context encoders, decoders, losses, and prediction outputs.

## Common API Patterns

Minimal supervised model:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    amount=rf.Number,
    merchant=rf.Category,
    label=rf.Category(mask=True),
)
```

Nested repeated context:

```python
model = rf.Model(
    d_model=64,
    n_layers=2,
    n_heads=4,
    line_items=rf.Branch(
        length=32,
        sku=rf.Category,
        quantity=rf.Number,
    ),
    returned=rf.Category(mask=True),
)
```

The generated root is anonymous, has address `/`, and is always a singleton;
use child `Branch(length=...)` for repeated data. All field addresses start
with `/`, such as `/amount` or `/line_items/sku`.

```python
model = rf.Model(
    d_model=32,
    n_layers=1,
    n_heads=4,
    embed=True,
    amount=rf.Number,
)
```

## Gotchas

- Do not use a public `Struct(...)` constructor. Public examples should use `Model(...)` and `Branch(...)`.
- `Model(...)` has no root-name option. `name=rf.Category` defines an ordinary
  child field at `/name`; neither `name="customer"` nor `root="customer"` names the model.
- Branch and leaf definitions have no `name` argument or positional arguments.
  Their parent supplies names through keywords, such as
  `amount=rf.Number` or `events=rf.Branch(...)`. Use a `fields` mapping for
  generated schemas or child names that collide with parent configuration:
  `rf.Branch(length=8, fields={"length": rf.Number})`.
  `model.extend(predicate, risk_score=rf.Number)` follows the same rule.
- Use bare tensorfield classes such as `rf.Category` when no options are needed;
  call them only to configure fields, such as `rf.Category(mask=True)`.
- Tensorfield options must be declared fields; pass them directly or unpack a
  mapping with `**options`. Use `description` for notes, and declare extension
  options on `RequestBase` subclasses. Undeclared options are rejected.
- Processed observation names and nesting match the schema by default. A node
  may opt into RelFlow's node-relative structural query syntax with paths such
  as `query="source.path"` or `query="items[-32:][*].sku"`; RelFlow never
  infers queries. Filters, joins, sorting, and derived values belong in an
  `rf.Preprocessor`.
- `transactions=rf.Branch(...)` reads the same-named child collection, and its
  leaves read keys such as `amount` from each child mapping.
- `Branch.length` defaults to `None`: collection width follows the largest
  retained collection in each batch, without changing the schema or parameters.
  A positive integer sets a hard limit; omitted overflow then defaults to
  `"head"`. Use `length=N, overflow="tail"` for recency-ordered histories or
  `overflow="error"` to reject excess rows. Unbounded branches require
  `overflow=None` and log one resource-exhaustion warning per model generation.
  The anonymous root remains `length=1, overflow="error"`.
- Bounded axes also compact execution to the observed batch width, up to their
  cap. Prediction writes every record collection as a variable Arrow list,
  even for bounded branches, without cap padding. Structural offsets and
  validity preserve null collections, empty collections, and null items.
  Tensorized input keeps branch `Geometry` under the reserved root address `/`
  alongside leaf tensorfields, so cloning, device transfer, and row selection
  preserve lengths and structural holes. Do not infer geometry from leaf values
  or masks.
- `mask=True` is shorthand for
  `Mask(skip=True, dropout=False, reconstruct=True)`: the field is never
  embedded and is decoded as a supervised reconstruction target.
- Every node has one canonical `mask` tuple. A float is uniform train-only
  masking; `Mask(query=...)` selects with an Arrow Boolean field; `skip=True`
  prevents tensorization and embedding; `reconstruct=True` adds an objective.
- `embed=True` emits an embedding in prediction output. It does not make the field a supervised target.
- `ffnx` is a positive integer feed-forward multiplier, defaulting to `8` on
  each node. The model root, child branches, and tensorfield decoders configure
  it independently. Hidden neural activations use tanh GELU and normalization
  uses RMSNorm; these have no configuration knobs.
- `Category`, `Set`, and `Enum` match string labels exactly; `strip=True` trims
  surrounding whitespace and `cased=False` lowercases labels before lookup, and
  the folded form is what vocabularies and predictions carry.
- `Hash` represents large identifiers with batch-salted hashes, preserving equality across fields in one encoded batch without learning a persistent vocabulary.
  Training salts derive from the run seed, epoch, and batch ordinal, never
  from the process-wide `random` state, so seeded runs repeat exactly.
- A `Category` or `Set` label admitted while training keeps a zero content row,
  so it embeds like an unknown label and admission alone never changes a
  prediction; the next training batch gives every still-zero assigned row a
  normal start before using it. Zero rows after `encode(strata="train")` are
  expected, not a bug.
- `Number` keeps source values, statistics, and readouts in float64 (float32 on
  MPS, which has no float64), so large offsets keep their spread. A float NaN
  tensorizes as the `other` state, not as a value or a null; ±inf stays valued
  but is never a regression target. Integers beyond ±2^53 are rejected.
- `DateParts` is for calendar parts. If elapsed time or recency matters, derive a `Number`.
  Zone-aware timestamps describe their own wall clock; strings carrying an offset
  are read as UTC. A `Vector` with a NaN or infinite element, and a float NaN label
  in `Category`, take the `other` state; a NaN `Set` member is dropped like a null.
- `Text(store="dir")` keeps frozen-encoder vectors on disk, keyed by encoder
  configuration and token row and shared by every split, run, and field with
  that encoder; misses are encoded once and appended as shards, the
  extension's callback flushes at epoch and phase ends, and
  `embedder.warm(strings)` fills it ahead of training. A store that fits
  `resident` MiB of accelerator memory (default a quarter of the device, up
  to 2 GiB) is uploaded once per index version and answered on the device;
  rows hash identically on numpy and torch. The in-process `cache` stays for
  runs without a store.
- `rf.Jitter(add=..., multiply=..., normalize=...)` is a shared frozen
  configuration whose `apply(inputs, mask)` method owns the vectorized affine
  tensor operation. Consumers own training activation, eligible coordinates,
  and normalization placement. `Number`, `Vector`, `DateParts`, and `Text`
  consume it. `Number` uses `normalize` to choose between raw and standardized
  values. The other built-ins have one continuous representation and apply
  jitter there without creating another normalizer.
- Tensorfield extensions declare canonical Arrow atom compatibility families with
  `Extension(types=...)`; custom Python atoms add extension-owned physical matchers
  with `Extension(..., arrow={Type: matcher})`. Keep datatype contracts out of
  `Request` and the shared ragged engine.
- Register components by role, such as
  `@extension.register(rf.Component.Request)` or
  `extension.register(rf.Component.write, custom_writer)`. Python names are
  independent of component roles; implementations follow the declared base-class
  and callable protocols.
- Preprocessors accept and return eager `polars.DataFrame` objects before
  query/coalescing. Use them for source renaming, expressions, windowing,
  joins, normalization, or row expansion/grouping. Their output is the
  canonical observation set. Persisted pipeline values and `RaggedField`
  members remain Arrow-backed.
- Postprocessors accept and return eager `polars.DataFrame` objects after
  prediction writing. Use them to reshape output for APIs or warehouses.

## Data And Training

Configure AdamW directly on `model.fit(...)`, such as
`model.fit(datamodule, batch_size=128, learning_rate=1e-4, weight_decay=0.01, max_epochs=30)`.
Omitted options use sensible defaults. Each fit starts a new optimizer run
bound to the current parameters; use `ckpt_path` to resume full training state.
Supply any scheduler for that run through `fit(scheduler=...)`. Keep
`rf.adamw(...)` factories for custom optimizers and direct `lit.Trainer` setup;
ordinary examples do not assign `model.optimizer` before calling `fit`.

Set `batch_size` on `fit`, `validate`, `test`, or `predict` to update the model's
current positive integer batch size. Arrow data modules read it when creating
loaders; custom Lightning data modules own their loader behavior. The size is per
device under DDP. Constructor `batch_size` remains the initial/checkpoint value
and supports native Trainer and Tuner workflows. Direct
`model.predict(inputs, batch_size=N)` preprocesses the full input once, chunks
processed Arrow rows before encoding, and postprocesses the combined output
once. Omitting the direct-prediction override preserves whole-input execution.

Use `ArrowDataModule(...)` for canonical examples. Documentation uses static
illustrations; use tiny data only in executable tests and proofs:

```python
import pyarrow as pa

table = pa.Table.from_pylist(records)
datamodule = rf.ArrowDataModule(
    model=model,
    train=table,
    validate=table,
    num_workers=0,
    persistent_workers=False,
    pin_memory=False,
)
```

For runtime smoke tests, train with `max_epochs=1`, `limit_train_batches=1`, and
`limit_val_batches=1`. Documentation may show realistic training configurations
without executing them.

Loader workers ship each batch packed: `ship()` consolidates the batch's
TensorDicts into one storage each and compacts the Arrow source so the batch
pickles at its own size rather than its shuffle window's. In-process batches
are untouched. Keep `pin_memory` off with workers; pinning is a main-process
cost, not an overlap.

`model.predict(...)` returns a `pyarrow.Table`. It accepts Arrow inputs directly
and retains a nonempty sequence of mappings as a small interactive convenience;
pass a typed empty Arrow object when there are no rows. Canonical decoded
results use actual schema child names under `predictions`, with branches stored
as `list<struct<...>>`. Rich leaf results retain `state`, extension-owned output,
and `inferred`. Requested embeddings are separate `representations` records:
`path` segments (root `[]`), repeated ancestor `parents` ordinals, intrinsic
`output_index`, and a fixed-width `embedding`. Omit unconfigured output columns;
`inputs` appears only when retention is selected. Models with no public output
reject prediction. The model's recorded RelFlow version travels with Arrow tables
and HTTP headers. Extension output list children must be named `element`.

PyArrow Parquet cannot round-trip fixed-size lists below null containers.
`rf.Writer` rejects those batches before writing; use Arrow IPC for sparse
vector results. Scalar rich results and nonnull representation vectors retain
normal Parquet support.

## Inference

Top-level inference exports:

- `rf.Writer` writes batch prediction output.
- `rf.Postprocessor` is the postprocess callable type.
- `rf.Deployment`, `rf.Accelerator`, `rf.JSONBackend`, and related serving types are lazy exports that require `relflow[serving]`.

## Code Style

RelFlow code should read as a short sequence of domain operations.

- Prefer the shortest precise noun or verb, such as `Plan`, `compile`,
  `bind`, `query`, `coalesce`, and `write`. Use a longer name when one word
  would hide a distinction that matters. Do not repeat the surrounding module
  or class name in an identifier.
- Never prefix a function or class name with `_`; control exposure with
  explicit `__all__` and root exports. Python protocol methods such as
  `__post_init__` are the exception.
- Name every Pydantic validator with a `check_` prefix, including validators
  that normalize or coerce inputs.
- Inline a one-use function when it only forwards a call, renames arguments,
  or hides a few incidental expressions. A named function should own a
  semantic phase, recursion, a reusable algorithm, callback identity, or an
  independently meaningful invariant.
- Prefer plain module functions for transformations. Add a class only when
  state and invariants travel together or a framework protocol requires it.
  Avoid `Manager`, `Helper`, and `Service` objects that merely relay calls.
- Keep control flow linear with validation and guard clauses followed by the
  main path. Avoid adapter chains, dispatcher pyramids, speculative
  abstractions, and parallel ways to perform the same operation.
- Comments and docstrings explain contracts, invariants, or non-obvious
  reasons. They do not narrate the syntax immediately below them.
- Error messages identify the affected domain object or address, the expected
  invariant, the actual value or type, and a useful remedy when one exists.
  Translate only exceptions whose meaning is understood, and preserve their
  cause.
- Do not preserve aliases, shims, or old and new execution paths unless
  compatibility is an explicit requirement. For an intentional breaking
  change, remove obsolete code, exports, dependencies, tests, and docs
  together.

Simplicity comes before cleverness. Vectorize work over the value axis when it
removes repeated Python traversal. A clear loop over schema nodes, addresses,
chunks, or a fixed number of tensor axes is acceptable; a recursive loop over
thousands of values is a signal to use Arrow, NumPy, or Torch.

## Ownership Boundaries

- Keep one canonical representation through a subsystem. Arrow is the CPU data
  plane; Polars and Python values are ingress adapters, Torch owns model
  computation, and Python objects reappear only at an extension-local library
  boundary that requires them or an explicit application/JSON boundary.
- Keep the batch dimension explicit through every row selection, expansion,
  grouping, shuffle, and postprocessing operation.
- Shared data code may understand Arrow containers, validity, offsets, shape,
  and shape. It must not know a built-in tensorfield name, configuration
  attribute, or value interpretation.
- A tensorfield extension owns its accepted Arrow families, semantic validation,
  tensorization, embedding, decoding, loss, output schema, writing, callbacks,
  and datatype-specific runtime resources.
- Reach optional behavior through a registered component or an explicit
  protocol. Do not infer a capability from a class name, string name, concrete
  extension import, or magic callable parameter.
- Registration must remain late-extensible. Avoid import-time snapshots of the
  registry in validation or serialization contracts.
- If adding a datatype requires a branch in shared architecture or data code,
  the extension boundary is incomplete.

## Review Heuristics

Before considering a change complete, ask:

- Can a third-party datatype use this path without editing shared data or
  architecture modules?
- Is each new helper a real phase or invariant, or just a one-use forwarding
  layer?
- Does any hot path materialize Arrow values as Python rows only to convert
  them back?
- Are there two representations, entry points, or compatibility paths where
  one would suffice?
- Are invariants enforced once at the boundary and then trusted internally?
- Do tests cover the architectural boundary as well as the built-in examples?

## Useful Commands

```bash
uv run ruff format --check
uv run ruff check
uv run pyrefly check
uv run pytest
uv run pytest tests/test_public_api.py
```

## Documentation And Learning Proofs

User documentation, executable modeling proofs, recorded learning results, and
branding assets live in the separate [relflow/docs repository](https://github.com/relflow/docs).
The website is published at [relflow.github.io/docs](https://relflow.github.io/docs/).
Follow that repository's agent guide for authoring, Typst diagrams, preview,
validation, proof execution, and publication. This package has no documentation
checkout, proof source, or site build dependency.

Coordinate documentation and proof changes with public API changes. Verify
claims against the implementation and use the current top-level imports.
Package runtime and architectural tests belong in this repository's `tests/`;
learning experiments and their canonical results belong in `relflow/docs`.
Documentation builds remain static; learning proofs run only when explicitly
invoked in that repository with RelFlow installed.
