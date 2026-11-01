# RelFlow documentation guide

This repository owns the RelFlow package, its static documentation and assets,
and canonical evidence experiments, runners, and recorded results. The root
[agent guide](../AGENTS.md) applies alongside this authoring guide. The site must
build from this checkout alone using the isolated static environment.

## Scope and build

- Authored website pages, assets, extensions, and `_quarto.yml` live in `docs/`.
  Canonical experiments, runners, protocol, and results live under `evidence/`.
  Each experiment owns its metadata and current result in frontmatter. Page folders
  mirror the sidebar groups (`learn/`, `use-cases/`, `guides/`, `concepts/`, `datatypes/`,
  `reference/`) and
  file names match page titles; when a page moves or is renamed, update every
  link in the same change. Preserve branding, layout, and mobile overflow
  behavior.
- Run `uv sync --project docs --locked` from the repository root, then use
  `make check`, `make check-docs`, `make render`, and `make dev` there. Static
  tools use `uv run --project docs` and `docs/.venv`; package tools use the root
  environment. Documentation tests live in `docs/tests/`.
- The build uses pinned Quarto 1.10.18, bundled Typst, and the committed
  extension. PyYAML is the only Python runtime dependency; pytest and Ruff are
  development tools. Never install or import RelFlow, Torch, Lightning, or a
  sibling checkout to render or validate the site.
- Python examples are static illustrations. Use ordinary `python` fences and
  the Markdown engine. Never execute examples, train models, read application
  data, or download model weights during a build.
- Evidence execution is a separate, explicit task in an environment containing
  the target RelFlow version. Runtime tests live in `evidence/tests/runtime/` and are excluded from default
  static validation.
- Read the Docs owns versioned hosting and pull request previews through the
  root `.readthedocs.yaml`. Its custom build runs
  `uv run --project docs python -m docs.publish`; GitHub documentation CI
  validates and uploads an artifact. Preserve the Quarto design and static
  dependency boundary when changing hosting.
- Hosted rendering takes its version URL from `READTHEDOCS_CANONICAL_URL` and
  its source revision from the checkout's Git `HEAD`; check
  `READTHEDOCS_GIT_COMMIT_HASH` when supplied. The global **View source** action
  links to that exact commit for branch, tag, and pull request builds. Do not use
  `READTHEDOCS_GIT_IDENTIFIER` as a source revision: pull request identifiers may
  be numeric. The publisher validates output before copying it to
  `READTHEDOCS_OUTPUT/html`. Keep links, canonical URLs, assets, and search within
  the current hosted version.
- Use Read the Docs' injected flyout for version selection and host search;
  configure Addons in its dashboard, without embedding a separate selector or
  hardcoded host script. Local Quarto preview and search remain available.
  Account connection, Gold membership, project ad-free settings, and domain
  cutover are external setup steps in `docs/README.md`. The owner is
  `granthamtaylor`; Gold will be enabled by the owner later, and the production
  domain remains unchanged during initial setup. Keep the default version on
  `latest`, deactivate unsupported `stable` targets, and activate only verified
  tags containing the consolidated documentation and build configuration.
  Existing tags through `v0.4.11` predate it. Do not automatically activate every
  semantic tag or retag/delete old releases; backfilling is a separate change.

## Teach in order

Use the product definition **model factory for typed, nested data**. The user
declares fields, nesting, and objectives; the schema generates the neural
architecture. Reinforce extensible field types, arbitrarily nested records
and variable-length collections, learned encodings and context, targets at
different depths, and compatible state retention when a trained schema evolves.
Use **serving** for the HTTP capability and models for the objects trained and
served. Keep infrastructure ownership explicit where it matters.

The foundation-model positioning describes the pretrain, adapt, and reuse
workflow: learn from broader records, fine-tune for downstream objectives, and
export learned representations. A new supervised model starts untrained.
Keep measured throughput and predictive comparisons tied to their workload,
hardware, task, baseline, metric, and evaluation protocol.

Start with one payment, one target, and a small model. Introduce repeated
transaction history next, then data preparation, training, and prediction.
Keep architecture internals, self-supervision, extension protocols, and
operational details on their own later pages. Give each concept one home and
link to it.

Introduce worked examples as a particular dataset and decision. Explain what
one row represents, which fields are inputs, and how the label is defined
before showing its record or model. Tutorial field names and layouts are
illustrative choices, not universal domain schemas. Use-case pages explain a
task design and its evaluation; attribute measured claims to recorded results.

Keep a progressive transaction-modeling path: one payment and a dispute target;
then a transaction sequence including that payment as its latest item; then
event attributes and account profile context with a merchant-repeat outcome;
then nested purchases, separate withdrawal history, and targets at transaction
and account levels. Introduce reconstruction and
representation reuse after the supervised workflow. Explain the business
decision and what each addition contributes before its implementation. Adapt
research use cases and terminology into native examples. Attribute specific
research claims where they are used.

Use coherent financial examples without implying that RelFlow is restricted to
finance. The first flat payment has `amount`, `merchant`, and
`disputed=rf.Boolean(mask=True)`. Sequence examples use one `trxns` branch
containing earlier transactions and the latest payment exactly once. Put
`booked_at`, `amount`, `merchant`, and `disputed` inside it; do not duplicate
current-payment fields at the root. Use
`rf.DateParts(dateparts=["day_of_month", "day_of_week", "hour_of_day"])`
for booking-time calendar features. Calendar cycles do not encode exact
elapsed durations. Oldest-to-newest order plus `overflow="tail"` keeps the
latest transactions, with no events after the scoring cutoff.

Use `disputed=rf.Boolean(mask=True)` inside `trxns`: every retained transaction
is a supervised target and all dispute labels are excluded from input.
Training observations have settled Boolean labels for every transaction.
At a common account-review cutoff, each label describes a new dispute during
the following 30 days; every input record is known by that cutoff. Prediction
requests omit `disputed` from the transaction records. Keep selectors,
`rf.Mask` policies, and dynamic-masking discussion out of introductory pages;
teach them only on their later dedicated pages.

## Public examples

Use `import relflow as rf`, `rf.Model(...)` or a size factory, and
`rf.Branch(...)`. Parent keywords name children; definitions have no `name` or
positional arguments. Use bare field classes for defaults. The model root is
anonymous at `rf.Address()`; diagrams may label it `payment` or `account`.
Use one argument per address component, such as
`rf.Address("trxns", "amount")`.

`mask=True` defines a supervised target kept out of the input. `embed=True`
exports context and creates no objective. `ffnx` defaults independently to 8
on each node; tanh GELU and RMSNorm are fixed.

Child branches default to `length=None, overflow=None`, retaining all selected
slots with batch-dependent widths. A positive integer length sets a hard cap;
only bounded branches accept `head`, `tail`, `random`, or `error`, with omitted
overflow resolving to `head`. With `overflow="random"`, training selects a
uniformly drawn contiguous window
of whole records; validation, test, and prediction retain the tail. Fields
and nested children stay aligned. Crops repeat within a fixed seeded loader
plan; batching, worker assignment, and row-order changes can change them.
Sources or preprocessors must supply more than the cap for sampling to vary,
with a separate resource bound on the candidate pool. Keep introductory sequence
examples on
`overflow="tail"`. Prediction writes every repeated
record collection as a variable Arrow list of structs, including bounded
branches; input caps do not add public padding. Fixed-width vectors retain
their feature axis. Explain memory costs and the warning
for public serving; automatic widths do not make padding free.

Prefer application-supplied Parquet file paths (`.parquet`) for ordinary
training and prediction examples. Pass paths to `rf.ArrowDataModule`, including
its `predict` split; the direct `model.predict(...)` input is an in-memory API.
Reserve table-input examples for explaining that API. Returned predictions
remain Arrow tables.

Use `rf.ArrowDataModule(train=train, validate=validate)` for canonical training
examples. Omit the model argument in every data-module example: model loops
and native Lightning Trainer loops bind the module automatically. Direct
loader access requires an explicit `data.bind(model)` first. Name
application-supplied inputs. Configure AdamW
through `model.fit(data, batch_size=128, learning_rate=1e-4, weight_decay=0.01, max_epochs=30)`;
omitted options use defaults. Each fit binds a fresh optimizer to the current
parameters, with any scheduler supplied for that run. Full training continuation
uses `ckpt_path`. Reserve `rf.adamw(...)` and explicit `model.optimizer` setup
for custom optimization or native `lit.Trainer` examples. Prediction returns
selected `predictions`, `representations`, and retained `inputs`
columns, omitting unselected planes and rejecting an empty public output plan.
Use ordinary child names in the prediction tree, never literal slash keys.
Repeated records always use variable Arrow lists, including bounded branches;
input caps do not add public padding. Fixed-width vectors retain their feature
axis. Extension output list children must be named `element`.
Embeddings live in representation records with `path`, `parents`, `output_index`,
and `embedding`. API claims must be verified against the targeted upstream source; recorded evidence
retains its original provenance even when executable output readers are
migrated. A reader migration does not establish fresh learning results. Show
observation and output examples as one YAML record, nesting repeated values
under their field names.
Keep examples realistic without constructing or executing miniature datasets.
Set batch size on the loop method in ordinary training examples. A positive
`batch_size` on any loop updates the persistent model setting; Arrow data
modules read it when creating loaders, while custom Lightning loaders own their
configuration. Under DDP it is per device. Direct prediction with explicit
`batch_size=N` preprocesses once, chunks processed Arrow rows before encoding,
and postprocesses the combined output once. Omission or `None` preserves direct
whole-input behavior. Constructor batch size remains the initial/checkpoint
setting and supports native Trainer/Tuner workflows.

Verify API claims against the targeted RelFlow version in upstream source;
existing prose alone is not evidence. Keep documentation builds independent of
installed model packages.

## Visuals and evidence

Use the shared `node` and `tree` Typst functions in
`docs/assets/typst/model-tree.typ`; read `docs/assets/typst/README.md` before
changing structural diagrams. For stages, decision times, branch tokens, and
masking policies, use `flow`, `timeline`, `tokens`, and `policies` from
`docs/assets/typst/diagrams.typ`, which share the same palette. Diagrams explain field roles and repeated context
rather than hyperparameters. Keep nearby code, YAML, labels, captions, and text
alternatives consistent. Preserve bundled extension and font licenses.

`evidence/render.py` publishes local canonical sources and reviewed
results without executing them. Generated pages and fragments under
`docs/evidence/` are ignored; never edit them as a second source. Follow
`evidence/SPEC.md` when changing experiments. Execution writes independent JSON
records. Select complete reports with `evidence/record.py` and review updates to
the experiment frontmatter; never alter measurements to make a check pass. Preserve
source fingerprints and provenance. Clean only generated publication output.
There are no Proofs redirects, snapshot importers, or optional Modal backend.
