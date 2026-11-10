# relflow CLI first release proposal

Status: Proposed. The commands, run configuration, and reports below define
the first CLI release; this change contains the design, not its implementation.

relflow users should be able to inspect a model, check its inputs, train it,
and use its saved state for batch prediction or HTTP serving from a shell.
This makes ordinary jobs easy to run from terminals, containers, CI, and
schedulers while preserving the same model, data, and extension contracts
available from Python.

The executable belongs in the base installation and uses standard-library
argument parsing. Serving continues to require `relflow[serving]`, and text
encoders retain their existing optional dependencies. Reusable inspection and
preflight operations also receive callable Python APIs; command handlers own
argument parsing, configuration, and process exit behavior.

## Release scope

| Command | Input | Result |
| --- | --- | --- |
| `relflow inspect MODEL` | Architecture template or learned checkpoint | Model contract and saved-state metadata. |
| `relflow check MODEL` | Template or checkpoint, optionally with data | Preflight report with the scope of every check. |
| `relflow train MODEL` | Template or learned artifact | Inference artifact, full checkpoints, and run reports. |
| `relflow predict MODEL` | Learned checkpoint and request files | Streamed Parquet or Arrow IPC output and a job report. |
| `relflow serve MODEL` | Learned checkpoint | Existing HTTP runtime configured for one worker. |

Use `check` for preflight. Reserve `validate` for a future evaluation command
consistent with `Model.validate(...)`, which computes validation losses.

V1 supports one device for training and prediction and one HTTP worker. The
CLI explicitly selects those limits; remaining omitted execution options use
SDK defaults. File sources use the existing source machinery, including its
supported filesystem URIs, without a second discovery implementation.

Automatic schema generation, architecture search, cloud submission, and
deployment infrastructure are separate workflows. Users supply the model's
schema and the application's observation, target, and split decisions.

## Model and run configuration

Keep architecture in the existing model template format. For a dataset with
one payment per row, `amount` and `merchant` are inputs and `disputed` is the
observed Boolean target. Training and validation contain the settled label;
prediction requests omit it and may include a retained `payment_id`.

`model.yaml`:

```yaml
preset: xs
fields:
  amount:
    type: number
  merchant:
    type: category
  disputed:
    type: boolean
    mask: true
```

The existing template loader requires explicit lowercase `type` entries.
Training options belong in a separate CLI run file, so the same architecture
remains usable by `rf.Model.from_yaml(...)`.

`train.yaml`:

```yaml
version: 1
seed: 42
data:
  train: payments/train
  validate: payments/validation
loader:
  num_workers: 0
  shuffle_rows: 65536
fit:
  batch_size: 128
  learning_rate: 0.0001
  weight_decay: 0.01
  max_epochs: 30
```

`version` identifies the run-file schema, independently of the installed
relflow version. Model YAML keeps its existing architecture and preset
resolution rules. The CLI does not introduce generic architecture overrides.

The run-file schema declares its accepted options explicitly. Its sections
are `data`, `loader`, `fit`, `predict`, and `serve`, with optional `seed`,
`imports`, `preprocessor`, and `postprocessor` entries. Each command consumes
its relevant sections and rejects incompatible combinations. V1 exposes
ordinary AdamW, stopping, batch, loader, and hardware settings; arbitrary
Trainer objects, callbacks, optimizer factories, and scheduler construction
remain Python workflows.

Explicit command-line options override the corresponding run-file values.
Omitted values keep SDK defaults, subject to the single-device and
single-server-worker limits. Reject unknown options and duplicate YAML keys.
Keep environment handling limited to existing backend settings such as
`Deployment` configuration; an explicitly resolved CLI value takes precedence.

Paths declared in a run file resolve relative to that file. Direct command-line
paths resolve relative to the working directory. Filesystem URIs keep their
existing meaning. The resolved configuration records the effective values and
path origins so moving between directories cannot silently select different
local data.

Seed model construction and the data module before execution. Record the seed
and execution layout; reproducing a seed alone does not promise identical
results across changed workers, batching, hardware, or dependency versions.

## Example workflow

The application supplies the payment datasets named in `train.yaml` and a
request dataset at `payments/requests`. These commands use the proposed CLI:

```bash
relflow inspect model.yaml

relflow check model.yaml --config train.yaml --execute --sample 128

relflow train model.yaml --config train.yaml --output runs/payment

relflow predict runs/payment/model.rf \
  --input payments/requests \
  --retain payment_id \
  --batch-size 256 \
  --output predictions/payment

relflow serve runs/payment/model.rf --host 127.0.0.1 --port 8080
```

The default prediction format is Parquet. Select `--format ipc` when Arrow
IPC is needed to preserve the output's types and structural nulls.

## Command contracts

### Inspect

For a template, report the resolved architecture tree, canonical addresses,
queries, branch limits, masking policies, reconstruction objectives, and
configured public outputs. For a checkpoint, additionally report the producing
relflow version, saved preset policy, and stored tensor counts and bytes.
State dictionaries contain buffers as well as parameters, so stored tensor
totals must not be labelled as trainable parameter counts.

Inspection reads declarations and saved state on the CPU without constructing
the neural graph, allocating accelerator memory, running a forward pass, or
loading optional encoders. Achieving this requires a reusable metadata path;
calling `Model.load(...)` constructs a graph and is insufficient for this
contract. Existing checkpoint deserialization semantics continue to apply.

`--json` returns a report with a `schema_version`, command, status, model
identity, and the same substantive information as the human-readable output.

### Check

Without execution, check declarations, configuration combinations, source
discovery, and structural schema and query compatibility where available.
Use the existing extension validators and binding contracts. Shared preflight
code must not introduce datatype-specific interpretations.

`--execute` opts into sampled preprocessing, tensorization, and any required
model construction. `--sample N` sets a positive source-row limit and requires
`--execute`; the default execution sample is 128 source rows. Execution may
load an optional encoder or populate a configured text store. It never saves
changes over the input checkpoint.

Report every phase as passed, failed, or skipped, with the source and processed
row counts actually observed. A source-row bound does not bound nested
collection sizes or rows emitted by a processor. Dataset-scoped processors
cannot establish their contract from a truncated dataset; sampled preflight
reports that phase as skipped instead of silently scanning the entire source.

Differentiate expected missing targets during prediction from suspicious input
paths. Structural compatibility, checks on sampled values, and scientific
validity are distinct results. A sample cannot certify all remaining rows,
and preflight does not prove that cutoffs or splits avoid leakage.

### Train

A template constructs fresh model state. A learned `.rf` artifact initializes
a new optimizer run from its saved architecture and learned state. Both paths
delegate to `Model.fit(...)` and save the resulting selected model with
`Model.save(...)`.

Preserve the SDK's default early stopping and rollback policies. When their
monitor requires validation data, a missing validation split is a configuration
error with a remedy. Training-only runs must explicitly select compatible
policies rather than silently disabling model selection.

`--resume RUN_DIRECTORY` loads the recorded configuration and the latest full
checkpoint identified by that run's report. It is mutually exclusive with a
positional model and a replacement run file. The checkpoint owns architecture
and learned state; only documented overrides such as a larger `max_epochs`
budget and a new output directory are accepted. Record the parent run and
checkpoint identity in the new report. Resume follows the SDK's full-checkpoint
guarantees and does not imply bit-identical replay after an execution-layout
change.

Support common explicit overrides, for example:

```bash
relflow train model.yaml --config train.yaml \
  --learning-rate 0.00003 --max-epochs 60 --output runs/payment-longer

relflow train --resume runs/payment \
  --max-epochs 60 --output runs/payment-resumed
```

An inference artifact contains schema and learned state. Full checkpoints also
preserve optimizer and loop progress. Loading an inference artifact and fitting
again must never be described as resuming the previous optimizer run.

### Predict

Build a prediction `ArrowDataModule` from the selected files and execute
`Model.predict(...)` with `rf.Writer` and `return_predictions=False`. Preserve
the canonical `predictions`, `representations`, and retained `inputs` planes,
including Arrow types and metadata at the batch-writing boundary.

`--retain` selects processed columns for joining outputs back to application
records. Preprocessors and postprocessors may change row counts; report
processed observations and written rows separately where observed. Preserve
existing source-order guarantees and do not invent row identity.

Add streaming IPC support to the public `rf.Writer`, keeping Parquet as its
default. Parquet rejects fixed-size lists beneath null containers; IPC must
round-trip those cases without materializing the entire prediction job.
Both formats require a stable schema across batches and publish a shard only
after successful completion. Choose one IPC file format supported by the
existing file-source reader, with a `.arrow` suffix, and document it explicitly.

### Serve

Construct `rf.Deployment` and call its existing `serve()` method. Expose its
checkpoint, host, port, accelerator, maximum batch size, batch timeout, queue
size, retention, and log-level settings. The CLI defaults to loopback binding;
users explicitly choose a public bind address. V1 uses one server worker so
extension imports and configured processors are established in that process.

Keep the existing HTTP routes, content negotiation, Arrow IPC and JSON
transport, and version headers. Process supervision, ingress, cloud resources,
and application authentication remain responsibilities of the host application.
The CLI reports a missing serving extra with its installation command.

## Application code and extensions

Applications can name installed extension modules and already configured
processor objects in the run file:

```yaml
version: 1
imports:
  - payments.extensions
preprocessor: payments.data:prepare
postprocessor: payments.output:warehouse
```

Import extension modules before parsing or restoring schema types. Resolve
processor references through their existing public protocols. A module may
export a processor configured with `.partial(...)`; v1 does not add a Python
expression evaluator or a second processor-parameter language.

Apply preprocessing before query resolution and retention, and postprocessing
at the existing output boundary. These processors are application code and
are not saved inside an inference artifact. Record their references in the
resolved configuration and require the application package when reusing them.

Third-party tensorfields must work through the live registry and their declared
components. CLI validation, reporting, and writing must not branch on built-in
datatype names or import concrete built-in extensions to infer capabilities.

## Run artifacts and completion

A successful training run produces:

```text
runs/payment/
  model.rf
  resolved.yaml
  report.json
  sources.json
  checkpoints/
    best.ckpt
    last.ckpt
  _SUCCESS
```

The report records actual checkpoint paths; the layout gives stable names to
the best and last full checkpoints produced by the selected policies. If a
policy produces neither, record their absence instead of fabricating files.
The saved inference model follows the final selected state from `fit(...)`.

Reports include a schema version, command, status, seed, package versions,
effective hardware, timestamps, measured outcomes, and artifact identities.
Record the input template or checkpoint digest. Source manifests reuse existing
file selection and include available paths, sizes, modification times, and
schema. File metadata does not establish immutable content identity; a content
hash is recorded only when actually computed. Credentials are excluded from
recorded configuration and source locations.

Prediction directories contain completed shards, `_SUCCESS`, and a
`_metadata/` directory holding the resolved configuration and source and job
reports. Existing source discovery skips underscore-prefixed paths, so these
sidecars do not become data fragments when consuming the output directory.
Handle an empty prediction source with its declared schema, an explicit
zero-row report, and a valid empty output rather than an ambiguous missing
artifact.

New jobs require a fresh output directory. Reserve the destination before
execution and reject an existing one, including an existing failed run.
Resuming writes a new run directory linked to its parent. This prevents stale
shards or concurrent jobs from being mistaken for current output.

Write reports atomically and create `_SUCCESS` only after every artifact is
finalized. On failure, preserve a failed report where possible, remove owned
partial output through the writer's lifecycle, and leave no success marker.
Finalized shards without `_SUCCESS` do not constitute a complete job.

## Shell behavior

Human progress and backend logs go to stderr. `inspect`, `check`, and completed
train and prediction reports support `--json`, emitting one versioned JSON
object to stdout without progress text or color. The persisted report has the
same meaning. Serving retains operational logging for its long-running process.

Exit `0` means success, `2` means invalid arguments or input contracts, and `1`
means an execution failure. A completed preflight with skipped phases states
those limits explicitly. Preserve normal interrupt handling and its nonzero
status. Error reports identify the failing phase and affected path or address;
`--debug` adds the underlying traceback to stderr.

## Implementation and release acceptance

Implement the release in three reviewable increments:

1. Add inspection and preflight APIs, strict configuration, versioned reports,
   and the executable entry point.
2. Add streamed prediction, IPC support in `rf.Writer`, and completion reports.
3. Add training, full-checkpoint resume, and serving through the existing APIs.

All five commands and both output formats are required before declaring v1
complete. Keep command handlers thin and place reusable domain behavior in
the existing owning subsystem. Package runtime and CLI tests belong in
`tests/`; published guides and their static checks are added with the actual
implementation. Documentation builds remain independent of model execution.

Release checks must exercise the installed executable in subprocesses, with
real training, serving, worker, and optional-encoder cases marked integration:

- Train a small nested model, save it, and compare CLI predictions with the
  same Python workflow, including retained identifiers and batch metadata.
- Resume a full checkpoint and verify that optimizer and loop progress are
  restored; verify that fitting an inference artifact starts a new run.
- Round-trip sparse fixed-width vectors and null, empty, and null-item
  collections through IPC. Verify precise Parquet rejection where required.
- Exercise a custom registered tensorfield and configured processors through
  model loading, preflight, training, prediction, and serving where supported.
- Verify metadata inspection and default preflight avoid graph construction
  and optional encoder downloads, and sampled checks report their actual scope.
- Check configuration precedence, relative paths, duplicate keys, invalid
  options, and incompatible validation policies.
- Verify missing optional dependencies, incompatible inputs, schema changes
  between batches, empty input, interruption, and existing output directories
  produce the specified reports and completion state.
- Start the server, wait for readiness, and compare an HTTP prediction with
  the Python model, including response types and version headers.
- Verify clean JSON stdout, stderr logging, exit codes, and wheel-installed
  entry points on the repository's supported Linux and Windows environments.

Existing package, documentation, and paired Evidence gates remain required.
These checks establish operational and data-contract correctness; they do not
substitute for application evaluation or establish a new learning claim.

## Existing implementation boundaries

The design builds on these current implementations:

- [Model templates](../src/relflow/architecture/template.py) own registered
  field parsing and existing template validation.
- [Model loops](../src/relflow/architecture/root.py) own training, checkpoint,
  and prediction entry points.
- [Checkpoint state](../src/relflow/architecture/checkpoint.py) owns saved
  metadata and learned-state reconstruction.
- [File sources](../src/relflow/data/sources.py) own discovery and manifests;
  [Arrow data modules](../src/relflow/data/datasets/arrow.py) own streaming
  batches, split behavior, preprocessing, and retention.
- [Writer](../src/relflow/inference/callback.py) owns batch persistence and
  needs the IPC addition.
- [Deployment](../src/relflow/inference/deployment.py) owns the HTTP runtime.

The main SDK additions are metadata inspection, reusable preflight reports,
and streaming IPC writing. Configuration resolution and job reports compose
these existing public operations into commands.
