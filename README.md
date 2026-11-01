<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/relflow/docs/main/docs/assets/branding/banners/banner.dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/relflow/docs/main/docs/assets/branding/banners/banner.light.svg" />
    <img alt="relflow" src="https://raw.githubusercontent.com/relflow/docs/main/docs/assets/branding/banners/banner.light.svg" width="100%" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/relflow/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/relflow?logo=pypi&amp;logoColor=white" /></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/ci.yml?query=branch%3Amain"><img alt="CI" src="https://github.com/relflow/relflow/actions/workflows/ci.yml/badge.svg?branch=main&amp;event=push" /></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/ci.yml?query=branch%3Amain"><img alt="Line coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frelflow%2Frelflow%2Fbadges%2Fcoverage.json" /></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/compatibility.yml?query=branch%3Amain"><img alt="Dependency compatibility" src="https://github.com/relflow/relflow/actions/workflows/compatibility.yml/badge.svg?branch=main" /></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&amp;logoColor=white" />
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-2E8B57" /></a>
  <a href="https://relflow.ai/"><img alt="Documentation build" src="https://github.com/relflow/docs/actions/workflows/docs.yml/badge.svg?branch=main&amp;event=push" /></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><img alt="Discord channel invite" src="https://img.shields.io/badge/discord-join%20the%20channel-5865F2?logo=discord&amp;logoColor=white" /></a>
  <!-- discord-invite:end -->
</p>

**Build and adapt foundation models for your own structured data.**

RelFlow is a model factory for typed, nested data. Describe your records and
prediction targets, and it generates a custom neural architecture, trains
the model, and serves predictions and learned representations.

Suppose your dataset contains purchases with line items, accounts with
transaction histories, and several types of values inside each record. RelFlow lets you keep those relationships in the model
instead of hand-designing a fixed set of aggregate features. Describe the
records and the fields to predict; the schema defines the architecture.
One declaration gives you field encodings, attention over related records,
prediction heads, and training objectives, with growing vocabularies and
reusable embeddings built in.

## How it works

1. **Fields represent values.** Tensorfields such as `Number`, `Category`, and
   `Text` convert typed inputs into embeddings. Learned state, including
   numeric normalization and categorical vocabularies, stays with the model. Relflow supports nearly dozen different data types, and you can build and register your own.
2. **Branches build context.** Branches describe arbitrarily nested records
   and variable-length collections. Attention combines their fields and
   repeated records; nested branches pass learned context toward the root.
3. **Targets read that context.** Each target has a decoder and a loss suited
   to its datatype. Decoders read context along their schema paths, so targets
   can live at the root or inside repeated records.

Apache Arrow carries the data, and PyTorch Lightning runs training. Your
application prepares the observations; RelFlow builds their encoders,
decoders, and objectives. Built-in and custom field types use the same
extension protocol, so new representations and objectives plug into the
model factory.

## Measured results

**1.18 million predictions per second on one NVIDIA A10G.** In a synthetic
Arrow IPC serving benchmark, a tiny, attention-free XS model processed
**50.3 million rows with zero failed requests**, including client encoding,
HTTPS, and response decoding. The setup and three measured passes are below.

The [evidence catalog](https://relflow.ai/evidence.html) adds **79 controlled
learning experiments** with code, comparisons, seeds, and recorded results.
Examples from the published full runs:

| Experiment | RelFlow result | Control |
| --- | --- | --- |
| [Match IDs absent from training](https://relflow.ai/evidence/identity/unseen-identity-equality.html) | **0.9991 AUC** with `Hash` | 0.5000 with unknown `Category` labels |
| [Recall a value by an unfamiliar ID](https://relflow.ai/evidence/relational/hash-recall.html) | **0.100 normalized RMSE**, median of three seeds | 1.401 after corrupting the lookup IDs |
| [Distinguish nulls from zero](https://relflow.ai/evidence/state/null-versus-zero.html) | **100% accuracy** with value states preserved | 50% after filling nulls with zero |

These are controlled synthetic tasks. Each page includes its protocol,
comparisons, limitations, and complete run report.

<details>
<summary>Serving benchmark setup and recorded runs</summary>

Recorded 2026-10-06: one NVIDIA A10G replica with 2 CPUs and 8 GiB host memory;
an untrained, 3,142-parameter `xs` model with attention disabled and mean
pooling. Each input contains four numeric/Boolean records generated from
seeded Gaussian numbers and random Booleans. This measures serving throughput,
not prediction quality or the default attention model.

The client sent 131,072 rows per request with four concurrent requests,
Zstandard Arrow IPC, and gzip responses. Each pass completed 128 requests
(16,777,216 rows). Input generation was outside the timer; Arrow validation,
encoding on every request, HTTPS, and response decoding were inside it.

| Pass | Rows/second | p95 request latency | Failed requests |
| --- | ---: | ---: | ---: |
| 1 | 1,179,732 | 923.3 ms | 0 |
| 2 | 1,234,752 | 611.9 ms | 0 |
| 3 | 1,122,590 | 749.2 ms | 0 |

The headline is the median of these three passes.

</details>

## Install

Requires Python 3.11 or newer:

```bash
uv add relflow
```

With pip, use `python -m pip install relflow`. Optional extras include
`relflow[text]` for Hugging Face text encoders and `relflow[serving]` for HTTP
serving.

## Define a model

Suppose you have historical payment records and want to decide which new
payments deserve review just after booking. Each row contains `amount`,
`merchant`, and whether a dispute was reported in the following 30 days.
Declare the two inputs and that observed outcome:

```python
import relflow as rf

model = rf.Model.xs(
    amount=rf.Number,
    merchant=rf.Category,
    disputed=rf.Boolean(mask=True),
)
```

Keyword names match the keys in your records: `amount=rf.Number` reads the
`amount` column. Use a bare class for default options, or call it to configure
the field. `mask=True` makes `disputed` a supervised target: its label trains
the decoder but is never included in the inputs. Mark additional fields to
train multiple targets in the same model.

`xs` supplies architecture defaults for the root, child branches, and field
decoders. The [preset ladder](https://relflow.ai/reference/model-presets.html)
runs from `rf.Model.xs(...)` through `.sm(...)`, `.md(...)`, `.lg(...)`, and
`.xl(...)`; each creates a fresh, untrained model from your declared schema.
Override options such
as `d_model`, `n_layers`, or `n_heads` when needed. Use `fields={...}` for
generated schemas or field names that collide with configuration options.

<p align="center">
  <a href="https://relflow.ai/learn/your-first-model.html">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://relflow.ai/assets/diagrams/fig-first-payment-dark.svg" />
      <img alt="A payment model combines Number amount and Category merchant inputs to predict the masked Boolean disputed target." src="https://relflow.ai/assets/diagrams/fig-first-payment-light.svg" width="640" />
    </picture>
  </a>
</p>

## Train and predict

Use your application's Parquet datasets, with one payment per row. Training
and validation contain `amount`, `merchant`, and the known `disputed` label;
request rows contain the inputs and a `payment_id` for joining results.

```python
import pyarrow.dataset as ds

data = rf.ArrowDataModule(
    train=ds.dataset("payments/train", format="parquet"),
    validate=ds.dataset("payments/validation", format="parquet"),
)
model.fit(data, batch_size=128, learning_rate=1e-4, max_epochs=30)

requests = rf.ArrowDataModule(
    predict=ds.dataset("payments/requests", format="parquet"),
    retain=("payment_id",),
)
model.predict(
    datamodule=requests,
    batch_size=256,
    callbacks=[rf.Writer("payment-predictions")],
    return_predictions=False,
)

model.save("payment.rf")
restored = rf.Model.load("payment.rf")
```

Datasets are scanned in Arrow batches with a bounded training shuffle buffer;
the full inputs need not fit in memory. `Writer` streams output to
`payment-predictions/rank-*.parquet`, and `return_predictions=False` avoids
collecting the complete result in memory. File paths, globs, `rf.source(...)`,
and restartable Arrow batch factories are also supported.

`fit` configures AdamW, early stopping, and restoration of the best validation
checkpoint. Supply hardware, logging, and other Lightning options through
`fit`. Saved models include the schema, weights, and learned field state.

Prediction files contain a `predictions.disputed` record with state
probabilities, content (the probability of `True`), and an `inferred` flag.
Retained identifiers appear under `inputs`. Open the output directory with
`ds.dataset("payment-predictions", format="parquet")` to process results in
batches too.

## Add nested context

To use an account's previous transactions alongside the current payment,
include a `history` list of records in each observation and describe it with
`rf.Branch(...)`:

```python
history_model = rf.Model.xs(
    amount=rf.Number,
    merchant=rf.Category,
    history=rf.Branch(
        length=32,
        overflow="tail",
        amount=rf.Number,
        merchant=rf.Category,
    ),
    disputed=rf.Boolean(mask=True),
)
```

`history` reads the same-named collection, and its fields read each child
record's keys. With history ordered oldest to newest, this branch retains the
most recent 32 records. `length` is a cap; shorter histories need no manual
padding. Branches can nest further, and `mask=True` on a child field requests
a prediction for each retained record.

Set `embed=True` on the model, a branch, or a field to export contextual
embeddings under a separate `representations` column. This requests output;
it does not add a training objective.

## Pretrain, adapt, and reuse

The foundation-model workflow is to pretrain on a broad collection of your
structured records using reconstruction objectives, then adapt the learned
model to downstream prediction tasks. Fine-tuning updates the learned encoders
and task heads. Exporting representations lets separate predictors use that
context while the exporting model stays fixed.

A supervised model can also start from fresh initialization. Evaluate reuse
on the intended downstream tasks and populations. See
[Pretraining and fine-tuning](https://relflow.ai/learn/pretraining-and-fine-tuning.html)
and [Reusable representations](https://relflow.ai/learn/reusable-representations.html).

[Schema mutation](https://relflow.ai/guides/schema-mutation.html) can add targets
or change the input tree while retaining compatible learned state. New or
incompatible components initialize as the schema changes.

## Extensible field types

Numbers, categories, booleans, sets, dates, text, vectors, hashes, quantiles,
and latent clusters have their own representations and objectives. They use
the same extension registry available to your application.

Define a custom tensorfield with `rf.Extension`: declare its accepted Arrow
types and validated Pydantic options, then register its tensorization,
embedder, decoder, loss, and output writer by `rf.Component` role. The shared
engine handles nesting, masking, batching, and device transfer. Your datatype
owns its semantics without adding special cases to the model or data loader.
See [Custom tensorfields](https://relflow.ai/datatypes/custom-tensorfields.html)
for the full contract.

## Serve the trained model

With `relflow[serving]` installed, save this module as `serve_payments.py` beside
the `payment.rf` artifact produced above:

```python
import relflow as rf

app = rf.Deployment(checkpoint="payment.rf").app()
```

From that directory, start the HTTP server:

```bash
uvicorn serve_payments:app --host 127.0.0.1 --port 8000
```

The application chooses the hosting infrastructure and can attach shared
preprocessors and postprocessors. Batch prediction and HTTP serving use the
saved schema and learned state. See [Serving](https://relflow.ai/guides/serving.html)
for requests, validation, and resource controls, and
[Performance](https://relflow.ai/guides/performance.html) for measuring throughput
with the workload, record shape, hardware, and requested outputs stated.

## Learn more

- [Your first model](https://relflow.ai/learn/your-first-model.html)
- [Use cases](https://relflow.ai/use-cases/index.html)
- [Pretraining and fine-tuning](https://relflow.ai/learn/pretraining-and-fine-tuning.html)
- [Reusable representations](https://relflow.ai/learn/reusable-representations.html)
- [Model structure and configuration](https://relflow.ai/concepts/model-tree.html)
- [Data types](https://relflow.ai/datatypes/index.html)
- [Arrow data loading](https://relflow.ai/guides/data-modules.html)
- [Streaming batch inference](https://relflow.ai/guides/batch-inference.html)
- [Prediction output](https://relflow.ai/reference/prediction-output.html)

Tutorials, guides, and canonical learning evidence live in the separate
[relflow/docs repository](https://github.com/relflow/docs). For package
development, see [CONTRIBUTING.md](CONTRIBUTING.md).

[Community](https://discord.gg/DVyZUkvTFA) · [Apache 2.0 license](LICENSE)

## Continuous Integration

`ci.yml` is the entry point for PRs to main, merge groups, main pushes, and
manual validation. PRs run the full Python/platform test matrix and the full
paired evidence suite in parallel. `CI / required` rejects errors, cancellations,
and missing required jobs. Evidence runs on lab through the pinned docs-owned
executor, with a separate reviewed suite commit. See the
[lab guide](https://github.com/relflow/docs/blob/main/evidence/lab.md).

Main pushes run the test matrix once and build the distributions consumed by
release publication. Successful main CI also publishes coverage. Nightly
dependency compatibility and configured accelerator checks remain independent.
Local pre-commit continues to run formatting, lint, types, and fast tests.
