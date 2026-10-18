<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/relflow/docs/main/docs/assets/branding/banners/banner.dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="https://raw.githubusercontent.com/relflow/docs/main/docs/assets/branding/banners/banner.light.svg" />
    <img alt="relflow" src="https://raw.githubusercontent.com/relflow/docs/main/docs/assets/branding/banners/banner.light.svg" width="100%" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/relflow/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/relflow?logo=pypi&amp;logoColor=white" /></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/tests.yml?query=branch%3Amain"><img alt="Tests" src="https://github.com/relflow/relflow/actions/workflows/tests.yml/badge.svg?branch=main&amp;event=push" /></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/tests.yml?query=branch%3Amain"><img alt="Line coverage" src="https://img.shields.io/endpoint?url=https%3A%2F%2Fraw.githubusercontent.com%2Frelflow%2Frelflow%2Fbadges%2Fcoverage.json" /></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/compatibility.yml?query=branch%3Amain"><img alt="Dependency compatibility" src="https://github.com/relflow/relflow/actions/workflows/compatibility.yml/badge.svg?branch=main" /></a>
  <img alt="Python 3.11+" src="https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&amp;logoColor=white" />
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-2E8B57" /></a>
  <a href="https://relflow.ai/"><img alt="Documentation build" src="https://github.com/relflow/docs/actions/workflows/docs.yml/badge.svg?branch=main&amp;event=push" /></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><img alt="Discord channel invite" src="https://img.shields.io/badge/discord-join%20the%20channel-5865F2?logo=discord&amp;logoColor=white" /></a>
  <!-- discord-invite:end -->
</p>

relflow builds PyTorch/Lightning models from nested records streamed through
Apache Arrow. Typed fields represent values; branches combine them into local
contexts; decoders learn to predict selected fields from the available context.

## Install

Python 3.11 or newer:

```bash
uv add relflow
```

Add `relflow[text]` for Hugging Face text encoders or `relflow[serving]` for the
HTTP runtime.

## Describe One Record

For example, predict whether a payment will be disputed:

```yaml
payment_id: payment-42
amount: 86.50
merchant: grocery
disputed: false
```

```python
import relflow as rf

model = rf.Model.xs(
    amount=rf.Number,
    merchant=rf.Category,
    disputed=rf.Boolean(mask=True),
)
```

Parent keywords name each field. `mask=True` makes `disputed` a supervised
prediction target whose label is excluded from the inputs. `payment_id` remains
available as application metadata. Add repeated contexts with `rf.Branch(...)`
when a prediction needs transaction history or other related records.

## Train And Predict

The application supplies Arrow tables named `train_table`, `validation_table`,
and `request_table`. Training and validation include the dispute label;
prediction requests omit it.

```python
data = rf.ArrowDataModule(
    model=model,
    train=train_table,
    validate=validation_table,
)
model.fit(data, batch_size=128, learning_rate=1e-4, weight_decay=0.01, max_epochs=30)

predictions = model.predict(request_table, batch_size=256, retain=("payment_id",))
model.save("artifacts/payment.rf")
```

Fitting includes early stopping and restoration of the best validation model.
Configure AdamW directly on `fit`; each call starts with the current model
parameters and fresh optimization state. Hardware, logging, and Lightning
options can also be supplied to the model methods.
Predictions are an Arrow table: decoded results use schema field names such as
`predictions.disputed`, and repeated branches become lists of records. Requested
embeddings appear separately in `representations`. Retention adds an `inputs`
column. For larger file-backed jobs, use an Arrow data module with `rf.Writer`
to stream predictions to Parquet.

## Documentation

- [Getting started](https://relflow.ai/getting-started.html)
- [Add transaction history](https://relflow.ai/tutorials/transaction-history.html)
- [Model structure](https://relflow.ai/core-concepts/model-tree.html)
- [Data types](https://relflow.ai/core-concepts/data-types.html)
- [Arrow data loading](https://relflow.ai/guides/data-modules.html)
- [Preprocessing](https://relflow.ai/guides/preprocessors.html)
- [Training and checkpoints](https://relflow.ai/guides/lightning.html)
- [Batch inference](https://relflow.ai/guides/batch-inference.html)
- [Prediction output](https://relflow.ai/guides/prediction-output.html)

Documentation, modeling proofs, and branding sources live in the separate
[relflow/docs repository](https://github.com/relflow/docs). Its website uses
static examples and Typst diagrams and builds independently of this package.
Run package checks here with `uv run pytest`; follow the documentation
repository's instructions to run learning experiments explicitly.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.

[Community](https://discord.gg/DVyZUkvTFA) · [Apache 2.0 license](LICENSE)
