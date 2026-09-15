<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/branding/banners/banner.dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="docs/assets/branding/banners/banner.light.svg" />
    <img alt="relflow" src="docs/assets/branding/banners/banner.light.svg" width="100%" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/relflow/"><img alt="PyPI version" src="https://img.shields.io/pypi/v/relflow?logo=pypi&amp;logoColor=white" /></a>
  <img alt="Python 3.12+" src="https://img.shields.io/badge/python-3.12%2B-3776AB?logo=python&amp;logoColor=white" />
  <a href="LICENSE"><img alt="Apache-2.0 license" src="https://img.shields.io/badge/license-Apache--2.0-2E8B57" /></a>
  <a href="https://relflow.github.io/relflow/"><img alt="Documentation" src="https://img.shields.io/badge/docs-Quarto-39729E?logo=quarto&amp;logoColor=white" /></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><img alt="Discord channel invite" src="https://img.shields.io/badge/discord-join%20the%20channel-5865F2?logo=discord&amp;logoColor=white" /></a>
  <!-- discord-invite:end -->
</p>

relflow builds PyTorch/Lightning models from nested records streamed through
Apache Arrow. Typed fields represent values; branches combine them into local
contexts; decoders learn to predict selected fields from the available context.

## Install

Python 3.12 or newer:

```bash
uv add relflow
```

Add `relflow[text]` for Hugging Face text encoders or `relflow[serving]` for the
HTTP runtime.

## Describe One Record

```yaml
line_items:
  - sku: A12
    quantity: 2
    price: 19.99
  - sku: B07
    quantity: 1
    price: 45.50
returned: false
```

YAML illustrates one observation. Store these records as nested Arrow structs
and lists in Parquet datasets; relflow scans them in batches.

```python
import relflow as rf

model = rf.Model(
    name="order",
    d_model=64,
    n_layers=2,
    n_heads=4,
    batch_size=128,
    line_items=rf.Branch(
        length=32,
        sku=rf.Category(size=4096),
        quantity=rf.Number,
        price=rf.Number,
    ),
    returned=rf.Boolean(mask=True),
)
```

Parent keywords name each field. `mask=True` makes `returned` a supervised
target whose value never enters the encoder. The branch builds line-item
context before its reduced representation reaches the order root.

## Stream Training And Prediction

The paths below are application-supplied Parquet datasets. Training and
validation records include `returned`; request records omit it.

```python
import lightning.pytorch as lit
import pyarrow.dataset as ds

train = ds.dataset("warehouse/train", format="parquet")
validation = ds.dataset("warehouse/validation", format="parquet")
requests = ds.dataset("warehouse/requests", format="parquet")

model.optimizer = rf.adamw(learning_rate=1e-3)
data = rf.ArrowDataModule(
    model=model, train=train, validate=validation, predict=requests
)
trainer = lit.Trainer(
    max_epochs=30, devices=1, callbacks=[rf.Writer("predictions")]
)
trainer.fit(model=model, datamodule=data)
trainer.predict(model=model, datamodule=data, return_predictions=False)

predictions = ds.dataset("predictions", format="parquet")
```

`ArrowDataModule` opens a fresh scan for each pass and prepares model batches
as records arrive. `rf.Writer` writes prediction batches to
`predictions/rank-0.parquet`; `return_predictions=False` avoids collecting
all outputs in memory. Open the output as an Arrow dataset for further batch
processing. Decoded values live under its `predictions` column at addresses
such as `order/returned`.

## Documentation

- [Getting started](https://relflow.github.io/relflow/getting-started.html)
- [Model structure](https://relflow.github.io/relflow/core-concepts/model-tree.html)
- [Data types](https://relflow.github.io/relflow/core-concepts/data-types.html)
- [Arrow data loading](https://relflow.github.io/relflow/guides/data-modules.html)
- [Preprocessing](https://relflow.github.io/relflow/guides/preprocessors.html)
- [Training and checkpoints](https://relflow.github.io/relflow/guides/lightning.html)
- [Batch inference](https://relflow.github.io/relflow/guides/batch-inference.html)
- [Prediction output](https://relflow.github.io/relflow/guides/prediction-output.html)

Docs use static examples and Typst model diagrams. Build them with `make render`;
run `make check-docs` to validate the render in a temporary directory. Package
checks use `uv run pytest`; synthetic learning checks use `make proofs`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.

[Community](https://discord.gg/DVyZUkvTFA) · [Apache 2.0 license](LICENSE)
