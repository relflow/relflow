# relflow

relflow builds PyTorch/Lightning models from nested records. Typed fields
represent values; branches combine them into local contexts; decoders learn to
predict selected fields from the available context.

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

YAML makes the shape visible. Applications supply Arrow tables or eager Polars
DataFrames with this structure.

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

## Train And Predict

`train_table`, `validation_table`, and `request_table` below are
application-supplied Arrow tables. Prediction requests omit `returned`.

```python
import lightning.pytorch as lit

model.optimizer = rf.adamw(learning_rate=1e-3)
data = rf.ArrowDataModule(
    model=model, train=train_table, validate=validation_table
)
trainer = lit.Trainer(max_epochs=30)
trainer.fit(model=model, datamodule=data)
predictions = model.predict(request_table)
```

For eager Polars DataFrames, use `rf.PolarsDataModule` with the same split
arguments. `predict` returns an Arrow table; decoded results live under its
`predictions` column at addresses such as `order/returned`.

## Documentation

- [Getting started](https://relflow.github.io/relflow/getting-started.html)
- [Model structure](https://relflow.github.io/relflow/core-concepts/model-tree.html)
- [Data types](https://relflow.github.io/relflow/core-concepts/data-types.html)
- [Arrow and Polars](https://relflow.github.io/relflow/guides/data-modules.html)
- [Preprocessing](https://relflow.github.io/relflow/guides/preprocessors.html)
- [Training and checkpoints](https://relflow.github.io/relflow/guides/lightning.html)
- [Prediction output](https://relflow.github.io/relflow/guides/prediction-output.html)

Docs use static examples and Typst model diagrams. Build them with `make render`;
run `make check-docs` to validate the render in a temporary directory. Package
checks use `uv run pytest`; synthetic learning checks use `make proofs`.
See [CONTRIBUTING.md](CONTRIBUTING.md) for development conventions.

[Community](https://discord.gg/DVyZUkvTFA) · [Apache 2.0 license](LICENSE)
