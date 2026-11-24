"""Stream a small churn training fixture from Parquet and score two customers."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

import polars as pl
import relflow as rf
import torch


class MonthlyUsage(rf.Branch):
    """Monthly observations from a subscriber's usage history."""

    month = rf.Period("month_of_year")
    minutes = rf.Number


class ChurnModel(rf.Model):
    """Predict cancellation from the subscriber's plan and usage history."""

    plan = rf.Category
    monthly_usage = MonthlyUsage(length=12, overflow="tail")
    churn_90d = rf.Boolean(mask=True)


def main() -> None:
    torch.set_num_threads(1)
    fixture = json.loads(Path(__file__).with_name("customers.json").read_text(encoding="utf-8"))
    with TemporaryDirectory(prefix="relflow-churn-") as directory:
        train = Path(directory) / "train.parquet"
        validation = Path(directory) / "validation.parquet"
        pl.DataFrame(fixture["train"]).write_parquet(train)
        pl.DataFrame(fixture["validation"]).write_parquet(validation)

        model = ChurnModel(preset="small")
        data = rf.ArrowDataModule(train=train, validate=validation, num_workers=0, pin_memory=False)
        model.fit(
            data,
            preset="small",
            accelerator="cpu",
            devices=1,
            precision="32-true",
            batch_size=8,
            max_epochs=1,
            limit_train_batches=1,
            limit_val_batches=1,
            num_sanity_val_steps=0,
            logger=False,
            enable_progress_bar=False,
            enable_model_summary=False,
        )

    predictions = model.predict(pl.DataFrame(fixture["requests"]), retain=("customer_id",))
    for row in predictions.iter_rows(named=True):
        customer = row["inputs"]["customer_id"]
        probability = row["predictions"]["churn_90d"]["content"]["probability"]
        print(f"{customer}: {probability:.3f}")


if __name__ == "__main__":
    main()
