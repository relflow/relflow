<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/starlight/src/assets/branding/header.dark.svg" />
    <source media="(prefers-color-scheme: light)" srcset="docs/starlight/src/assets/branding/header.light.svg" />
    <img alt="relflow" src="docs/starlight/src/assets/branding/header.light.svg" width="360" />
  </picture>
</p>

<p align="center">
  <a href="https://pypi.org/project/relflow/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/pypi/relflow.svg?variant=outline&amp;size=sm&amp;mode=dark" /><img alt="PyPI version" src="https://shieldcn.dev/pypi/relflow.svg?variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <a href="https://www.relflow.ai/latest/"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/docs-read%20online-68b78a.svg?logo=readthedocs&amp;variant=outline&amp;size=sm&amp;mode=dark" /><img alt="Read the documentation" src="https://shieldcn.dev/badge/docs-read%20online-185a3c.svg?logo=readthedocs&amp;variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/ci.yml?query=branch%3Amain"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/ci/relflow/relflow.svg?workflow=ci.yml&amp;branch=main&amp;variant=outline&amp;size=sm&amp;mode=dark" /><img alt="CI on main" src="https://shieldcn.dev/github/ci/relflow/relflow.svg?workflow=ci.yml&amp;branch=main&amp;variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/compatibility.yml?query=branch%3Amain"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/github/ci/relflow/relflow.svg?workflow=compatibility.yml&amp;branch=main&amp;label=compatibility&amp;variant=outline&amp;size=sm&amp;mode=dark" /><img alt="Dependency compatibility on main" src="https://shieldcn.dev/github/ci/relflow/relflow.svg?workflow=compatibility.yml&amp;branch=main&amp;label=compatibility&amp;variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <a href="https://github.com/relflow/relflow/actions/workflows/coverage.yml?query=branch%3Amain"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Frelflow%2Frelflow%2Fbadges%2Fcoverage.json&amp;query=%24.message&amp;label=coverage&amp;variant=outline&amp;size=sm&amp;mode=dark" /><img alt="Line coverage on main" src="https://shieldcn.dev/badge/dynamic/json.svg?url=https%3A%2F%2Fraw.githubusercontent.com%2Frelflow%2Frelflow%2Fbadges%2Fcoverage.json&amp;query=%24.message&amp;label=coverage&amp;variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <picture>
      <source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/pypi/python/relflow.svg?variant=outline&amp;size=sm&amp;mode=dark" />
      <img alt="Required Python version" src="https://shieldcn.dev/pypi/python/relflow.svg?variant=outline&amp;size=sm&amp;mode=light" />
    </picture>
  <a href="LICENSE"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/license-Apache--2.0-68b78a.svg?variant=outline&amp;size=sm&amp;mode=dark" /><img alt="Apache 2.0 license" src="https://shieldcn.dev/badge/license-Apache--2.0-185a3c.svg?variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <!-- discord-invite:start -->
  <a href="https://discord.gg/DVyZUkvTFA"><picture><source media="(prefers-color-scheme: dark)" srcset="https://shieldcn.dev/badge/Discord-Join-68b78a.svg?logo=discord&amp;variant=outline&amp;size=sm&amp;mode=dark" /><img alt="Discord community" src="https://shieldcn.dev/badge/Discord-Join-185a3c.svg?logo=discord&amp;variant=outline&amp;size=sm&amp;mode=light" /></picture></a>
  <!-- discord-invite:end -->
</p>

<p align="center">
  <a href="https://www.relflow.ai/latest/">User Guide</a> ·
  <a href="https://www.relflow.ai/latest/benchmarks/">Benchmarks</a> ·
  <a href="https://www.relflow.ai/latest/api-map/">API map</a> ·
  <a href="https://discord.gg/DVyZUkvTFA">Community</a>
</p>

# Featureless modeling, shaped by your data

relflow is a predictive modeling framework, intended for any predictive task.
You describe your records, their relationships, and the outcomes you want to
predict using a typed Python schema. relflow builds a model from that schema,
then learns features and predictions together during training.

A customer has usage history and support tickets. An asset has sensor readings
and maintenance records. An insurance claim has invoices and line items. Keep
those records and their relationships in the model, without first turning every
history into a hand-crafted table of counts and averages.

## Why relflow?

- **Featureless.** Learn features from typed values and nested records inside the
  model, without a separate feature engineering pipeline.
- **Modular.** Compose models from fields and branches. Extend them with your own
  custom leaf types and connect to your own custom data sources.
- **Scalable.** Stream massive datasets from disk or cloud storage through batched
  training and inference.
- **Flexible.** Predict probabilities, numbers, or several outcomes together.
  Targets can live at the root or inside nested collections of records.
- **Deployable.** Use the trained model for batch prediction and real-time serving,
  with its learned vocabularies and normalizers included.
- **State of the art.** On the nine published RelArena tasks with more than one
  million training observations, relflow beats TabPFN-3 on **9 of 9** and
  TabPFN-3.5 on **6 of 9**. [Inspect the scores and comparison records](https://www.relflow.ai/latest/benchmarks/).

## Philosophy

**relflow was designed to be the first-of-its-class featureless modeling
paradigm, providing a scalable solution in which model architecture embodies
the latent data structure.**

By internalizing feature processing as model components, relflow keeps the same
learned vocabularies and normalizers throughout training and inference. The
typed schema describes both the data contract and the architecture, including
how related records contribute context to each prediction.

## Install

Requires Python 3.12 or newer. With [uv](https://docs.astral.sh/uv/):

```sh
uv add relflow
uv run relflow --version
```

With pip, use `python -m pip install relflow`. Image fields use
`uv add "relflow[image]"`. For text encoders and HTTP serving, use
`uv add "relflow[text,serving]"`. See
[Installation](https://www.relflow.ai/latest/installation/) for the full list.

## Describe your data. Define your predictions.

Suppose you want to predict whether a subscriber will cancel within 90 days.
The model can use their current plan, monthly usage, and support history:

```python
import relflow as rf


class MonthlyUsage(rf.Branch):
    """One month of a subscriber's product usage."""

    month = rf.Period("month_of_year")
    minutes = rf.Number
    data_gb = rf.Number
    overage_fee = rf.Number


class SupportTicket(rf.Branch):
    """One support ticket in a subscriber's history."""

    channel = rf.Category
    resolved = rf.Boolean
    days_open = rf.Number


class ChurnModel(rf.Model):
    """A subscriber with usage and support histories."""

    plan = rf.Category
    tenure_months = rf.Number
    monthly_usage = MonthlyUsage(length=12, overflow="tail")
    support_tickets = SupportTicket(length=32, overflow="tail")
    churn_90d = rf.Boolean(mask=True)


if __name__ == "__main__":
    model = ChurnModel(preset="small")
    data = rf.ArrowDataModule(
        train="customers/train/*.parquet",
        validate="customers/validation/*.parquet",
        test="customers/test/*.parquet",
    )
    model.fit(data, preset="small")
    validation_metrics = model.validate(data)
    test_metrics = model.test(data)
    model.save("churn.rf")
```

**The schema is the architecture.** Fields describe individual values; branches
keep related records together. relflow uses that structure to build its
representations, context encoders, prediction heads, and learning objectives.
Class declarations describe the contract; constructing `ChurnModel(...)` creates
a new, untrained model.

The constructor preset selects architecture defaults. The preset on `fit`
independently selects training defaults, including the optimizer, batch size,
and training budget. The bundled `small` and `medium` presets are alternative
recipes; you do not need to configure those settings individually to get started.

`mask=True` makes `churn_90d` a supervised target: the historical answer trains
the model without becoming an input. Validation and test records also include
observed answers, hidden in the same way for evaluation. New prediction requests
may omit the target. Each history is supplied oldest to newest;
`overflow="tail"` retains the most recent records up to the declared cap. See
[Targets and masking](https://www.relflow.ai/latest/concepts/targets/) for
multiple objectives and targets within nested collections.

The Parquet files contain one subscriber observation per row, with nested usage
and ticket records and the historical `churn_90d` answer. Your application owns
the decision cutoff, labels, and evaluation split. Training streams those files;
the complete dataset does not need to fit in memory. Cloud storage paths use
the same data-module interface.

### Predict for new records

Load the saved model and pass a Polars DataFrame. New request records contain the
same known inputs and a `customer_id` to retain, with the future answer omitted:

```python
import polars as pl
import relflow as rf

model = rf.Model.load("churn.rf")
requests = pl.read_parquet("customers/requests.parquet")
result = model.predict(requests, batch_size=256, retain=("customer_id",))

print(result.select("inputs", "predictions"))
```

The result is an eager Polars DataFrame. Each row retains its customer identifier
under `inputs`; `predictions.churn_90d.content.probability` holds the estimated
probability of cancellation. For larger prediction jobs, stream output to files
with [batch inference](https://www.relflow.ai/latest/advanced/inference-and-deployment/#stream-a-prediction-job-to-files).
Use `rf.PolarsDataModule` when training and validation data are already in memory.

## What can you build with relflow?

The User Guide uses six examples to show how the same typed modeling approach
generalizes across datasets and prediction tasks:

| Use case | Available context | Predictions |
| --- | --- | --- |
| [Support tickets](https://www.relflow.ai/latest/?case=tickets#case-tickets) | Product details and a conversation | Escalation probability and time to resolution |
| [Customer churn](https://www.relflow.ai/latest/?case=churn#case-churn) | Subscriber details, usage, and support histories | Cancellation within 90 days |
| [Demand forecasting](https://www.relflow.ai/latest/?case=demand#case-demand) | Sales history and planned promotions | Units sold for each supplied future week |
| [Credit risk](https://www.relflow.ai/latest/?case=credit#case-credit) | An application, credit accounts, and inquiries | Default within 12 months |
| [Predictive maintenance](https://www.relflow.ai/latest/?case=maintenance#case-maintenance) | Sensor readings and completed work orders | Failure within 30 days |
| [Insurance claims](https://www.relflow.ai/latest/?case=claims#case-claims) | Policies, claims, invoices, and line items | Payout and fraud probability for each claim |

These illustrate the same modeling contract across different data structures,
leaf types, and prediction targets. [Explore the data, model, and class schema](https://www.relflow.ai/latest/featureless/#example)
and [learn about leaf types](https://www.relflow.ai/latest/concepts/data-types/).

## Results you can inspect

**On the nine published RelArena tasks with more than one million training
observations, relflow beats TabPFN-3 on 9 of 9 and TabPFN-3.5 on 6 of 9.**
This is the recorded comparison checked November 10, 2026; the
[benchmark page](https://www.relflow.ai/latest/benchmarks/) includes individual
scores, task sizes, comparison rules, and source records.

A retained Amazon item-lifetime-value model has **926,554 task-trained
parameters**, about **0.42%** of the approximately **220 million** in the
TabPFN-3.5 base checkpoint. Including its shared frozen text encoder, the relflow
model has **23,640,282 parameters**. See the
[model-size comparison](https://www.relflow.ai/latest/benchmarks/#count-the-whole-model)
for exactly what is counted.

The [Evidence library](https://www.relflow.ai/latest/evidence/) explores
individual learning capabilities with controlled comparisons, source code,
measurements, and execution provenance.

<details>
<summary>Selected historical learning checks · relflow 0.4.2</summary>

<!-- evidence-results:start -->

| Experiment | relflow result | Control |
| --- | --- | --- |
| [Match IDs absent from training](https://www.relflow.ai/latest/evidence/P021/) | **0.9991 AUC** with `Hash` | 0.5000 with unknown `Category` labels |
| [Recall a value by an unfamiliar ID](https://www.relflow.ai/latest/evidence/P025/) | **0.100 normalized RMSE**, median of three seeds | 1.401 after corrupting the lookup IDs |
| [Distinguish nulls from zero](https://www.relflow.ai/latest/evidence/P038/) | **100% accuracy** with value states preserved | 50% after filling nulls with zero |

<!-- evidence-results:end -->

These are recorded synthetic tasks, with their original controls and revision.
Each linked page explains the setup, limitations, and retained results.

</details>

## Go further

- [Models and schemas](https://www.relflow.ai/latest/concepts/models/): how typed declarations shape the model.
- [Context branches](https://www.relflow.ai/latest/concepts/branches/): collections, nested records, and learned context.
- [Targets and masking](https://www.relflow.ai/latest/concepts/targets/): hidden answers, multiple objectives, and nested predictions.
- [Data flow](https://www.relflow.ai/latest/concepts/data-flow/): follow an observation from loading to prediction or a training update.
- [Runtime](https://www.relflow.ai/latest/concepts/runtime/): fit, validate, test, and predict with presets.
- [Data modules](https://www.relflow.ai/latest/advanced/data-modules/): streaming files, queries, and custom readers.
- [Custom leaf types](https://www.relflow.ai/latest/advanced/custom-data-types/): extend the same registration system used by built-in fields.
- [Inference and deployment](https://www.relflow.ai/latest/advanced/inference-and-deployment/): batch output and HTTP serving.
- [API map](https://www.relflow.ai/latest/api-map/): find public classes and methods with links to their source.

Join the [Discord community](https://discord.gg/DVyZUkvTFA), contribute through
[CONTRIBUTING.md](CONTRIBUTING.md), or work on the site using the
[documentation guide](docs/README.md).

relflow is free and open source under the [Apache License, Version 2.0](LICENSE)
(`Apache-2.0`). This covers project-authored code, documentation, examples, schema
machinery, cache implementations, and preset definitions unless a file states
otherwise. See [NOTICE](NOTICE) for project attribution and
[licensing](https://www.relflow.ai/latest/license/) for details.
