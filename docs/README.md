# RelFlow documentation

**Build and adapt foundation models for your own structured data.**

[RelFlow](https://github.com/relflow/relflow) is a model factory for typed,
nested data. Describe your records and prediction targets, and it generates
a custom neural architecture, trains the model, and serves predictions and
learned representations.

This directory owns the documentation for the RelFlow package in this repository, published at
[relflow.ai](https://relflow.ai/). Examples start with one payment prediction,
then add nested histories, multiple objectives, pretraining, and reuse.

The site and its assets live under `docs/`; canonical experiments, runners, and
recorded results live under [`evidence/`](../evidence/README.md). Building uses
the isolated static environment in `docs/pyproject.toml` and does not require
installing the RelFlow package.

## What the documentation teaches

- **Generate the architecture from your schema.** Named fields, nesting, and
  objectives construct the model's encoders, context modules, and decoders.
  Start with [Your first model](learn/your-first-model.qmd) and
  [Model tree](concepts/model-tree.qmd).
- **Describe extensible, nested data.** Built-in and
  [custom field types](datatypes/custom-tensorfields.qmd) represent values
  inside arbitrarily nested records and variable-length collections. Predict
  targets at the root or within those collections.
- **Pretrain, adapt, and reuse.** Learn context through
  [reconstruction pretraining](learn/pretraining-and-fine-tuning.qmd),
  adapt the objectives and fine-tune, or export
  [representations](learn/reusable-representations.qmd) for separate
  predictors. Evaluate transfer across the intended downstream tasks.
- **Evolve and serve trained models.**
  [Schema changes](guides/schema-mutation.qmd) retain compatible learned
  state. Save the resulting model for batch prediction and
  [HTTP serving](guides/serving.qmd) in application-owned infrastructure.

The [use cases](use-cases/index.qmd) formulate forecasting,
recommendations, customer outcomes, and relational prediction. The
[evidence](evidence.qmd) records mechanism experiments;
[Performance](guides/performance.qmd) explains workload-specific
throughput measurement. Predictive comparisons need named tasks, baselines,
metrics, and evaluation protocols.

## Work locally

Install Python 3.12 or newer and uv for this documentation build, then run from
the repository root (the RelFlow runtime itself supports Python 3.11 or newer):

```bash
uv sync --project docs --locked
make dev
```

The preview opens at `http://127.0.0.1:4200`. Edit pages under [docs](index.qmd);
the website project starts at [_quarto.yml](_quarto.yml).

```bash
make check
make check-docs
make render
```

`make check` checks static examples, links, and evidence
reporting. `make check-docs` also renders all pages and checks their links,
anchors, and Typst diagrams. HTML output goes to `docs/site/`.

Quarto is pinned to **1.10.18** through an isolated uv tool invocation and uses
its bundled Typst. To use that exact Quarto version already installed locally,
pass `QUARTO=quarto` to make. Python tooling installs only PyYAML, pytest, and
Ruff into `docs/.venv`, separately from the package environment. Building never
imports RelFlow, executes model examples, trains models, or downloads model
weights.

## Write documentation

Follow [AGENTS.md](AGENTS.md) for the authoring contract. Keep introductory
examples small and use the same payment fields throughout a topic. Add details
where readers need them, and link to their canonical explanation.

Use the shared [Typst tree renderer](assets/typst/README.md) for structural
visuals. Branding, fonts, styles, mobile scrolling behavior, and the vendored
Typst extension live here. Regenerating branding is an optional maintenance
task described in [its README](assets/branding/README.md).

## Evidence

Canonical experiments, the protocol, lab runner, and results live under
[`evidence/`](../evidence/README.md). Each script owns its metadata and current result in frontmatter; `evidence/render.py` generates ignored pages under
`docs/evidence/` directly from local canonical sources. There is no vendored
snapshot or import/export step, and rendering never executes experiments.

The reusable `evidence-lab.yml` workflow runs all 79 experiments against the
library's exact base and candidate wheels on lab over Tailscale. The trusted
workflow and executor remain pinned separately from the candidate suite in
this repository. `CI / required` requires the full library test matrix, static
documentation checks, and evidence before merging into main.
The [lab guide](../evidence/lab.md) describes its OIDC trust and isolation.

`publish-evidence.yml` retains a selected complete run in an immutable Release
in this repository, with JSON results, logs, an HTML report, tested wheels,
runtime requirements, and checksums. It also validates a candidate update to
the experiment frontmatter and uploads a review patch. Apply the patch on a branch
at the recorded repository base and open a normal PR to publish those measurements.
Ordinary builds read the committed results and need no runtime or cloud credentials.

Runtime tests are explicit and excluded from default checks. See
[evidence/README.md](../evidence/README.md) for execution commands. Each experiment publishes one selected full run with its complete provenance. Old Proofs routes are no longer published.

## Publication

Pull requests render and validate once, uploading a review artifact. The
`ci.yml` entry point calls the reusable `docs.yml` workflow; dispatch `docs.yml`
manually to validate a branch's documentation without a PR. Pushes to `main` publish the
validated site through GitHub Pages when the Actions variable
`PUBLISH_PAGES=true` is enabled; forks leave it unset to build without publishing.

The production domain remains [relflow.ai](https://relflow.ai/). To move its
publication from relflow/docs to relflow/relflow, transfer the `relflow.ai` custom
domain in GitHub Pages settings and keep the destination's build source on
GitHub Actions before enabling `PUBLISH_PAGES=true` there. Disable publication
in the previous repository during that cutover. Importing the source does not
change either repository's Pages settings or move the custom domain.

The build does not inspect RelFlow branches or check out another repository.
Local preview and rendering work without enabling publication.

RelFlow-authored material retains the [Apache 2.0 license](../LICENSE) and
[NOTICE](../NOTICE). The vendored Typst extension retains its
[MIT license](_extensions/mcanouil/typst-render/LICENSE), and the bundled
font retains its [SIL Open Font License](assets/fonts/OFL.txt).
