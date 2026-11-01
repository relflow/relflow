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

Read the Docs builds and hosts each active documentation version from this
repository using [`.readthedocs.yaml`](../.readthedocs.yaml). Its custom build
installs the isolated static tools and runs:

```bash
uv run --project docs python -m docs.publish
```

The publisher validates sources, renders with pinned Quarto, checks the
rendered site, and copies the validated HTML to `READTHEDOCS_OUTPUT/html`.
`READTHEDOCS_CANONICAL_URL` supplies the full URL of that hosted version.
Source links use the actual checkout's Git `HEAD`, checked against
`READTHEDOCS_GIT_COMMIT_HASH` when that variable is supplied. The **View source**
action therefore opens the exact commit for a branch, release, or pull request
build. Links and canonical URLs remain within the selected version. Building
remains static and uses no other checkout.

GitHub's `ci.yml` calls the reusable `docs.yml` workflow to validate and upload a
review artifact. It does not deploy the site. Dispatch `docs.yml` manually to
validate documentation on a branch; Read the Docs owns hosting and pull request
previews.

### Connect the project

The Read the Docs owner is **`granthamtaylor`**. The source configuration does not
create a hosted project or configure the account. Complete these steps in that
owner account:

1. Sign in to [Read the Docs Community](https://app.readthedocs.org/), install
   its GitHub App for `relflow/relflow`, and grant that repository access.
   Choose **Projects → Add project**, select `relflow/relflow`, and retain the
   repository URL. Follow the [GitHub connection guide](https://docs.readthedocs.com/platform/stable/reference/git-integration.html).
2. Create the project using the connected repository's default branch,
   `main`, and the root configuration path `.readthedocs.yaml`. The import
   wizard does not expose a branch override. Before this migration merges,
   the initial `latest` build will fail because `main` lacks that file.
3. Open **Versions**, include inactive versions, and find the branch
   `stage/0.4.2` (normally the URL identifier `stage-0.4.2`). Activate that
   version and save; activation triggers its build. Inspect the build and its
   **View docs** link. The project branch selector only lists discovered
   branches, so changing the default branch is unnecessary for this preview.
   Read the Docs synchronizes versions on repository pushes and builds; if
   the branch is missing, resolve version discovery before retrying the
   selector. See the [version synchronization reference](https://docs.readthedocs.com/platform/stable/versions.html#version-syncing).
4. In **Settings → Pull request builds**, enable previews. In
   **Settings → Addons → Flyout Menu**, enable the version menu; in
   **Settings → Search**, enable the search modal. Enable the pull request
   notification under **Settings → Addons → Notifications**. Verify these
   controls on hosted pages; the local preview does not receive host injection.
5. After this branch merges, confirm the project's **Default branch** is `main`
   and rebuild `latest`. Keep `latest` as the default documentation version
   during setup. In **Versions**, deactivate an unsupported `stable` alias and
   leave older tags inactive. Activate a release tag only after verifying that
   it contains the consolidated docs and root build configuration. Backfilling
   older release documentation is a separate change.
6. Verify a hosted branch or pull request preview, including assets, nested
   links, evidence pages, version navigation, and search. Initial connection
   leaves DNS and the existing production deployment unchanged.

Gold is **not yet enabled**. The owner will enable the agreed **$5/month**
membership later through [Gold settings](https://app.readthedocs.org/accounts/gold/),
then select the Gold-backed **Ad-free** option in the project's **Advertising**
settings. Verify project-wide removal on hosted pages without an ad blocker at
that time. Account subscription and billing are outside this source change.
Gold's project-wide removal is described in the
[advertising policy](https://docs.readthedocs.com/platform/stable/advertising/ethical-advertising.html#opting-out).

### Versions and domain cutover

Read the Docs provides native [versions](https://docs.readthedocs.com/platform/stable/versions.html):
`latest` follows the configured default branch, release tags get their own
versions, and `stable` normally follows the greatest stable semantic release.
Pre-release tags do not become the automatic `stable` version.

At migration time, GitHub already has tags through `v0.4.11`, while the latest
published GitHub Release is `v0.4.1`. Those older tags lack this consolidated
build configuration, so automatic `stable` selection may point to an unsupported
tag. Keep the default version on `latest`, deactivate an unsupported `stable`,
and leave unsupported tags inactive. Do not enable automatic activation of all
semantic tags. Activate and build only verified supported releases in the
**Versions** tab.

Select `stable` as the default only after its target contains the consolidated
docs and builds successfully. If version ordering continues to select an old
unsupported tag, a separately reviewed branch named `stable` can define a custom
stable target until release numbering supersedes those tags. This migration does
not retag or delete existing tags.

The host injects its [flyout menu](https://docs.readthedocs.com/platform/stable/flyout-menu.html)
with available versions and current-version search. Quarto retains its own
search for local and static builds. Configure host features in the dashboard;
do not add a second version selector or hardcode Read the Docs scripts into
pages. [Pull request previews](https://docs.readthedocs.com/platform/stable/pull-requests.html)
use separate preview domains and carry a status check on GitHub.

The production domain remains [relflow.ai](https://relflow.ai/) during setup;
the production switch is pending. A later cutover requires verified hosted
previews and ad-free behavior, custom-domain setup on Read the Docs, redirects
for existing unversioned page URLs, and its provided DNS instructions. Retire
the previous GitHub Pages deployment as part of that verified cutover. Source
configuration alone does not move the domain.

The built-in `/` redirect handles the root URL; it does not preserve links such
as `/guides/serving.html`. For the later cutover, the authorized project account
must manually add these ten **Exact Redirects** in the dashboard, with **Force
redirect disabled** for every entry:

| From URL | To URL |
| --- | --- |
| `/learn/*` | `/page/learn/:splat` |
| `/use-cases/*` | `/page/use-cases/:splat` |
| `/guides/*` | `/page/guides/:splat` |
| `/concepts/*` | `/page/concepts/:splat` |
| `/datatypes/*` | `/page/datatypes/:splat` |
| `/reference/*` | `/page/reference/:splat` |
| `/evidence/*` | `/page/evidence/:splat` |
| `/evidence.html` | `/page/evidence.html` |
| `/index.html` | `/` |
| `/assets/*` | `/en/latest/assets/:splat` |

The built-in `/page/` prefix resolves pages in the default version. The asset
alias retains existing root asset URLs used by README branding and diagrams.
These ten rules fit within Community's 100-redirect limit. Do not add a `/*`
catchall: it could intercept native version URLs or create redirect loops.
Verify representative old URLs from every directory, the evidence catalog,
the index, and public assets on the regular hosted project domain before any
DNS change; redirects do not apply on pull request preview domains. See the
[Read the Docs redirect reference](https://docs.readthedocs.com/platform/stable/user-defined-redirects.html).
These dashboard changes remain a later manual task.

RelFlow-authored material retains the [Apache 2.0 license](../LICENSE) and
[NOTICE](../NOTICE). The vendored Typst extension retains its
[MIT license](_extensions/mcanouil/typst-render/LICENSE), and the bundled
font retains its [SIL Open Font License](assets/fonts/OFL.txt).
