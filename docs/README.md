# relflow documentation

**Build and adapt foundation models for your own structured data.**

[relflow](https://github.com/relflow/relflow) is a model factory for typed,
nested data. Describe your records and prediction targets, and it generates
a custom neural architecture, trains the model, and serves predictions and
learned representations.

This directory owns the documentation for the relflow package in this repository, published at
[www.relflow.ai](https://www.relflow.ai/). Examples start with one payment prediction,
then add nested histories, multiple objectives, pretraining, and reuse.

The site and its assets live under `docs/`; canonical experiments, runners, and
recorded results live under [`evidence/`](../evidence/README.md). Building uses
the isolated static environment in `docs/pyproject.toml` and does not require
installing the relflow package.

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
[Benchmarks](benchmarks.qmd) section presents relflow's published RelArena placement
and comparisons on million-row training tasks. The
[evidence](evidence.qmd) records mechanism experiments;
[Performance](guides/performance.qmd) explains workload-specific
throughput measurement. Predictive comparisons need named tasks, baselines,
metrics, and evaluation protocols.

## Work locally

Install Python 3.12 or newer and uv, then run from the repository root:

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
pass `QUARTO=quarto` to make. Python tooling installs PyYAML 6.0.3, plus pytest
and Ruff for development, into `docs/.venv`, separately from the package
environment. Building never imports relflow, Torch, or Lightning, executes
model examples, trains models, or downloads model weights.

## Write documentation

Follow [AGENTS.md](AGENTS.md) for the authoring contract. Keep introductory
examples small and use the same payment fields throughout a topic. Add details
where readers need them, and link to their canonical explanation.

Use the shared [Typst tree renderer](assets/typst/README.md) for structural
visuals. Branding, fonts, styles, mobile scrolling behavior, and the vendored
Typst extension live here. Regenerating branding is an optional maintenance
task described in [its README](assets/branding/README.md).

## Evidence

Canonical experiments, protocol, hosted runner, and tests live under
[`evidence/`](../evidence/README.md). Scripts own definitions, explanatory prose,
and presentation metadata. Measured results are separate JSON publication
assets; rendering never rewrites or executes experiments.

Generated Evidence pages put the behavioral criteria before the Example and
show recorded metrics as diagnostic figures, plain readouts, labeled groups, and tables. Expanded
details separate Measurements, Method, Reproduce, and Code. Provenance and the
complete raw record remain available in their own disclosures. See the
[presentation contract](../evidence/SPEC.md#documentation) for authoring guidance.
The [diagnostic workflow](../evidence/diagnostics/README.md) retains observations
inside every paired Evidence execution and renders static SVG/JSON in an offline
container. After CI, a frozen publisher retains immutable per-attempt GitHub
Release assets and requests a Read the Docs rebuild. Configure the
`READTHEDOCS_TOKEN` Actions secret once for automatic rebuild requests. PR and
branch previews select the newest exact-commit attempt, showing pending status
until its plots are available. Package releases freeze the matching bundle.

The complete paired evidence gate still runs for PRs, merge groups, and main
CI. It requires valid full-budget results for both exact wheels and blocks
execution errors, incomplete coverage, invalid provenance, and regressions of
baseline-passing checks. Verified reuse keeps the original execution records
and adds a separate receipt binding the current request. See the
[hosted guide](../evidence/hosted.md).

Canonical versioned publication happens **only with package version releases**. Before publishing
`v<version>`, the release workflow freezes a validated `evidence-v<version>`
Release containing `publication.json`, a generated measurements table, complete
report and input archives, checksums, and execution/reuse provenance. Published
versions retain their original selection. Ordinary CI creates no publication
commits or follow-up pull requests.

Local `make check`, `make render`, and preview builds read the preserved
[`evidence/results.json`](../evidence/results.json) bootstrap snapshot, so they
remain offline and independent of model packages. The Read the Docs publisher
fetches the public asset for the version it is building and checks its GitHub
SHA256 digest and package source revision. A missing or invalid asset for a
published version fails the build. Unreleased branches and PRs use the latest
released measurements, or the bootstrap snapshot before the first asset-based
release. Downloaded JSON is cached under `.cache/evidence/`; it never becomes
an experiment-source edit or a new measurement.

Tagged documentation builds require their matching release asset. The `latest`
site reads the latest released selection when it builds; existing HTML changes
only after a rebuild. If its main-push build finished before release publication,
rebuild `latest` from the Read the Docs dashboard after the release completes.

To render a specific released selection locally:

```bash
uv run --project docs python -m evidence.assets --version 0.4.3 --required
RELFLOW_EVIDENCE_PUBLICATION="$PWD/.cache/evidence/evidence-v0.4.3.json" make check-docs
```

The ordinary README measurement table is a labeled historical snapshot. New
release measurements are published as assets and in the versioned docs, without
rewriting that table. See [evidence/README.md](../evidence/README.md) for explicit
execution and release publication commands.

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
Published versions use `https://www.relflow.ai/en/<version>/`, with `latest`
following `main`. Pull request builds keep their separate preview domains and
version URLs. Rebuild affected versions after changing the canonical domain.
Source links use the actual checkout's Git `HEAD`, checked against
`READTHEDOCS_GIT_COMMIT_HASH` when that variable is supplied. The **View source**
action therefore opens the exact commit for a branch, release, or pull request
build. Links and canonical URLs remain within the selected version. Building
remains static and uses no other checkout.

GitHub's `ci.yml` calls the reusable `docs.yml` workflow to validate and upload a
review artifact. It does not deploy the site. Dispatch `docs.yml` manually to
validate documentation on a branch; Read the Docs owns hosting and pull request
previews.

### Hosted project

The Read the Docs project is connected to `relflow/relflow`; the owner is
**`granthamtaylor`**. Its default branch is `main`, its configuration path is
`.readthedocs.yaml`, and its default documentation version is `latest`.
Manage the GitHub connection, pull request builds, Addons, search, and domain
settings in the [project settings](https://app.readthedocs.org/dashboard/relflow/edit/).
Verify hosted changes anonymously, including assets, nested links, evidence
pages, version navigation, and search. Local previews do not receive host
injection.

Gold is enabled, and `relflow` is assigned through the account's
[ad-free projects list](https://app.readthedocs.org/accounts/gold/projects/).
The anonymous hosted Addons response reports `addons.ethicalads.ad_free: true`.
Manage membership through [Gold settings](https://app.readthedocs.org/accounts/gold/);
the project's **Advertising** settings alone do not assign Gold sponsorship.
Verify hosted pages without an ad blocker after changing the assignment.
Gold's project-wide removal is described in the
[advertising policy](https://docs.readthedocs.com/platform/stable/advertising/ethical-advertising.html#opting-out).

### Versions and navigation

Read the Docs provides native [versions](https://docs.readthedocs.com/platform/stable/versions.html).
The default version, `latest`, follows `main`. The enabled
[release activation rule](https://app.readthedocs.org/dashboard/relflow/rules/3813/)
activates and builds new final package tags from `v0.4.5` onward. It applies only
to tags and excludes prereleases, internal evidence tags, and unsupported older
versions. The rule uses this custom match:

```text
^v(?:0[.]4[.](?:[5-9]|[1-9][0-9]+)|0[.](?:[5-9]|[1-9][0-9]+)[.][0-9]+|[1-9][0-9]*[.][0-9]+[.][0-9]+)$
```

Release tags must contain the documentation build configuration and their frozen
evidence publication. Keep `stable` and the current release active so Read the
Docs also advances the stable alias when a newer release arrives. Existing
versions require separate activation; new automation rules do not backfill them.
Keep unsupported historical versions inactive until their documentation can be
built from the appropriate source.

Version activation is a project setting, separate from `.readthedocs.yaml`.
If the header only shows **Latest**, check the project's
[Versions settings](https://app.readthedocs.org/dashboard/relflow/versions/):
an inactive or unbuilt release cannot appear in the picker. Activate a verified
release tag, build it successfully, and check the public version URL before
enabling `stable`. Adding a Git tag alone does not publish its documentation.
Read the Docs caches Addons metadata, so a newly built version can be live before
an existing page's picker lists it. Check the Addons response's `Cache-Control`
and `Age` headers when diagnosing a delayed menu update.

The header shows the current version beside the relflow logo. When multiple
versions are available, it opens a keyboard-accessible menu of published
versions, retaining the current page, query string, and fragment when switching.
Release versions appear as they are activated and built; a single version is
shown as a badge, and pull request previews identify their current PR. If a
page does not exist in an older version, Read the Docs serves that version's
404 page.

The picker consumes the supported [Addons event](https://docs.readthedocs.com/platform/stable/addons.html#custom-event-integration)
and host-provided URLs. It hides the injected
[flyout](https://docs.readthedocs.com/platform/stable/flyout-menu.html) only after
initializing successfully, retaining it as a fallback. Keep the Flyout Menu
addon enabled in the dashboard. Host search and Gold remain independent;
Quarto retains its own search for local and static builds. Do not hardcode a
release list or the host's injected script URL.
Keep **File tree diff** and **Visual diff** disabled in **Settings → Addons**.
Their floating review controls interfere with Quarto's fixed navigation; pull
request previews and the header version picker work without them.
[Pull request previews](https://docs.readthedocs.com/platform/stable/pull-requests.html)
use separate preview domains and carry a status check on GitHub.

### Domains

[www.relflow.ai](https://www.relflow.ai/) is the canonical public address. Its
CNAME points to `readthedocs.io`. The root URL redirects to the default version
at `/en/latest/`. Link to complete version URLs for pages and assets, such as
`https://www.relflow.ai/en/latest/guides/serving.html`; unversioned page and
asset paths have no compatibility aliases. Pull request previews keep their
own domains and version paths.

Squarespace forwards only the apex `relflow.ai` to `https://www.relflow.ai`
with a **301 permanent redirect**, **Maintain paths**, and **Automatic HTTPS**.
Keep the destination scheme explicitly HTTPS: Automatic HTTPS provides the
incoming certificate and does not select the destination scheme. After a
forwarding change, verify that the first redirect from both HTTP and HTTPS
points directly to HTTPS on `www`, preserving a complete version path. Keep
the `www` CNAME and email records intact.

Squarespace's apex forwarding drops query strings. Use direct `www` URLs when
a query string must survive. Direct `www` URLs and the header version picker
retain query strings and fragments.

relflow-authored material retains the [Apache 2.0 license](../LICENSE) and
[NOTICE](../NOTICE). The vendored Typst extension retains its
[MIT license](_extensions/mcanouil/typst-render/LICENSE), and the bundled
font retains its [SIL Open Font License](assets/fonts/OFL.txt).
