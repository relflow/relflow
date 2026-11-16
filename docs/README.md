# relflow documentation

The website is built with Astro and Starlight in [`starlight/`](starlight/README.md).
Pages, examples, components, search, and browser tests live there. Shared package
branding and the licensed wordmark font remain in [`assets/`](assets/branding/README.md).
Canonical learning experiments and recorded results live in
[`evidence/`](../evidence/README.md).

## Work locally

Use Node 24 and Python 3.12 or newer. From the repository root:

```sh
make prepare
make check
make check-docs
make dev
```

Open `http://127.0.0.1:4300/`. `make render` builds static HTML and Pagefind
search in `docs/starlight/dist/`; `make preview` serves that production build.
`make check-docs` also validates built links, canonical URLs, and page/bundle
budgets. To verify browser behavior:

```sh
cd docs/starlight
npx playwright install chromium
cd ../..
make check-browser
```

The site uses its committed npm lockfile. The separate `docs/pyproject.toml`
and `uv.lock` isolate static publisher and Evidence verification tools; they do
not install the SDK. `make check-python` runs those checks independently.
Builds never import relflow, Torch, or Lightning or execute example models.

Follow [`AGENTS.md`](AGENTS.md) and the
[Starlight authoring guide](starlight/AGENTS.md) for conventions. The
[Starlight README](starlight/README.md) describes the reader path, six use cases,
dataset explorer, prediction trees, code notes, API map, and Evidence rendering.

## Publication

Read the Docs builds hosted versions and pull request previews using its
[custom build support](https://about.readthedocs.com/tools/custom/) and the root
[`.readthedocs.yaml`](../.readthedocs.yaml). It installs the locked static tools
and runs:

```sh
uv run --project docs python -m docs.publish
```

The publisher selects verified Evidence assets, builds Starlight, validates
the hosted site, and copies it into `READTHEDOCS_OUTPUT/html`.
`READTHEDOCS_CANONICAL_URL` supplies the complete version URL;
`READTHEDOCS_GIT_COMMIT_HASH`, when supplied, must match the checkout.
Site pages, assets, search, and navigation stay inside that version path.

The canonical public host is `www.relflow.ai`. The `latest` version follows
`main`; release versions and pull request previews are separate host builds.
Use full version URLs when linking publicly, such as
`https://www.relflow.ai/en/latest/concepts/models/`. No old-route aliases are
provided. GitHub's reusable `docs.yml` validates the same site and uploads a
review artifact; it does not deploy production.

Manage the repository connection, active versions, pull request builds, native
version flyout, and Addons in the
[Read the Docs project settings](https://app.readthedocs.org/dashboard/relflow/edit/).
Release activation and the default version are hosting settings, separate from
the repository configuration. Gold ad-free assignment is managed through the
account's [project list](https://app.readthedocs.org/accounts/gold/projects/).
Verify any hosted setting changes anonymously after rebuilding the affected
version; local validation does not establish deployed behavior.

The `www` CNAME points to Read the Docs. Preserve the apex HTTPS redirect and
email DNS records when changing hosting. Direct `www` URLs preserve the full
version path, query string, and fragment. Domain and account changes are
separate from local documentation maintenance.

## Evidence selection

The site consumes one schema-1 publication via `RELFLOW_EVIDENCE_PUBLICATION`.
Local builds use the explicitly historical
[`evidence/results.json`](../evidence/results.json)
bootstrap unless a publication is selected:

```sh
RELFLOW_EVIDENCE_PUBLICATION=/absolute/path/to/publication.json make render
```

Historical downloads are generated from this canonical file, preserving the
existing `/evidence/results.json` and single-proof download contracts. The
selected publication remains a separate, byte-preserved download.

Hosted branch and PR builds show the latest formal relflow release's Evidence.
Tag builds require their own version's frozen publication and exact source
revision. Successful PR checks remain in GitHub Actions and do not replace
published measurements or trigger an Evidence-specific docs rebuild.

The Release workflow freezes verified assets once under `evidence-v<version>`,
with a title such as **relflow 0.5.0 evidence**. Downloads use these names:

- `relflow-0.5.0-evidence.json`
- `relflow-0.5.0-evidence-reports.tar.gz`
- `relflow-0.5.0-evidence-inputs.tar.gz`
- `relflow-0.5.0-evidence-plots.zip`, when captured plots are available
- `relflow-0.5.0-evidence-provenance.json`
- `relflow-0.5.0-evidence-measurements.md`
- `relflow-0.5.0-evidence-checksums.txt`
- `relflow-0.5.0-evidence-reuse.json`, when a verified execution was reused

Earlier published assets retain their original names and bytes. Trusted
selection and verification remain in `evidence/assets.py`,
`evidence/publication.py`, and the diagnostic helpers. Rendering documentation
does not run learning experiments or rewrite measurements.

The complete archived moderation example and its executed synthetic-workflow
receipt are preserved under
[`evidence/archives/moderation-onboarding/`](../evidence/archives/moderation-onboarding/README.md).
They remain historical records, separate from current tutorials and competitive
benchmarks.

relflow-authored material retains the [Apache 2.0 license](../LICENSE) and
[NOTICE](../NOTICE). The shared font retains its
[SIL Open Font License](assets/fonts/OFL.txt).
