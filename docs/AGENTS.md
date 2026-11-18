# relflow documentation guide

The website lives in `starlight/`. Follow the root [agent guide](../AGENTS.md)
and [Starlight authoring guide](starlight/AGENTS.md). The repository also owns
shared branding/font sources and an isolated Python publisher. Canonical learning
experiments, their protocol, runtime tests, and recorded results live in
[`evidence/`](../evidence/AGENTS.md).

- Write the project name as `relflow`. Center enterprise data scientists while
  keeping the product explanation accessible to business and engineering readers.
  Explain the supplied observations, learned model, and prediction outputs before
  architecture details. Describe current open-source capabilities; keep Studio
  plans and proprietary promises out of the documentation.
- Prefer class-based `rf.Model` and `rf.Branch` schemas with short record
  docstrings. Keep declaration, model construction, and fitting distinct. Explain
  how typed fields and nested records shape model components; substantiate
  performance claims with their precise recorded task, baseline, and revision.
- Pages live in `starlight/src/content/docs/`; reusable components and example
  contracts live beside them under `src/`. The Starlight guide owns the six-case
  selector, dataset explorer, prediction tree, contextual code notes, consistent
  leaf type sections, Pine Needle palette, spacing, and accessible interactions.
- Use Node 24 (minimum 22.23.4) for the locked Astro/Starlight environment.
  `make prepare` installs both static tool environments. `make check` validates
  source contracts, types, formatting, and static publisher/Evidence tooling.
  `make check-docs` builds HTML/search, validates links and canonical URLs,
  and enforces page/bundle budgets.
  `make check-browser` additionally checks browser interactions after Chromium
  is installed. `make dev` previews on port 4300.
- Builds remain static. Never install/import relflow, Torch, or Lightning,
  execute tutorial models, download model weights, or run learning experiments
  while building or checking documentation. Python publisher tests live in
  `docs/tests/`; site tests live in `starlight/tests/`.
- `pyproject.toml` and `uv.lock` isolate the Python publisher and static Evidence
  verification from the SDK environment. Preserve them: trusted Evidence and
  release workflows also use this environment. Protocol tests belong in
  `evidence/tests/`; explicit runtime checks remain in `evidence/tests/runtime/`.
- The site consumes the schema-1 publication selected by
  `RELFLOW_EVIDENCE_PUBLICATION`. Without a selection, local measurements remain pending.
  Missing results remain missing; authored prose never supplies measurements.
  Preserve exact scripts, checks, metrics, seeds, budgets, and provenance. Builds
  must not modify experiment sources or records.
- Read the Docs uses `.readthedocs.yaml` and `python -m docs.publish` to select
  released Evidence assets and build Starlight for its exact hosted version URL.
  Preserve checkout-revision checks and digest-verified release selection.
  Branch and PR docs use the latest formal release at version >=0.5.0, or show
  pending measurements until one exists; tag builds require the
  matching version and source revision. Never select PR-head CI measurements
  for hosted docs or publish per-commit Evidence releases. Preserve static
  validation before copying to `READTHEDOCS_OUTPUT/html`. Keep links, assets,
  canonical URLs, search, and browser navigation within the hosted version.
  GitHub documentation CI validates and retains a review artifact; local changes
  do not publish the production site.
- Shared package branding remains in `assets/branding/`; the font and its OFL
  remain in `assets/fonts/`. The website uses its selected prepared SVGs under
  `starlight/src/assets/branding/`; its authoring lives in
  `assets/branding/site/`. Optional branding regeneration is separate
  from normal builds. Downloaded diagnostic bundles may stage under the ignored
  `assets/evidence-plots/` path; their verification remains owned by Evidence.
- Update public README links when routes change. Do not add old-route aliases.
  Verify the production build, search, exact copying, keyboard/touch use, both
  themes, narrow layouts, and no-JavaScript reading when changing components.
  Technical checks do not establish reader comprehension: use unfamiliar
  enterprise DS readers to validate whether they can explain the product and
  complete the onboarding tasks without coaching.
