# Starlight documentation

This subtree owns the Astro + Starlight documentation site. Follow these build
and authoring instructions alongside the parent documentation guide.
The user authorized a complete content and design reset. Earlier page structure,
examples, palette, typography, analogies, and components are not requirements.
Preserve the actual public API and the complete scientific Evidence records,
protocol, revisions, controls, seeds, budgets, and provenance.

- Install with `npm ci`; run `npm run format:check`, `npm run check`, `npm test`,
  and `npm run build`. Browser checks use `npm run test:browser` against the built
  site on port 4310; install Chromium with `npx playwright install chromium` first.
  CI runs the same checks. Keep failure traces and screenshots out of source.
  Root Make targets own the shared setup/check/build sequence used by CI;
  npm scripts remain the site-level commands. Keep hosting selection and
  publication validation in `docs.publish`.
  Use `npm run format` for implementation changes; its exclusions preserve raw
  evidence bytes and authored examples. Implementation JavaScript is type checked.
- After building, run `npm run check:performance`. `performance.json` owns
  explicit page/asset budgets; `performance-report/budgets.json` records the
  result. Keep these build-only parsers out of browser imports. Browser tests
  protect first-paint selection geometry, deferred diagram setup, and delegation.
- Use `npm run dev` at `http://127.0.0.1:4300` for page-by-page content review
  with live reload. Keep iterations local; do not push preview commits or trigger
  Read the Docs builds unless the user requests publication. Use `npm run preview`
  when checking the production build. Content iteration does not run Evidence CI.
- Author documentation in `src/content/docs/` as Markdown or MDX. Keep reusable
  presentation in Astro components. Use ordinary fenced code blocks or the
  Starlight `Code` component. Do not execute examples during site builds.
- Prefer positional enum labels: `rf.Enum("low", "normal", "high")`. Unpack
  generated labels with `rf.Enum(*labels)` rather than using `values=` in examples.
- Pine Needle is the selected theme: neutral surfaces with `#185a3c` light
  accents and `#68b78a` dark accents, matching the palette lab's resolved values.
  Theme tokens live in `src/styles/site.css`; code colors live in
  `src/styles/code-themes.mjs`. Keep both modes and Note/Tip boxes consistent.
  Custom components must make an example easier to understand.
- Keep reader content in the shared 48rem column, with one fixed page gutter.
  Use `--rf-panel-padding`, `--rf-picker-padding`, `--rf-picker-gap`, and
  `--rf-section-gap` for repeated component spacing. Do not add wide breakouts,
  negative content margins, or viewport-dependent padding/gap changes. Responsive
  layouts may change columns, orientation, and wrapping; spacing stays fixed.
  Markdown section gaps use rem so responsive heading sizes do not change them.
- The header logo and favicon use the selected soft-canopy mark from the
  `logo-rework` worktree, including its Black pine light ink and matching
  Libertinus Serif Semibold wordmark. Prepared SVGs live in `src/assets/branding/`;
  regeneration and source provenance are documented in `branding/README.md`.
  Keep both theme variants, accessible home-link text, and mobile sizing intact.
- The palette lab explores accent variations of the current neutral theme.
  Keep surfaces, body text, and semantic string/number colors identical across
  its options; vary only accent and keyword colors. The lab remains a preview.
- Keep the page table of contents disabled globally. Do not add an "On this
  page" sidebar or mobile menu on individual pages.
- Sidebar summaries use every page's frontmatter description. The Sidebar
  override wraps the native Starlight component, preserving navigation, group
  state, scroll restoration, and the mobile footer. Hover and focus reveal a
  summary on the page link itself, without extra icons or controls.
  Keep native link navigation, Escape dismissal, movement into the popup,
  and both themes working. Descriptions provide native title hints without JavaScript.
- The standalone API map groups public names and common model methods by task.
  Its curated catalog is `src/data/api-map.ts`; each entry links to an exact
  source definition and an existing guide when available. Preserve the documented
  source scopes, optional serving requirements, search/filter URLs, entry anchors,
  and full content without JavaScript. Never import the Python runtime to build it.
- The homepage highlights six use cases through entity relationship diagrams,
  showing their records, relationships, and prediction targets. Use `DataExplorer`
  with the canonical datasets from `src/data/datasets.ts`; keep schemas, record
  snippets, prediction outputs, and code tabs out of this component. Keep all
  six choices visible, with natural-height diagram panels, column descriptions,
  and accessible keyboard selection. The shared class schemas remain available
  to Core Concepts and other guides.
  Shared `ExampleSelection.astro` resolves the initial panel before parsing it.
  Reserve the picker's natural space before enhancement; do not insert it later
  or impose a shared height on example panels. Preserve all-case no-JS content.
- Use `DataExplorer.astro` as the standard dataset description in authored guides
  and examples. Supply a page-unique `id` and an authored `Dataset` from
  `src/datasets/types.ts`; the renderer does not infer a schema from sample values.
  Keep column descriptions, source types, nested record boundaries, and historical
  target roles accurate. Collections of records get related entities; scalar
  lists such as Set and Vector remain columns. Query source timestamps are actual
  source fields, not fabricated derived-number inputs. Exact records remain
  inspectable and copyable as YAML; prediction results use the shared value tree.
  `src/data/datasets.ts` owns the six source-table ERDs; `src/data/observations.ts`
  reuses their field descriptions for the nested Core Concepts records.
  Mark source-only keys separately from modeled inputs. Only source-table views
  add illustrative joins; nested examples do not need extra PK/FK fields.
  `src/datasets/` owns shared layout, measured SVG connectors, and column notes.
  Keep instance IDs unique, tooltips independent, both themes, keyboard/touch use,
  Escape, no-JavaScript descriptions, and shared use-case selection working.
- Use `Tree.astro` for conceptual hierarchies, with `label`, optional `detail`,
  and nested `children` on each node. `src/tree/tree.css` owns shared connectors,
  spacing, and native disclosure styling for schema and value trees. Keep rich
  prediction interpretation and charts in `ValueTree` and `ValueNode`.
- Every Core Concepts page supports all six original homepage use cases. The
  shared scenario records live in `src/data/concepts/scenarios.ts` and reuse the
  canonical homepage schemas. Each concept owns six focused examples in
  `src/data/concepts/<topic>.ts`. Keep shared field names, labels, nesting, and
  target alignment consistent. The shared selector preserves the use case across
  concepts and the homepage, with explicit `?case=` links taking precedence over
  saved preferences. Keep keyboard selection, code notes, natural panel heights,
  native navigation, and readable examples without JavaScript.
- `DataFlow.astro` follows the six canonical observations through connected,
  expandable stages. `src/data/concepts/data-flow.ts` owns the authored views;
  `src/data-flow/types.ts` owns prediction/training stage order. Keep the four
  common stages, the prediction output path, and the loss/backward/update path
  distinct. Historical answers align with their supplied records and remain
  hidden from input embedding. Reuse existing prediction illustrations; tensors,
  gradients, and loss values stay schematic. Preserve native disclosures,
  stage deep links, case/mode navigation, exact code copying, and both paths
  without JavaScript. Keep fixture data out of client bundles.
- The Model trees page uses `ModelExplainer.astro` and `src/data/model-explainer.ts`
  to derive three steps from all six canonical schemas: one answer, added
  context, and another answer. Keep additional targets' historical-label
  requirements explicit and their dataset columns aligned. Validate projections
  and source-line/path bindings through `src/model-explainer/`; each case shares
  one highlighted `Code` block, with exact selected-step copying. Preserve
  complete schemas and readable no-JavaScript explanations. The explainer is
  static and executes no model or training code; it creates no learning evidence.
- Follow the current shared [Google Doc](https://docs.google.com/document/d/1q04H63KEJnykRBGzkKQo86DohyOmMO-4OThlyrFn4ig/edit)
  page by page. Refresh it before starting the next page and keep content review
  incremental. Preserve its intended content, emphasis, and page boundaries.
  Treat layout notes and outline shorthand as authoring instructions. Apply
  light copy edits and normal web formatting; do not replace the message with
  new positioning or insist on verbatim wording unless the user requests it.
  Do not fill unwritten sections with new tutorials; the purchase quickstart was
  deliberately removed. Explain inputs,
  learning, and outputs before architecture details.
- Advanced Usage owns custom data modules/types, online vocabulary, mutations,
  embeddings, distributed data, checkpoints and state, masking/pretraining, and
  inference/deployment. Keep basic predictions in Core Concepts. Mutations and
  embeddings retain the six-case selector through `ConceptPage.astro`, with
  lesson data in `src/data/advanced/`. Other advanced pages may use a focused
  example. Verify extension protocols and lifecycle claims against named source
  revisions and contract tests; distinguish static example checks from executed
  training, serving, accelerator, or learning validation. Do not run those
  workloads while authoring or building the site.
- Contextual code notes apply automatically to ordinary code fences and `Code`.
  Edit shared explanations in `src/reference.mjs`; keep matching, lexical
  context, ranges, and field binding in `src/code/config-notes.mjs`. The central
  reference also owns fixed model-inspector representation definitions. Keep
  page summaries in frontmatter and dataset-specific meanings with their data.
  Review matched ranges. The lexical context distinguishes model constructors and fit settings;
  unrelated Python calls, comments, and strings must not receive model notes.
  Shared popup behavior lives in `src/ui/tooltip.ts`, with typed code, sidebar,
  and dataset adapters. Keep one active popup and restore original accessible
  descriptions when it closes. Notes must preserve source copying, syntax highlighting, layout,
  hover-to-popup movement, keyboard focus, tap, and Escape dismissal.
  Delegate trigger events rather than binding handlers per token. Initialize
  diagrams near the viewport, pause hidden/offscreen resize work, and batch
  geometry reads separately from SVG writes.
- Every individual data type page follows one contract: a short introduction,
  then `Example`, `Inputs`, `Predictions`, `Options`, `Request Methods`,
  `Footguns`, `Related types`, and `Evidence`, in that exact order. Options tables use
  `Option`, `Default`, and `Purpose` columns. Examples use class schemas with
  a one-line docstring.
  Explain when to choose the type, accepted values and missingness, output
  semantics, and verified evidence scope. The type-selection overview is separate.
  Use `DataTypePage` with the existing example ID and typed examples. It owns
  the Example heading and example composition; keep the remaining seven sections
  as ordinary Markdown in its body. The build checks their labels, anchors, and
  order. Do not recreate this layout or introduce a separate slot for each section.
  Request methods tables use `Method`, `Returns`, and `Purpose`, followed by
  short examples calling `rf.Type.method(model, "/field")` with absolute paths.
  Include explicit Python type annotations on result assignments in every
  Request Methods snippet, matching the helper contract and example label types.
  Describe only intended type-specific helpers; exclude validation hooks and
  inherited schema machinery. State explicitly when a type has no such helpers.
- Footguns use concise bullets: a bold, specific mistake, its consequence, an
  actionable check or remedy, and an inline supporting link. Prefer three or
  four useful pitfalls over generic warnings. Draw empirical lessons from the
  retained results and their controls, preserving task and revision scope;
  a passing limitation check is not proof that the model overcame it. Keep
  documented API constraints distinct from measured learning results, and never
  present unmeasured declarations or historical implementations as current proof.
- Every type page uses the shared `TypeExamples.astro` showcase with three
  scenarios in `src/data/examples/<type>.ts`. The shared example contract and
  rich-result helper live in `src/data/type-example.ts`.
  Keep all choices visible and panel heights natural.
  Schema, known inputs, hidden training answers, and results switch together.
  Six-case views use the shared `CasePicker` and `ExampleSelection` primitives;
  preserve the separate six-case persistence and datatype-scenario controllers.
  Each `TypeExample` owns a required `dataset` matching its input and answer
  source fields. Show the explorer first, the model schema next, exact source
  records in a disclosure, and complete prediction output below.
  Show output as a complete prediction row through `ValueTree.astro` with
  `predictions` enabled. Keep actual field names, branch lists, and source-record
  alignment. Visualize Boolean event probabilities, category candidates, independent
  Set memberships, scalar values, and vector coordinates. Do not invent scalar
  ranges, uncertainty intervals, confidence for Ordinal labels, or normalized Set
  distributions. Full `state`/`content`/`inferred` and extension fields stay
  inspectable in Returned fields; state.valued is not answer confidence.
  Preserve nulls, empty lists, original scalar types, and exact full-row JSON
  copying. Generic nested values use the same component without `predictions`;
  they must not be reinterpreted as rich outputs based on field names alone.
  Source inputs, training answers, and complete training records remain YAML.
  These presentations do not change the model's prediction-table return type.
  Illustrative outputs are not executed predictions or evidence. Nested answers
  and outputs align with the supplied records; Set probabilities are independent.
  Preserve code notes, linked field highlighting, copying, keyboard/touch use,
  deep links, and readable content without JavaScript.
- Input-only types demonstrate useful predictions on other fields. Do not
  invent string, identifier, or datetime generation for Text, Hash, or Period.
  Group examples use visible identity inputs and downstream targets; its decoded
  group head has no built-in group reconstruction objective. Duration examples
  may supply timestamp sources from which target values are derived; explain
  that transformation in `answerNote` rather than inventing numeric source data.
  Ordinal uses keyword `values=(...)` and writes native labels or integer ranks.
- The datatype reference, schemas, and workflow contracts follow the public
  source at `a1d0be3e373cf6c183e4926f1bfcbef845f2f883` on stage/0.5.0
  (13 canonical types).
  Compare source exports, registered extensions, pages, examples, and navigation
  together before declaring coverage complete. Use Period for calendar fields.
  Preserve the selected historical Evidence snapshot.
- Prefer semantic HTML and CSS for small explanatory visuals. Builds use the
  prepared branding SVGs and require no diagram compiler.
- Recorded evidence retains its measured revision and protocol, including unmet
  checks and absent measurements. Tutorial data demonstrates a workflow and does
  not establish real application quality or competitive accuracy.
- Evidence results come from one schema-1 CI publication selected with
  `RELFLOW_EVIDENCE_PUBLICATION`; local builds may use the explicitly labeled
  historical bootstrap. Never hardcode counts, check outcomes, or readout values
  in pages or a separate catalog. Pending selections stay pending; explicit bad
  inputs fail. Preserve all raw metrics, named checks, and provenance.
  Validate the fields consumed by presentation. Results outside the current
  declaration catalog require exact archived source bytes, a matching script
  digest and ID, and their recorded original path/lane identity; reject unknown
  identities rather than silently omitting measurements or reusing another proof.
  Display authored interpretation separately from the CI-derived verdict. Exact
  archived scripts keep their own API and source hashes. Run `npm test` for the input-driven
  adapter checks as well as the normal type check and build.
  The complete raw record uses the compact `JsonRecord.astro` inspector, keeping
  full SSR JSON/copy/downloads without thousands of highlighting spans. Python
  examples and exact source viewers retain the standard code component.
- Do not add compatibility routes or redirect adapters. The root
  `docs.publish` entry point owns Read the Docs publishing and Evidence
  selection. Check narrow layouts, dark mode, search, copying, and links.
