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
  with live reload. After changing `astro.config.mjs` or the sidebar configuration,
  restart the running server with `npm run dev -- --force --background` so the
  preview loads the new configuration. Keep iterations local; do not push preview commits or trigger
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
- Standalone lowercase `relflow` mentions in page prose use an inline serif
  wordmark. Keep authored text plain: `src/middleware.ts` applies the server-only
  `src/site/brand.mjs` HTML transform in dev and static prerendering. Code,
  commands, URLs, identifiers, exact data viewers, and metadata stay untouched;
  `data-no-brand` opts an element and its descendants out. Preserve original
  text and attributes, heading anchors, selection, and copying. Style the shared
  `.rf-wordmark` in `src/styles/site.css`; the prepared, licensed font subset and
  regeneration command are documented in `branding/README.md`. Do not add
  browser text-replacement scripts or manually wrap each authored mention.
- The palette lab explores accent variations of the current neutral theme.
  Keep surfaces, body text, and semantic string/number colors identical across
  its options; vary only accent and keyword colors. The lab remains a preview.
- Keep the page table of contents disabled globally. Do not add an "On this
  page" sidebar or mobile menu on individual pages.
  Advanced Usage is collapsed by default in the sidebar. The former Use your
  data pages are removed; link to Core Concepts and the Featureless Modeling
  ERD examples instead. Keep the individual leaf type reference pages.
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
- The homepage highlights six use cases using `DataViews` with canonical
  datasets from `src/data/datasets.ts` and schemas from `src/data/use-cases.json`.
  Each use case offers Data, Model, and Schema. Keep all six choices visible,
  natural panel heights, column notes, and accessible keyboard selection.
  Prediction output trees remain separate from the homepage views.
  Shared `ExampleSelection.astro` resolves the initial panel before parsing it.
  Reserve the picker's natural space before enhancement; do not insert it later
  or impose a shared height on example panels. Preserve all-case no-JS content.
- Use `DataViews.astro` as the standard dataset description in authored guides
  and examples. It composes Data, Model, and Schema using one dataset and its
  matching class schema. `DataExplorer.astro` remains the internal diagram
  renderer; do not embed a standalone source diagram in authored pages.
  Explicit custom `leaves` map extension names to their descriptions.
  Supply a page-unique `id` and an authored `Dataset` from
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
  Source descriptions explain field meaning, units, timing, and relationships,
  without model configuration or encoding details. Model ERDs use shared
  representation descriptions and declarations. The shared viewer sets
  `view="model"` internally for model-input and prediction-target labels.
  `src/datasets/` owns shared layout, measured SVG connectors, and column notes.
  Keep instance IDs unique, tooltips independent, both themes, keyboard/touch use,
  Escape, no-JavaScript descriptions, and shared use-case selection working.
- Use `Tree.astro` for conceptual hierarchies, with `label`, optional `detail`,
  and nested `children` on each node. `src/tree/tree.css` owns shared connectors,
  spacing, and native disclosure styling for schema and value trees. Keep rich
  prediction interpretation and charts in `ValueTree` and `ValueNode`.
- Featureless Modeling uses `FeatureEngineeringFlow` for horizontal workflow
  lanes. Keep the steps left-to-right on narrow screens with local scrolling.
  `FeaturelessExamples` switches between the six canonical source ERDs and model
  ERDs in one viewer using `ErdToggle` and the same `DataExplorer` presentation.
  Keep the Data / Model / Schema controls usable with the keyboard and without JavaScript.
  Source-only join keys stay on the data side; model ERDs show the declared
  fields, types, branches, and targets. Preserve the Google Doc's page copy and use its
  visual notes as implementation instructions.
- Core Concepts contains exactly Model trees, Context branches, Leaf types,
  Data flow, Targets and masking, and Runtime. Runtime owns the basic
  `fit`/`validate`/`test`/`predict` workflow and prediction outputs. Start its
  examples with named constructor and fit presets rather than individual
  optimizer, architecture, batch-size, or training-budget settings. Architecture
  and fit presets are independent; `small` and `medium` are alternative recipes,
  not an ascending model-size ladder.
  Model trees, Context branches, Data flow, and Runtime support all six original
  homepage use cases. Targets and masking uses authored prose with one focused
  class schema instead of a use-case explorer. The
  shared scenario records live in `src/data/concepts/scenarios.ts` and reuse the
  canonical homepage schemas. Each concept owns six focused examples in
  `src/data/concepts/<topic>.ts`. Keep shared field names, labels, nesting, and
  target alignment consistent. The shared selector preserves the use case across
  concepts and the homepage, with explicit `?case=` links taking precedence over
  saved preferences. Keep keyboard selection, code notes, natural panel heights,
  native navigation, and readable examples without JavaScript.
- Refer to relflow field representations as leaf types in reader-facing prose.
  The Leaf Types reference follows the shared document's Continuous data,
  Discrete data, Calendar data, Multimodal data, and Other data groups. Keep
  extension guidance visible under Other data; list implemented leaf types and
  retain their actual API names and source paths.
  The Core Concepts Leaf Types page keeps its contracts, states, addresses, and
  `ConceptSummary` explanation without a per-type catalog or use-case explorer.
- `DataFlow.astro` follows the six canonical observations in one workspace with
  a connected vertical stage rail. The rail selects a stage; Back/Continue move
  through the same path, and arrow keys, Home, and End navigate the rail.
  `src/data/concepts/data-flow.ts` owns the authored views;
  `src/data-flow/types.ts` owns prediction/training stage order and the shared
  representation handoffs between stages. Keep the four
  common stages, the prediction output path, and the loss/backward/update path
  distinct. Historical answers align with their supplied records and remain
  hidden from input embedding. Reuse existing prediction illustrations; tensors,
  gradients, and loss values stay schematic. Keep one active stage visible after
  enhancement, with source records and configuration in native disclosures.
  Preserve native stage disclosures as the no-JavaScript fallback,
  stage deep links, case/mode navigation, exact code copying, and both paths
  without JavaScript. Keep fixture data out of client bundles.
- The Model trees page follows the Google Doc’s tree definition, linked
  Branches/Leaves bullets, and Model Presets section with its benchmark callout.
  Keep **Explore a use case** visible and reuse `FeaturelessExamples.astro` with
  `id="models"` for all six canonical cases. Its existing `ErdToggle`
  switches among Data, Model, and Schema everywhere. Diagram views use
  `DataExplorer` column notes; Schema uses the shared `Code` renderer
  with contextual tooltips and exact copying. The schema slot uses the same native radio/CSS controls without JavaScript. Do not substitute the three-step `ModelExplainer`
  inspector or add another concept summary. The same comparison remains on the
  Featureless Modeling page, using its default `featureless` ID prefix.
- Follow the current shared [Google Doc](https://docs.google.com/document/d/1q04H63KEJnykRBGzkKQo86DohyOmMO-4OThlyrFn4ig/edit)
  page by page. Refresh it before starting the next page and keep content review
  incremental. Preserve its intended content, emphasis, and page boundaries.
  Treat layout notes and outline shorthand as authoring instructions. Apply
  light copy edits and normal web formatting; do not replace the message with
  new positioning or insist on verbatim wording unless the user requests it.
  Do not fill unwritten sections with new tutorials; the purchase quickstart was
  deliberately removed. Explain inputs,
  learning, and outputs before architecture details.
- Advanced Usage owns data modules and preprocessing, training and learned state,
  regularization, custom data modules/types, online vocabulary, mutations,
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
- Every individual leaf type page follows the shared Google Doc's Number page:
  a short introduction followed by `Input values`,
  `Standardization and processor pipelines`, `How is this leaf trained?`,
  `Leaf-specific parameters`, `Errors raised`, and `Relevant evidence`, in that
  exact order. Keep `Predictions` under training, `Request Methods` under
  parameters, `Footguns` under errors, and `Related types` under evidence as
  level-three subsections. Options tables use
  `Option`, `Default`, and `Purpose` columns.
  Explain when to choose the type, accepted values and missingness, output
  semantics, and verified evidence scope. The type-selection overview is separate.
  Use `DataTypePage` with the leaf name; keep the six reference sections
  as ordinary Markdown in its body. Predictions retains the authored output
  explanation and adds a standalone `ValueTree` fed by
  `src/datatypes/predictions.ts`, without datasets, schemas,
  or scenario selectors. Keep output values illustrative and field names generic.
  Keep the explanation as prose; use optional display-only tree notes for field
  details. Copied JSON must remain the actual output shape. Hash, Text, and Period
  show requested representations rather than fabricated decoded identifiers,
  text, or dates. The build checks section labels, anchors, and order. Do not
  recreate this layout or introduce a separate slot for each section.
  Request methods tables use `Method`, `Returns`, and `Purpose`, followed by
  short examples calling `rf.Type.method(model, "/field")` with absolute paths.
  Include explicit Python type annotations on result assignments in every
  Request Methods snippet, matching the helper contract and example label types.
  Describe only intended type-specific helpers; exclude validation hooks and
  inherited schema machinery. State explicitly when a type has no such helpers.
  Explain each type's input representation, decoder/readout, and actual content
  loss without executing it. Number, Duration, and Ordinal share raw/log numeric
  modeling; other types own their different normalization and loss behavior.
  Use `LeafEvidence type="Number"` (or another canonical type) for the three-column proof,
  hypothesis, and release-status table. The selected publication owns every
  status; do not copy measured outcomes or versions into these tables. A missing
  run remains missing and a declaration is not a measurement.
- Footguns use concise bullets: a bold, specific mistake, its consequence, an
  actionable check or remedy, and an inline supporting link. Prefer three or
  four useful pitfalls over generic warnings. Draw empirical lessons from the
  retained results and their controls, preserving task and revision scope;
  a passing limitation check is not proof that the model overcame it. Keep
  documented API constraints distinct from measured learning results, and never
  present unmeasured declarations or historical implementations as current proof.
- Leaf reference pages do not embed `TypeExamples.astro`. Its existing scenarios
  in `src/data/examples/<type>.ts` remain available for focused guides; preserve
  their authored records. The shared example contract and rich-result helper
  live in `src/data/type-example.ts`. Generic nested values and display notes
  are typed in `src/values/types.ts`, independent of example records.
  Keep all choices visible and panel heights natural.
  Schema, known inputs, hidden training answers, and results switch together.
  Six-case views use the shared `CasePicker` and `ExampleSelection` primitives;
  preserve six-case persistence and the standalone showcase controller when used.
  Each `TypeExample` owns a required `dataset` matching its input and answer
  source fields. Show `DataViews` first, exact source records in a disclosure, and complete
  prediction output below. The schema lives in its Schema view.
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
- The leaf type reference, schemas, and workflow contracts follow the public
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
  `src/evidence/catalog.ts` owns interrogative display titles, neutral modeling summaries,
  and primary leaf-type tags. The tags drive catalog filters and `LeafEvidence`
  reverse links; do not tag retired Cluster proofs as current Group evidence.
  Lead each report with how the synthetic task is modeled, then one primary
  comparison using `EvidenceComparison`. `src/evidence/comparison.ts` selects
  metric paths, conditions, and units; every plotted number still comes from the
  selected record. Prefer recorded raw RMSE/MAE/MSE over normalized error when
  the same raw quantity exists for all compared arms. Distinguish controlled
  comparisons, baseline references, before/after runs, and recorded conditions;
  never invent a control for a proof that lacks one. Keep check totals and unmet
  verdicts visible, with every criterion, secondary metric, and the complete
  investigation closed by default. Charts must remain readable without scripts.
  Display authored interpretation separately from the CI-derived verdict. Exact
  archived scripts keep their own API and source hashes. Run `npm test` for the input-driven
  adapter checks as well as the normal type check and build.
  The complete raw record uses the compact `JsonRecord.astro` inspector, keeping
  full SSR JSON/copy/downloads without thousands of highlighting spans. Python
  examples and exact source viewers retain the standard code component.
- Do not add compatibility routes or redirect adapters. The root
  `docs.publish` entry point owns Read the Docs publishing and Evidence
  selection. Check narrow layouts, dark mode, search, copying, and links.
