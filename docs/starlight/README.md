# relflow documentation

A fresh reader path built with Astro and Starlight. The site introduces the
modeling job, gets a reader through a first training workflow, and helps them
formulate predictions on their own data. It preserves the scientific Evidence
and benchmark records while replacing the previous presentation.

## Run

The [operations guide](../README.md#work-locally) owns full setup, the canonical
Make targets used by CI, and hosting. For site-only authoring, use Node.js
22.23.4 or newer and npm; Node 24 is recommended.

```sh
cd docs/starlight
npm ci
npm run check
npm test
npm run format:check
npm run build
npm run check:performance
npx playwright install chromium
npm run test:browser
npm run preview
```

Open `http://127.0.0.1:4300`. Use `npm run dev` for live authoring.
Keep page-by-page content review on this local server with live reload;
Read the Docs preview builds are separate publication work.
Set `ASTRO_TELEMETRY_DISABLED=1` in restricted build environments.
`package-lock.json` pins the dependencies.

`npm run test:browser` starts its own production preview on port 4310 and checks
desktop and touch layouts in both themes. Build first. CI installs Chromium with
its system dependencies and runs these same checks in the documentation workflow;
Read the Docs publishing uses the same site. Browser failures retain
screenshots, traces, and an HTML report. An existing Chromium installation can be
selected locally with `PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH`.

`npm run check:performance` checks the built homepage, tutorials, six-case pages,
and large Evidence pages against `performance.json`. Limits cover raw/compressed
HTML, parsed element counts, initial JavaScript, stylesheets, and individual
JavaScript chunks. Build-only parsers follow actual static imports, deduplicate
assets, and count inert template contents. The report is saved to
`performance-report/budgets.json` and uploaded by CI. Gzip sizes are reproducible
build estimates, not measurements of hosting compression. Revise a budget
deliberately when justified by content changes and browser measurements.

The homepage renders all six ERDs and their column descriptions in HTML so the
examples remain readable without JavaScript. Its 26 KiB gzip HTML budget allows
about 20% headroom over the 21.8 KiB ERD version; its HTML, element, JavaScript,
and stylesheet limits remain unchanged.

The Model trees page has a larger HTML budget for six interactive construction
walkthroughs and their complete no-JavaScript references. Each case reuses one
highlighted schema for all three steps; copying derives from those same source
lines. Its page budget allows about 20% headroom over the current 466 KiB HTML,
65 KiB gzip, and 4,100 elements. Shared JavaScript and stylesheet limits remain
unchanged.

Data Flow renders the six cases and both paths as static content, with one
stage expanded after enhancement. Its page budget allows about 20% headroom
over the initial 585 KiB HTML, 66 KiB gzip, and 9,019 elements. Its controller
imports only the small stage-path definitions; schemas, datasets, and prediction
fixtures stay out of the browser bundle. Shared JavaScript/CSS budgets still apply.

Browser regressions delay enhancement modules until after first paint to check
picker geometry and layout stability. They also verify delegated interaction
handlers, visible-only diagram setup, and complete JSON copying/no-JavaScript
reading for Evidence inspectors. These checks complement the build budgets;
they do not establish real-device or deployed performance guarantees.

Use `npm run format` for implementation and test formatting. Immutable public
records, archived evidence sources, and authored content are excluded. The type
check includes implementation JavaScript as well as TypeScript and Astro.

The production build includes Pagefind search. Starlight provides the
navigation, light/dark selection, and copyable code examples. The page table of
contents is disabled globally, including its mobile control.
Building the site executes no model code and requires no Python runtime or
diagram compiler. The separate hosted publisher uses Python for trusted
Evidence selection; see [the publication guide](../README.md).

The selected theme is **Pine Needle**, with neutral surfaces, green links and
code keywords, and the existing green/amber syntax colors. Light and dark
accents are `#185a3c` and `#68b78a`, matching the palette lab's resolved colors.

The local [palette lab](http://127.0.0.1:4300/palettes/) compares accent variations
of the current theme on identical samples and a full documentation preview.
Every option uses the same neutral surfaces, typography, body text, and semantic
syntax colors. Only accent and keyword colors vary, with a light/dark pair for
each option. Its source
is `src/pages/palettes.astro`, `src/palettes/`, and `src/data/palettes.ts`.
Palette exploration is separate from the selected documentation theme.

The header and favicon use the selected soft-canopy logo from `logo-rework`,
with matching light/dark SVGs and the serif wordmark. See `branding/README.md`
for source provenance and regeneration. Normal builds use the prepared vectors.

The sidebar shows every page's frontmatter `description` when its link is hovered
or focused with the keyboard. Links retain their native navigation behavior,
without extra icons or controls. `Sidebar.astro` wraps the native Starlight
sidebar; `src/sidebar/` owns the interaction and layout, using the shared tooltip
styles. Descriptions also provide native title hints without JavaScript.

## Reader path

| Route | Reader's question |
| --- | --- |
| `/` | What does relflow do, and where does it fit? |
| `/featureless/` | How does relflow learn from records without a separate feature engineering pipeline? |
| `/modular/` | How do custom data types, data sources, and specialized model components fit together? |
| `/installation/` | How do I install relflow and the optional capabilities I need? |
| `/data/` | What is one example in my application? |
| `/datasets/` | How do the example datasets, entities, and columns fit together? |
| `/predictions/` | Which answers should my model learn? |
| `/data-types/` | Which representation fits each value? |
| `/concepts/models/` | How do the core ideas fit my use case? |
| `/concepts/data-flow/` | How do source records become predictions or training updates? |
| `/advanced/` | How do I extend, scale, adapt, and deploy the modeling workflow? |
| `/api-map/` | Which class, method, or extension contract handles this job? |
| `/benchmarks/` | How does it compare on published tasks? |
| `/evidence/` | What capabilities have controlled experiments tested? |

## Author

Pages live in `src/content/docs/` as Markdown or MDX. Reusable presentation
belongs in `src/components/`. `src/styles/site.css` contains the small site
customizations; code colors live in `src/styles/code-themes.mjs`.

### Component library

Keep explanations in ordinary Markdown. Use these domain components to render
the repeated teaching patterns; their typed inputs own the relevant contracts.

| Reader task | Component | Authored input |
| --- | --- | --- |
| Learn one datatype | `DataTypePage` | A stable example ID, three `TypeExample` records, and the seven reference sections as Markdown. |
| Follow a concept across use cases | `ConceptPage` | A `Concept` with six focused examples, principles, and related guides. |
| Follow records through the pipeline | `DataFlow` | The six canonical cases, typed stage views, and prediction/training paths in `src/data/concepts/data-flow.ts`. |
| Understand supplied records | `DataExplorer` | A page-unique ID and a `Dataset` with explicit entities, columns, and relationships. |
| Explain a hierarchy | `Tree` | An accessible label and nodes with `label`, optional `detail`, and nested `children`. |
| Read nested predictions | `ValueTree` | The complete result row, known inputs, and the `predictions` flag. |
| Inspect how a schema shapes a model | `ModelExplainer` | A page-unique ID and one of the six canonical case IDs. |
| Show source code with contextual help | Markdown fences or Starlight `Code` | Unmodified source, language, and an optional contract annotation. |

`DataTypePage` supplies the anchored **Example** heading and the interactive
`TypeExamples` view. Its Markdown body contains **Inputs**, **Predictions**,
**Options**, **Request Methods**, **Footguns**, **Related types**, and **Evidence**,
once each in that order. The build validates the rendered section contract;
authors keep Markdown headings and code fences rather than filling seven named
slots. Datatype meaning, examples, and documented limitations remain authored.

`UseCases`, `ConceptPage`, and `DatasetExplorer` share `CasePicker` and one
choice shape: `{ id, panel, label, summary }`. `ExampleSelection` serializes only
the IDs and panel bindings needed before first paint. These primitives retain
the six-case URL/storage controller; the datatype scenarios retain their own
hash selection and field highlighting. Compose the domain components above
when authoring a page instead of recreating their picker or layout.

The sections below describe the data contracts and examples for each component.

Reader content shares one 48rem column and a fixed 1rem page gutter. Common panel
padding, picker padding/gaps, and section spacing use the `--rf-panel-padding`,
`--rf-picker-padding`, `--rf-picker-gap`, and `--rf-section-gap` tokens in
`site.css`. Responsive layouts can stack or wrap, but keep those spacing values
fixed. Avoid wide component breakouts and viewport-specific padding changes.
Markdown heading gaps use rem independently of responsive heading sizes.

The homepage's six use cases render through `UseCases.astro` as entity relationship
diagrams from `src/data/datasets.ts`. Labels and task descriptions come from
`src/data/use-cases.json`; its class schemas remain available to other guides.
Keep schemas, record snippets, prediction outputs, and code tabs out of the
homepage examples. All six choices remain visible; the selected diagram uses
its natural height, without reserving space for longer examples. Column
descriptions retain hover, keyboard, touch, and no-JavaScript access.
Examples show the range of data structures and prediction targets. Benchmark
records provide the measured predictive comparisons.

### Core Concepts

The Core Concepts pages each include all six homepage use cases. Choose a case
once and follow it through schemas, data types, branches, masking, data modules,
training, regularization, and predictions. Mutations and embeddings live in
Advanced Usage and also support all six cases. The selector remembers the
choice and carries it through concept and advanced links; an explicit
`?case=claims` URL or example anchor can select a different case. It also
connects to the homepage.

`ExampleSelection.astro` prepares the initial case inline before panels parse,
using the same hash/query/saved-choice precedence as the interactive selector.
The picker reserves its natural grid space while its controls wait for handlers,
so initialization does not insert a new row. Without JavaScript, all cases
remain readable. Example panels retain their natural heights.

`src/data/concepts/scenarios.ts` owns the shared illustrative records and reuses
the canonical class schemas in `src/data/use-cases.json`. Each topic in `src/data/concepts/` owns six focused
examples and its explanation. `ConceptPage.astro` renders them using the same
contextual code notes as the rest of the site. Examples are static illustrations,
not executions or measured results. All six remain readable without JavaScript.
`src/concepts/selection.ts` owns keyboard behavior, URL selection, saved preference,
and native concept-link updates.

`/concepts/models/` adds `ModelExplainer.astro`: select a field, branch, or target
to inspect its representation, context, learning objective, and output location.
`src/data/model-explainer.ts` derives all six cases from the canonical schemas.
`src/model-explainer/schema.ts` projects three steps: one answer, added context,
and another answer. Additional example targets explicitly describe the new
historical labels they require; their dataset columns stay aligned.

`src/model-explainer/code.mjs` validates projected source and maps declarations
to schema paths. Each case shares one highlighted `Code` block across its steps;
visible lines and copied source follow the selected step. Keep complete schemas
and readable fallback explanations available without JavaScript. This is a
static explanation of the declarations: builds execute no model or training
code, and the diagrams establish no measured performance result.

### Advanced Usage

`/advanced/` organizes ten implementation guides: custom data modules, custom
data types, online vocabulary, mutations, embeddings, distributed training and
streaming, checkpoints and model state, masking and pretraining, inference
and deployment, and the command line. Keep Core Concepts focused on the basic modeling contract;
these guides own extension protocols and operational lifecycle details.

Mutations and embeddings reuse `ConceptPage.astro` and all six original cases,
with their examples in `src/data/advanced/`. Other advanced guides use focused
source-backed examples and the shared dataset explorer when describing records.
All code uses the standard contextual notes. Verify examples against their
named implementation revisions, keep resource and application-data preconditions
explicit, and separate static API verification from runtime or learning evidence.
Site builds never run the examples, launch servers, or import the Python runtime.

### Example datasets

`/datasets/` explores the same six cases as entity relationship diagrams. These
are hypothetical source-table layouts, with illustrative PK/FK columns added to
make joins explicit. The matching schemas consume nested observations and do not
require a relational database or the extra join keys as model inputs. Targets
remain at their actual schema depth. Column types describe source values.

`src/data/datasets.ts` owns the source-table entities, column descriptions, and
relationships. The reusable `DataExplorer.astro` renders accessible HTML cards;
`src/datasets/erd.ts` measures their positions to attach SVG relationships on
desktop and in the stacked mobile layout. The same relationships are written
below each diagram for readers and assistive technology. No diagram library,
Python runtime, or Typst is required.

Column descriptions open on hover, focus, or tap and close with Escape or an
outside tap. Static descriptions and native title hints remain without JavaScript.
The shared use-case selector connects diagrams to the homepage and Core Concepts;
each case also includes the matching input in YAML and the original class schema.

### Describe data anywhere

Use the same explorer for dataset descriptions across the docs:

```mdx
import DataExplorer from '../../components/DataExplorer.astro';
import { repeatPurchase } from '../../data/observations';

<DataExplorer id="purchase-data" dataset={repeatPurchase} headingLevel={3} />
```

Give every instance a page-unique `id`. `headingLevel` sets entity headings to
fit the surrounding document. Author the `Dataset` contract from
`src/datasets/types.ts`: entities contain columns with names, source-value types,
descriptions, and roles; relationships name their entities and optional column
anchors. Shared children and disconnected entities render once. Mark repeated
records with a collection column and relationship; primitive lists remain one
column. A single flat record needs one entity and an empty relationships array.
The renderer validates entity IDs and relationship anchors before building.

All 39 data type examples supply their own dataset contract. Their source
diagrams lead the example, followed by the schema, inspectable exact YAML
records, and visual prediction trees. Duration examples describe actual
timestamp sources and distinguish training-only endpoints from a derived
prediction target. Column focus and hover share field highlighting with code.

Core Concepts reuses the nested observation contracts in
`src/data/observations.ts`, with original field meanings and no source-only join
keys. The introductory data page uses the same component.
Descriptions are authored deliberately, independent of example values, so an
empty list, null value, or short fixture cannot silently redefine the dataset.
Keep measured Evidence in its existing result views.

### Visualize hierarchies, values, and predictions

`Tree.astro` renders conceptual hierarchies such as model collections and nesting.
Its nodes use `label`, optional `detail`, and optional `children`; `TreeNode.astro`
renders them recursively with native disclosures. `BranchShapes.astro` uses this
component for all three schema shapes, keeping their Python code examples beside
the diagrams. The connector lines, spacing, and disclosure styles live in
`src/tree/tree.css` and are also used by the value and prediction trees.

```mdx
<Tree label="Schema structure" nodes={[
  { label: 'readings', detail: 'Branch · repeated records', children: [
    { label: 'value', detail: 'Number' },
  ] },
]} />
```

`ValueTree.astro` is the reusable view for nested values. Pass a page-unique `id`
and `value`. Enable `predictions` for a complete prediction row; rich views apply
only inside its `predictions` column. Pass matching `inputs` to identify repeated
record slots with their known context. Ordinary nested data uses the same
component without that flag, preserving objects that happen to have fields
called state, content, or inferred.

```mdx
<ValueTree id="example-result" value={example.output} inputs={example.inputs} predictions />
```

The renderer shows scalar answers, Boolean probabilities of True, returned
classification candidates, independent membership probabilities, and vector
coordinates. It preserves original ordering and label types without inventing
numeric ranges, uncertainty, or confidence. Empty lists and nulls stay distinct.
`Returned fields` exposes the full state/content/inferred record and extension
fields as another value tree. The toolbar expands or collapses the entire
hierarchy, including all returned fields and their nested values, and
copies the unchanged complete row as JSON. Native disclosures work without
JavaScript and all controls support the keyboard.

`src/values/tree.ts` owns the display projection; `ValueNode.astro` renders it
recursively and `src/values/tree.css` owns prediction-specific styles. All 39 type examples
and the six Core Concepts prediction examples use this view. Results remain
static illustrations with their existing values and notes; no model executes
during rendering. Source inputs and training answers remain inspectable YAML.

### Context inside code

Contextual notes apply automatically to every code example:

````md
```python
orders = Order(length=32, overflow="tail")
```
````

For a component, use `<Code code={source} lang="python" />`.
`src/reference.mjs` owns the shared type, method, option, workflow, output, and
contract explanations. Edit wording there to update every matching code note.
Its `representations` group supplies the fixed type representations used by the
model inspector; model-specific paths and learning context remain with that
component. Static notes have `{ title, text }`; parameterized notes are functions
with named arguments.

Other Astro components or MDX pages can import these same entries. For example,
from a component in `src/components/`:

```astro
---
import { reference } from '../reference.mjs';
---

<p>{reference.types.Number.text}</p>
```

`src/code/config-notes.mjs` owns trigger matching, lexical context, annotation
ranges, and field highlighting. Change it when adding or changing a trigger.
Generated HTML carries the explanation attributes and native `title` fallback;
authors do not write those attributes into examples. These are authored notes,
not inferred Python type information. Review matching when adding a block and
keep essential workflow instructions in page text.

Page summaries remain in frontmatter, and dataset column meanings remain in
their dataset definitions. Those sources already feed the sidebar and data
explorer tooltips, respectively; context-specific descriptions do not need to
be copied into the shared API reference.

Every individual data type page uses the same eight sections: **Example**,
**Inputs**, **Predictions**, **Options**, **Request Methods**, **Footguns**, **Related types**,
**Evidence**.
Use `Option | Default | Purpose` for options tables, class schemas with a short
docstring for examples, and verified input/output contracts. The overview at
`/data-types/` helps readers choose between representations.
Request methods use a `Method | Returns | Purpose` table and a short Python
example showing the type helper, model, and absolute field address. Include
explicit result type annotations in each snippet, using the example’s label
types and the helper’s actual return contract. Include
return-value and lifecycle details that affect usage. Types with no specific
helpers say so; validators and inherited schema methods are excluded.

Footguns connect a specific mistake to its consequence and an actionable check
or remedy, with an inline Evidence or implementation link. Keep the bullet format
consistent across types. Scope empirical lessons to the measured task and
revision, including controls and unmet checks. An unmeasured experiment is a
question, not a finding; API constraints and unsupported outputs can be explained
from pinned implementation/tests without calling them learning results.

All thirteen type pages use `TypeExamples.astro` to explore three scenarios per type.
Author schemas, known inputs, training answers, and complete illustrative
prediction rows together in `src/data/examples/<type>.ts`. The shared contract
and rich-result helper live in `src/data/type-example.ts`. Preserve the API's
`predictions` structure and rich leaf results. The component
builds a complete training record by merging answers into aligned input records.
The selector updates the full example, supports keyboard navigation and deep
links, and leaves every example readable without JavaScript. Field names link
schema and data notes through `contract` metadata on the shared code renderer.
Keep the illustration label visible; these examples do not execute models.
Inputs, training answers, complete training records, and prediction rows render
as YAML with two-space indentation and soft wrapping.
The model still returns a prediction table; this is a display format. Copying
retains the complete record and its probabilities. YAML keys, including quoted
keys and list-member keys, use the shared contextual notes and field highlighting.
Text, Hash, and Period examples show how those inputs support other targets;
they do not show generated text, identifiers, or timestamps. Group examples use
known identities as context for downstream predictions. Ordinal writes native
ordered labels or integer ranks. Duration can derive a numeric target from
timestamp sources in the training record; use `answerNote` to explain that step.

The datatype inventory, schema declarations, and workflow APIs follow
stage/0.5.0 revision `a1d0be3e373cf6c183e4926f1bfcbef845f2f883`.
Keep this source scope explicit when validating examples or preparing publication;
recorded Evidence and benchmark runs retain their own revisions.

`ec.config.mjs` registers the plugin for both Markdown fences and the `Code`
component, alongside the syntax themes and shared code formatting.

`notes-client.ts` supports hover, focus, and tap; `notes.css` provides the visual
cue and popup styles. Notes render outside the code, so explanations never
enter copied Python or change the schema panel's height. Escape dismisses a
note without moving focus. Without JavaScript, native title hints remain.

`src/ui/tooltip.ts` owns the shared popup lifecycle for code notes, sidebar
summaries, and dataset columns. Keep opening, dismissal, positioning, and accessible
description restoration there; adapters own their content and trigger semantics.
Only one popup opens at a time. The code annotation plugin uses a small lexical
context filter to distinguish constructor and fit presets and avoid annotating
unrelated calls, comments, or strings. It does not resolve arbitrary Python types.

Tooltip events are delegated by root instead of attached to every token or
column. Dataset diagrams begin enhancement near the viewport and pause resize
observation when hidden or offscreen. Resize updates are grouped per frame, with
geometry read before SVG paths are written.

Prefer a concrete task, a short declaration, and an interpretable result.
Introduce an implementation detail when it helps a reader make progress.
Describe the known inputs, prediction targets, and observation boundary before
asking the reader to apply the workflow to their own records.

### API map

`src/data/api-map.ts` is the curated static API catalog. `ApiMap.astro` groups
it by workflow, searches names and purposes, and links entries to the existing
guides and exact source definitions. Keep public exports, optional serving names,
common model methods, and preset APIs consistent with their owning source
snapshot. The page states the separate schema/workflow and field-type revisions.
Site builds do not import the Python package or inspect an installed runtime.

The map supports section filters, shareable search URLs, entry anchors, and
readable full content without JavaScript. Verify filtering, reset, direct entry
links, source targets, narrow layouts, and both themes when changing it.

## Preserve Evidence

The canonical experiment sources and learning protocol remain under the
repository's `evidence/` directory. Their identifiers, generators, controls,
seeds, budgets, checks, and provenance are independent of this website.

`src/evidence/data.ts` reads the single CI publication selected by
`RELFLOW_EVIDENCE_PUBLICATION`. It consumes the existing schema-1 publication
contract, including the full records, check dictionaries, nested measurements,
and execution provenance. That environment variable is also the selection
interface used by the existing documentation publisher. To build against a
verified CI artifact:

```sh
RELFLOW_EVIDENCE_PUBLICATION=/absolute/path/to/publication.json npm run build
```

Without an explicit selection, local builds use the canonical `../../evidence/results.json`,
the unchanged historical bootstrap publication, and label it as an archived
snapshot. A selected pending publication stays empty; a missing or invalid
explicit path fails instead of falling back to older results. CI previews with
an unmet aggregate gate retain their candidate records and individual checks.
The site consumes an already selected publication; it does not replace
the trusted artifact verifier, change CI orchestration, or run experiments.

The static `results.json` and `P021-record.json` download routes generate their
historical payloads from that canonical file. Tests bind both to their original
bytes. `/evidence/selected.json` continues to emit the exact selected raw bytes,
including an explicitly pending selection; historical downloads never fill a
missing selected measurement.

Every result count, outcome, criterion, readout, and measurement derives from
that input. Experiment descriptions come from the canonical declarations.
Readout declarations contain metric paths and display formats;
their numbers are looked up in the selected record. The renderer enumerates
canonical experiment declarations without importing Python. It selects exact
executed source using the recorded revision and script hash. The raw-source-only
`src/data/evidence-archive.json` keeps the bootstrap's original scripts available
offline, including their original comments; embedded old result headers are
never used as measurements.

Each experiment has a page at `/evidence/Pxxx/`. Measurements, Method, Reproduce,
and Code panels retain complete results and source context. Authored interpretation
is folded under Method, labeled as source commentary rather than an automatic CI
verdict. Unmet checks and missing measurements remain visible. The selected
publication downloads byte-identically from `/evidence/selected.json`; individual
record and exact-source downloads are generated from the same selection.

Selected historical results may refer to experiments outside the current
execution catalog. `src/data/evidence-source-paths.json` binds each retained
script digest to its stable ID, original path, and lane; it stores no outcomes.
Only an exact source/ID/digest match can retain such a result. These pages are
labeled as archived experiments and direct reproduction to the recorded
checkout instead of offering the current runner's ID command. Current
experiments without selected measurements stay pending, including replacement
proofs with different IDs.

`npm test` changes publication inputs to verify that checks, metrics, and outcomes
follow CI, pending results cannot borrow historical measurements, and invalid
selections fail. The complete raw record remains available alongside readable
nested metric tables. No per-experiment result files are authored or maintained.

`public/benchmarks/` holds the frozen comparative records and parameter counts.
Keep named task cohorts, counting boundaries, dates, and source revisions
attached to comparative claims. A presentation change creates no new
measurement and must not relabel a historical run as current execution.

Before handing off changes, run the type check and production build, then check
links, search, copying, narrow screens, and both themes. The root publisher owns
hosted version paths and trusted Evidence selection. There are no compatibility
routes.
