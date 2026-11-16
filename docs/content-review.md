# relflow documentation content review

Working checklist from the documentation sweep against the shared
[Google Doc](https://docs.google.com/document/d/1q04H63KEJnykRBGzkKQo86DohyOmMO-4OThlyrFn4ig/edit).
This is an internal review document, outside the published website content.

All remediations are proposals and start unchecked. Mark them as resolved as you
make or decide against an edit. Suggested wording is a starting point, not a
replacement for your editorial judgment.

The homepage, Featureless, and Modular pages follow the Google Doc most closely.
The largest drift is in Advanced Usage, where implementation mechanics and
qualifications often appear before the reader understands the benefit.

Items 1–13 cover website discrepancies and presentation improvements. Items
14–18 identify wording already present in the Google Doc, or an associated
technical clarification, and need an editorial decision rather than an automatic
rewrite. Items 19–23 are small copy edits.

## Website discrepancies and presentation

### 1. Restore the business purpose of mutations

Source: [Mutations introduction](starlight/src/data/advanced/mutations.ts).

**Current:**

> A model schema can evolve after instantiation. Mutation methods select nodes in the resolved tree, apply an intentional change, and synchronize the runtime graph with that new schema.

The Google Doc leads with adapting a shared pretrained model for different
enterprise teams. The website leads with graph mechanics.

- [ ] Lead with adapting trained models to new inputs, targets, and team-specific requirements.
- [X] Restore the shared pretrained model story from the Google Doc.
- [X] Explain graph rebuilding and restrictions on editing active models after establishing the purpose.
- [X] Preserve the distinction between retaining compatible state and guaranteeing unchanged predictions; new targets still require training.

**Possible opening:**

> Adapt a trained model as your data or task changes. Add prediction targets, work with different subsets of inputs, or create team-specific versions from a shared pretrained model while retaining compatible learned state.

### 2. Complete the code- and docstring-backed API reference

Sources: [API map page](starlight/src/content/docs/api-map.mdx),
[authored API index](starlight/src/data/api-map.ts), and
[API map renderer](starlight/src/components/ApiMap.astro).

**Current:** A manually authored index with short descriptions and source links.
Its entries store names, descriptions, source locations, and optional guide
links. It does not present a docstring-derived reference with full signatures,
parameters, and return documentation.

The Google Doc explicitly requires a reference that mirrors the code and uses
its docstrings. This is an implementation gap, not just a wording issue.

- [ ] Keep the task-oriented map as the navigation layer.
- [X] Provide actual signatures, parameter/default descriptions, and return documentation for entries.
- [X] Feed the reference from source definitions and docstrings, retaining exact source links.
- [ ] Preserve the static build boundary: generating the reference must not require importing relflow, Torch, or Lightning.

### 3. Fix the custom-type capability table reference

Source: [Custom leaf types](starlight/src/content/docs/advanced/custom-data-types.mdx).

**Current:**

> An input-only type can use the first two capabilities.

The first two table rows are field declaration and optional derivation. Input
encoding is the third row, so the sentence points readers at the wrong
capabilities.

- [X] Name the required capabilities directly instead of referring to row positions.
- [ ] Keep optional derivation distinct from input representation and supervised output support.

**Possible wording:**

> An input-only field needs a request definition, tensorization, and an embedder. Derivation is optional. Prediction targets additionally need a decoder and loss.

### 4. Make the custom data module opening task-oriented

Source: [Custom data modules](starlight/src/content/docs/advanced/custom-data-modules.mdx).

**Current:**

> Customize how observations reach the model while preserving their structure, split ownership, and encoding state.

This sounds like an internal design document. The reader first needs to know
which source adapter to choose and what it lets them do.

- [X] Lead with connecting a source that the built-in loaders do not already handle.
- [X] Introduce the Python-record and Arrow-source options before discussing ownership and internal batching.
- [X] Retain the detailed source, worker, and distributed contracts further down the guide.

**Possible wording:**

> Connect a source that the built-in loaders do not already handle. Use `CustomDataModule` for Python records, or `ArrowDataModule` when your connector already returns Arrow.

### 5. Remove documentation-build commentary from reader-facing prose

Sources: [Shared concept examples](starlight/src/components/ConceptPage.astro),
[Advanced overview](starlight/src/content/docs/advanced/index.mdx),
[Custom data modules](starlight/src/content/docs/advanced/custom-data-modules.mdx),
and [Custom leaf types](starlight/src/content/docs/advanced/custom-data-types.mdx).

**Current examples:**

> Examples are static illustrations.

> They do not execute during the documentation build.

> Code is a static illustration; site builds do not import the extension or train this model.

These statements describe our publishing process rather than helping a reader
use relflow. Repetition also makes the examples feel less concrete.

- [X] Remove explanations of how the documentation build works from product guides.
- [ ] Keep authoring/build instructions in the documentation maintainer guides.
- [X] Keep “Illustrative prediction output” labels where they distinguish sample outputs from measured Evidence.
- [X] Retain a short tooltip interaction hint where useful, without repeating the publishing disclaimer.

### 6. Lead the embeddings guide with a useful action

Source: [Embeddings introduction](starlight/src/data/advanced/embeddings.ts).

**Current:**

> An embedding is a model representation of the supplied context.

The opening then immediately explains what enabling output on an untrained
model does not accomplish. Purpose and use should come before qualifications.

- [X] Open with requesting representations from a fitted model and what can be exported.
- [ ] Explain the separation between output requests and learning objectives after the introduction.
- [ ] Preserve the requirement to evaluate a downstream use separately; do not imply that any exported embedding is automatically useful for similarity.

**Possible opening:**

> Use `embed=True` to export learned representations from a fitted model. Export context for a complete observation or a particular branch, then evaluate those vectors for your downstream task.

### 7. Move raw source revisions out of guide introductions

Sources include [Online vocabulary](starlight/src/content/docs/advanced/online-vocabulary.mdx),
[Distributed training](starlight/src/content/docs/advanced/distributed-data.mdx),
[Checkpoints](starlight/src/content/docs/advanced/checkpoints-and-state.mdx), and
[Masking and pretraining](starlight/src/content/docs/advanced/masking-and-pretraining.mdx).

**Current example:**

> This guide follows the field-state implementation at `a1d0be3e`.

The raw commit interrupts the practical introduction. Exact implementation
provenance is useful, but it need not lead the lesson.

- [X] Use a readable release label such as “Applies to relflow 0.5.0” near the top when needed.
- [ ] Place exact revisions and implementation/test links in a consistent source-details area.
- [ ] Keep historical version labels and scientific scope conspicuous on Evidence pages.

### 8. Broaden the Featureless page description to match the promise

Source: [Featureless frontmatter](starlight/src/content/docs/featureless.mdx).

**Current:**

> Learn predictive features from typed values and related records without a separate aggregation table.

“Aggregation table” narrows the featureless proposition to one preparation
pattern. The Google Doc describes eliminating a separate feature engineering
pipeline.

- [X] Replace “without a separate aggregation table” with “without a separate feature engineering pipeline.”
- [ ] Keep the page description consistent with the homepage feature bullet, since descriptions also appear in sidebar tooltips.

### 9. Make the Leaf Types reference link fulfill its promise

Source: [Modular modeling](starlight/src/content/docs/modular.mdx).

**Current:**

> The Leaf Types reference expands on each individual type, as well as the contracts given for a leaf type extension.

The link goes to the Core Concepts page about states and registry contracts,
not the individual type reference pages.

- [X] Distinguish the individual type references from the conceptual extension contract in the sentence.
- [ ] Point each link at a destination that actually provides the promised content.
- [ ] Review the similar “built-in type” link in the custom leaf type guide.

### 10. Give type-selection guidance an obvious home

Sources: [Leaf Types navigation](starlight/astro.config.mjs) and the
[individual references](starlight/src/content/docs/data-types/).

**Current:** Continuous and Discrete data are collapsible sidebar groups.
Selection advice is scattered among the individual pages and their Related
types sections.

The Google Doc explicitly asks for comparisons such as Number versus Quantile
and Category versus Enum.

- [ ] Decide where short Continuous and Discrete type comparisons should live.
- [X] Make Number versus Quantile and Category versus Enum easy to find before reading every individual reference.
- [ ] Include the distinct purposes of Group, Hash, Set, and Boolean in the discrete comparison.
- [ ] Link to relevant Evidence without changing the uniform individual reference structure.
- [ ] Respect the earlier removal of a generic “Choose your types” section from Core Concepts when choosing placement.

### 11. Make the remaining implementation-heavy Evidence titles more direct

Source: [Evidence editorial reviews](starlight/src/evidence/reviews.ts),
particularly P013 and P041.

**Current:**

> Can relflow recover a total from a Mean summary that includes count?

> Can a count-augmented Mean summary retain different totals?

These still lead with implementation vocabulary rather than observable
behavior. P001 is the clearer standard.

- [X] Reword P013 around the prediction task, for example: “Can relflow predict the total of repeated values?”
- [X] Give P041 the same behavior-first treatment while keeping its distinction from P013 clear.
- [ ] Keep the Mean implementation and count-signal explanation in the setup and scope notes.
- [ ] Preserve all experiment identities, original sources, criteria, measurements, and CI-derived verdicts.

### 12. Move execution metadata below the Evidence question and setup

Source: [Evidence report header](starlight/src/components/EvidenceReport.astro).

**Current:** Device, seed, and run mode appear before “How is the question
modeled?”

The Google Doc makes the modeled question primary and execution details
tertiary.

- [ ] Keep the experiment ID, leaf tags, verdict, publication/version, and historical status visible.
- [X] Move device and seed into Method or Reproduction.
- [ ] Review run-mode placement while retaining an obvious distinction between full and limited executions.
- [ ] Preserve every underlying execution/provenance field and its downloads.

### 13. Restore the requested community links in the header

Source: [Header social configuration](starlight/astro.config.mjs).

**Current:** GitHub is the only social link. The Google Doc asks for Discord,
GitHub, and LinkedIn.

- [X] Add the established Discord invitation to the header.
- [X] Confirm the approved company LinkedIn URL and add it: https://www.linkedin.com/company/relflow
- [ ] Keep the GitHub link and check the header layout on narrow screens.

## Decisions about wording already in the Google Doc

These are not examples of the website independently replacing the intended
message. They are ambiguities or scope questions to resolve in the source copy
and then carry through consistently.

### 14. Distinguish Padded from Null

Source: [State table](starlight/src/content/docs/concepts/data-types.mdx).

**Current Padded definition:**

> The data observed does not exist.

This does not clearly distinguish padding from a missing field value.

- [X] Clarify Null as a missing/null value for a field.
- [X] Clarify Padded as the absence of a collection item at a position used to align differently sized collections.
- [ ] Apply the agreed distinction to the Google Doc and website together.

**Possible definitions:**

> **Null:** The observation has no value for this field (`None`/null).
>
> **Padded:** No collection item exists at this position; padding aligns differently sized collections.

### 15. Clarify which observations share attention

Source: [Context branches](starlight/src/content/docs/concepts/branches.mdx).

**Current excerpt:**

> every other value from every other observation

This can sound like attention crosses unrelated examples in a training batch.

- [ ] Define “observation” at this point in the explanation.
- [X] State that the described interactions occur within one top-level model observation.
- [ ] Distinguish child records inside a branch from independent top-level observations in a batch.

**Possible wording:**

> Within one model observation, attention connects fields within each child record and related records within the branch.

### 16. Align Mean descriptions with its count signal

Sources: [API summary](starlight/src/data/api-map.ts),
[branch reductions](starlight/src/content/docs/concepts/branches.mdx),
[MeanPool implementation](../src/relflow/architecture/pool.py), and
[P013/P041 explanations](starlight/src/evidence/reviews.ts).

**Current API summary:**

> Reduce a branch by averaging its present encoded coordinates.

P013 and P041 explain that the branch implementation also preserves count
information. The conceptual description and reference should make that
distinction clear.

- [X] Explain the branch Mean reduction’s count signal alongside its averaged content.
- [ ] Align the conceptual page, API summary, relevant source docstrings, and Evidence explanations.
- [ ] Keep branch reduction behavior distinct from other internal uses of mean pooling.

**Possible wording:**

> Mean averages the encoded values while retaining a learned count signal.

### 17. Expand HPO and NAS on first use

Source: [Model presets](starlight/src/content/docs/concepts/models.mdx).

**Current excerpt:**

> the result of objective HPO & NAS

- [X] Expand both abbreviations on first use.
- [ ] Preserve the presets’ benchmark-derived origin and link to the supporting benchmark discussion.
- [ ] Keep architecture defaults distinct from fit defaults, as explained in Runtime.

**Possible wording:**

> The presets were selected through hyperparameter optimization and neural architecture search on relational benchmarks.

### 18. Make the scope of scaling claims easy to inspect

Sources: [Model Scaling](starlight/src/content/docs/scaling.mdx) and
[benchmark counting boundaries](starlight/src/content/docs/benchmarks.mdx).

**Current excerpts:**

> less than 1% of competing models

> 50K observations per second on a humble L4 GPU

These claims originate in the Google Doc. The concern is making their scope
as explicit as the more detailed Benchmarks page, not removing the SOTA
positioning.

- [ ] Identify the parameter-count example as the retained predictive network and name the compared checkpoint.
- [ ] Keep the frozen text encoder’s parameter count readily available alongside the task-trained network count.
- [ ] Link the 50K/L4 throughput statement to a receipt with workload, model configuration, batching, and hardware details. No corresponding published throughput receipt was found in the audited docs; that does not establish that no receipt exists elsewhere.
- [ ] Clarify the scope and reproducibility conditions of “perfectly deterministic behavior at inference.”
- [X] Retain strong, precisely scoped claims and their supporting records.

**Possible parameter-count wording:**

> The retained predictive network has 926,554 task-trained parameters—about 0.42% of the compared TabPFN-3.5 base checkpoint.

## Small copy edits

### 19. Restore “prolong the lifecycle”

Source: [Featureless modeling](starlight/src/content/docs/featureless.mdx).

**Current:** “exacerbate the lifecycle.” The Google Doc uses “prolong.”

- [X] Replace “exacerbate the lifecycle” with “prolong the lifecycle.”

### 20. Remove the unexplained asterisk

Source: [Information Reductions](starlight/src/content/docs/concepts/branches.mdx).

**Current:** `reductions*` has no corresponding explanation.

- [X] Remove the asterisk or supply the intended footnote if one was omitted.

### 21. Use “leaves” consistently

Sources: [Context branches](starlight/src/content/docs/concepts/branches.mdx)
and [Leaf Types](starlight/src/content/docs/concepts/data-types.mdx).

**Current:** Both “leaves” and “leafs” appear in prose.

- [X] Replace editorial uses of “leafs” with “leaves.” Preserve actual API identifiers and exact historical source bytes.

### 22. Match the custom-type link label to navigation

Source: [Registry Contracts discussion](starlight/src/content/docs/concepts/data-types.mdx).

**Current:** “BYOD section” links to a page titled “Custom leaf types.”

- [X] Use “Custom leaf types” as the link label, or explicitly introduce BYOD before using it.

### 23. Standardize editorial casing

Sources: [Navigation](starlight/astro.config.mjs),
[Context branches](starlight/src/content/docs/concepts/branches.mdx), and
[Model trees](starlight/src/content/docs/concepts/models.mdx).

**Current examples:** “Model Scaling,” “Core Concepts,” “Context branches,”
“State vs Content,” and lowercase “discord” are mixed.

- [X] Decide a consistent heading/navigation capitalization convention, taking the source Doc’s intended labels into account.
- [X] Capitalize Discord consistently.
- [X] Keep the project name as `relflow` in ordinary prose; preserve API identifiers, protocol headers, and exact recorded source names.

## Preserve while editing

- Prominent SOTA positioning with precisely scoped, inspectable comparisons.
- The intended Philosophy positioning and featureless-modeling message.
- The six use cases and shared Data / Model / Schema explorer.
- Class-based model and branch declarations.
- The consistent leaf-reference sections, Request Methods, and Footguns.
- Visible Evidence verdicts, historical labels, meaningful limitations, and CI-derived measurements.
- Exact scientific records, criteria, source identities, and reproduction details.
- The intentional hidden table of contents, removed WIP tutorial, and current implemented leaf-type inventory; do not treat older outline notes as instructions to restore them.

Suggested starting order: the Mutations introduction (1), shared build-process
commentary (5), and API-reference gap (2).
