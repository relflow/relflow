import { scenarios } from '../concepts/scenarios';
import type { CaseId, Concept } from '../concepts/types';

const context: Record<CaseId, { title: string; description: string; branch: string }> = {
  tickets: { title: 'Represent a ticket and its conversation', description: 'Train on escalation and resolution time, then export the learned root representation for the same ticket observations.', branch: 'the supplied conversation' },
  churn: { title: 'Represent a subscriber using both histories', description: 'The churn objective trains context shared by account details, monthly usage, and support history. Export a representation from that fitted model.', branch: 'the monthly usage collection' },
  demand: { title: 'Represent a store-product forecast origin', description: 'Learn the per-week sales targets, then export root context for the store, product, history, and supplied horizon at that forecast origin.', branch: 'the observed sales history' },
  credit: { title: 'Represent an application at assessment', description: 'The default objective trains from the application and its supplied account and inquiry snapshots. Export that model’s application-level context.', branch: 'the inquiry collection' },
  maintenance: { title: 'Represent an asset assessment', description: 'Train from failure outcomes using the sensor and maintenance histories known at each cutoff, then export the root context.', branch: 'the sensor-reading collection' },
  claims: { title: 'Represent a policy review and its claim context', description: 'The per-claim payout and fraud objectives train from nested invoices and line items. A root embedding represents the supplied policy review.', branch: 'the supplied claims collection' },
};

export const embeddings: Concept = {
  id: 'embeddings', title: 'Embeddings',
  description: 'Export learned representations while preserving node identity, repeated-parent coordinates, and model provenance.',
  intro: [
    'Relflow models all users to set embed=True to export learned representations from a fitted model. Export context for a complete observation or a particular branch, then evaluate those vectors for your downstream purposes.',
    'Representations are returned separately from target predictions. Use them for a downstream experiment such as retrieval or another model, and evaluate that use explicitly: a useful supervised target does not automatically establish semantically meaningful distances between embeddings.',
  ],
  examples: scenarios.map((scenario) => ({
    caseId: scenario.id,
    title: context[scenario.id].title,
    description: context[scenario.id].description,
    blocks: [
      {
        label: 'Train the model and request its root representation', lang: 'python' as const,
        code: `import polars as pl
import relflow as rf

# Use the complete ${scenario.modelClass} class schema below.
# training_data supplies labeled splits; inputs is a Polars frame or Arrow table
# containing the matching known records without target answers.
model = ${scenario.modelClass}(preset="small", embed=True)
model.fit(training_data, batch_size=128, max_epochs=20)
outputs: pl.DataFrame = model.predict(inputs, batch_size=256)
representations: pl.Series = outputs["representations"]`,
        note: 'The constructor creates an untrained model. The existing supervised targets provide its objectives; embed=True adds an output request without adding another loss. Every exported vector must match the common width declared by model.schema.d_model.',
      },
      {
        label: 'Inspect complete representation records with their source row', lang: 'python' as const,
        code: `# outputs is the Polars prediction frame from the fitted model above.
for row_index, records in enumerate(outputs["representations"].to_list()):
    for record in records:
        print(
            row_index,
            record["path"],
            record["parents"],
            record["output_index"],
            record["embedding"],
        )`,
        note: 'Keep the outer observation row and the full path/parents/output_index key when storing vectors. Do not treat record order alone as a stable join key. Vector coordinates are returned values, not probabilities or human-named features.',
      },
      {
        label: 'Temporarily export an additional branch representation', lang: 'python' as const,
        code: `# model is fitted and no model operation is active.
branch = rf.where("address") == rf.Address("${scenario.primaryBranch.slice(1)}")
with model.override(branch, embed=True):
    branch_outputs: pl.DataFrame = model.predict(inputs, batch_size=256)`,
        note: `This requests context for ${context[scenario.id].branch}. Branch outputs follow the node’s reduction and repeated ancestors; they are not automatically one vector per source item in that collection. The temporary schema setting is restored on exit.`,
      },
    ],
    takeaway: 'Record the model artifact, output address, repeated-parent coordinates, and reduction configuration alongside downstream vectors. Rebuild or re-evaluate the downstream representation set when those assumptions change.',
  })),
  principles: [
    { title: 'Output selection is separate from an objective', text: 'predict=True declares a supervised reconstruction target. embed=True requests a representation. A target-free model can export representations, but creating that output alone provides no learning objective. Individual field embeddings require the extension’s matching decoder capability.' },
    { title: 'Preserve the representation key', text: 'Each output record contains path, parents, output_index, and embedding. The outer DataFrame row identifies the observation. path names the requested schema node, parents gives repeated-ancestor ordinals, and output_index identifies the node’s intrinsic reduction output.' },
    { title: 'Reduction determines how many vectors appear', text: 'A root representation has path=[] and parents=[]. A reduction with several outputs can emit several root records with different output_index values. For a nested branch, parent ordinals identify its repeated enclosing records; the branch’s own retained collection is handled by its reduction.' },
    { title: 'Keep the representation space tied to its model', text: 'Further training, schema mutations, vocabulary changes, and normalization changes can alter representations. Independent model instances need not share a coordinate space. Version the model and data preparation used to create an index or train a downstream model.' },
  ],
  takeaway: 'Export context from a trained model, preserve its structural coordinates, and measure the downstream use of those representations independently.',
  links: [
    { label: 'Read target predictions', href: '/concepts/runtime/' },
    { label: 'Mutate output selection', href: '/advanced/mutations/' },
    { label: 'Save model state', href: '/advanced/checkpoints-and-state/' },
    { label: 'Own preprocessing and loading', href: '/advanced/custom-data-modules/' },
  ],
};
