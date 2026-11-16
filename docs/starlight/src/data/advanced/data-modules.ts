import { stringify } from 'yaml';
import { scenarios } from '../concepts/scenarios';
import type { CaseId, Concept } from '../concepts/types';

const observations: Record<CaseId, { title: string; description: string; preparation: string; source: string; takeaway: string }> = {
  tickets: {
    title: 'Keep a ticket conversation inside one observation',
    description: 'Each row is a ticket as known at triage. Its messages remain together inside the row; escalation and resolution labels are added only to training and evaluation records.',
    preparation: `@rf.preprocess
def prepare_ticket(frame):
    return frame.rename({"source_product": "product"})

`,
    source: `data = rf.ArrowDataModule(
    train="tickets/train.parquet",
    validate="tickets/validation.parquet",
    preprocessor=prepare_ticket,
)`,
    takeaway: 'This optional rename handles a source that calls product source_product. It runs before schema queries. The application has already assembled each conversation and cut it at triage; a partition-level rename needs no whole-dataset collection.',
  },
  churn: {
    title: 'Keep both subscriber histories under the same cutoff',
    description: 'One row contains a subscriber’s usage and support histories at one cutoff. Arrange later evaluation cutoffs and complete 90-day labels before handing the splits to the loader.',
    preparation: '',
    source: `data = rf.ArrowDataModule(
    train="churn/train.parquet",
    validate="churn/validation.parquet",
    test="churn/test.parquet",
)`,
    takeaway: 'The loader shuffles training observations, not individual months between subscribers. It does not define the time split or decide whether overlapping subscriber windows create leakage.',
  },
  demand: {
    title: 'Batch forecast origins with their supplied horizon slots',
    description: 'One row is a store and product at one forecast origin. Its past history and complete requested horizon travel together; observed future units are labels on the horizon during training.',
    preparation: '',
    source: `data = rf.ArrowDataModule(
    train=rf.source("demand/train/*.parquet"),
    validate=rf.source("demand/validation/*.parquet"),
)`,
    takeaway: 'File shards can stream without flattening the horizon. A batch size counts store-product observations; the horizon branch separately limits future slots. Keep evaluation labels and available promotions consistent with each forecast origin.',
  },
  credit: {
    title: 'Assemble accounts and inquiries before loading applications',
    description: 'Each row is an application with accounts and inquiries known then. Prepare the joins and the settled default label in the source pipeline, with later evaluation applications kept separate.',
    preparation: '',
    source: `data = rf.ArrowDataModule(
    train="credit/applications/train.parquet",
    validate="credit/applications/validation.parquet",
    test="credit/applications/test.parquet",
)`,
    takeaway: 'Arrow preserves the account and inquiry collections. A field query selects within that prepared structure; it does not join a bureau table or remove post-application updates for you.',
  },
  maintenance: {
    title: 'Scan asset snapshots while preserving their event histories',
    description: 'One row is an asset at an assessment time, with sensor readings and completed work known then. The source dataset contains these nested snapshots, not a flat stream of unrelated sensor points.',
    preparation: '',
    source: `data = rf.ArrowDataModule(
    train="maintenance/train/*.parquet",
    validate="maintenance/validation/*.parquet",
)`,
    takeaway: 'The file shards are scanned again for each pass. The loader can stream snapshots, but the application must still bound and order each history and avoid including work performed after the assessment.',
  },
  claims: {
    title: 'Preserve claim, invoice, and line-item boundaries',
    description: 'One row is a policy review containing submitted claims and their invoice evidence. Each claim’s payout and fraud labels remain attached to that claim in the labeled splits.',
    preparation: '',
    source: `data = rf.ArrowDataModule(
    train="claims/policy-reviews/train.parquet",
    validate="claims/policy-reviews/validation.parquet",
)`,
    takeaway: 'Do not turn every line item into an independent root row unless you intend to change the learning problem. Streaming the nested observation preserves which provider billed which items and which claim owns the answers.',
  },
};

export const dataModules: Concept = {
  id: 'data-modules',
  title: 'Data modules',
  description: 'Deliver complete observations and separate splits through a repeatable Arrow data pipeline.',
  intro: [
    'A data module supplies training, validation, test, or prediction observations. The model schema describes what those observations mean; the data module controls how the prepared rows are read, shuffled, and batched.',
    'ArrowDataModule is the usual starting point for nested Parquet files, Arrow tables, or datasets. The paths below name files prepared by your application. The explorer describes one processed input observation, with exact values available as YAML; supervised training and evaluation files also contain its answer fields.',
  ],
  principles: [
    {
      title: 'One batch row is one complete observation',
      text: 'A customer, conversation, forecast origin, application, asset assessment, or policy review keeps its related records inside the row. The model batch size counts these outer rows. Branch limits independently bound the records inside them.',
    },
    {
      title: 'Split by the decision you will evaluate',
      text: 'Prepare training, validation, and test populations before loading. For time-dependent tasks, preserve each cutoff and wait for complete labels. The default training shuffle does not make an unsafe split valid.',
    },
    {
      title: 'Preprocess before structural queries',
      text: 'Preprocessors receive eager Polars frames and produce the canonical records used by the schema. Use them or your source pipeline for renaming, joins, filtering, ordering, and derived values. A node query selects within those records; it does not infer joins. Dataset-scoped preprocessing can collect the split in memory.',
    },
    {
      title: 'Stream repeatable sources with a stable schema',
      text: 'Files and Arrow Datasets are read in batches without requiring a full split in memory. A callable source must create a fresh Arrow stream for each pass; pass a Dataset rather than a one-shot reader. Keep the processed schema consistent across chunks.',
    },
    {
      title: 'Let the model loop bind the loader',
      text: 'fit, validate, test, and data-module prediction bind an unbound relflow data module to the model that owns its schema and batch size. Use data.bind(model) only for direct loader access. Create a new data module when using another loaded model.',
    },
  ],
  examples: scenarios.map((scenario) => ({
    caseId: scenario.id,
    title: observations[scenario.id].title,
    description: observations[scenario.id].description,
    blocks: [
      {
        label: 'One processed prediction observation',
        lang: 'yaml', records: true,
        code: stringify(scenario.inputs, { indent: 2, lineWidth: 0 }),
        note: 'The enclosing row stays intact. Training and evaluation add the answer fields shown on the Targets page.',
      },
      {
        label: 'Read prepared splits and bind them during fitting',
        lang: 'python',
        code: `import relflow as rf

${observations[scenario.id].preparation}${observations[scenario.id].source}

model = ${scenario.modelClass}(preset="small")
model.fit(data, preset="small")`,
        note: scenario.id === 'tickets'
          ? 'Use the selected TicketModel definition. In this source-renaming variant, input files call the product column source_product; the preprocessor produces the illustrated product field.'
          : `Use the selected ${scenario.modelClass} definition. The files or dataset directories are application-supplied; constructing the model starts it untrained, and fit binds the data module.`,
      },
    ],
    takeaway: observations[scenario.id].takeaway,
  })),
  takeaway: 'Prepare meaningful observation boundaries and splits first. Then stream and batch those records through the same schema contract, keeping source preparation separate from learned feature extraction.',
  links: [
    { label: 'Build a custom data module', href: '/advanced/custom-data-modules/' },
    { label: 'Describe the observation with a model tree', href: '/concepts/models/' },
    { label: 'Follow training and learned state', href: '/advanced/training/' },
    { label: 'Find data modules, sources, and preprocessors', href: '/api-map/' },
    { label: 'Inspect experimental data and controls', href: '/evidence/' },
  ],
};
