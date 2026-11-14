import { prediction, type ExampleValue } from '../type-example.ts';
import type { Concept } from './types';
import { scenarios } from './scenarios.ts';

const examples: Record<(typeof scenarios)[number]['id'], {
  title: string; description: string; output: Record<string, ExampleValue>; takeaway: string;
}> = {
  tickets: {
    title: 'Read a probability and a duration from one ticket',
    description: 'The fitted TicketModel has two supervised heads. One estimates the probability of escalation; the other predicts hours to resolution in its original numeric units.',
    output: {
      predictions: {
        escalated: prediction({ probability: 0.28 }),
        hours_to_resolve: prediction(7.5),
      },
    },
    takeaway: 'Different target types share the same input context and prediction row. A Boolean content.probability is the probability of True, not a preselected business action.',
  },
  churn: {
    title: 'Score cancellation for one supplied subscriber',
    description: 'Usage months and support tickets are context. The requested output is the root churn_90d target, so their input-only fields do not reappear as decoded predictions.',
    output: { predictions: { churn_90d: prediction({ probability: 0.64 }) } },
    takeaway: 'The output follows the configured target, rather than copying every known input field. Choose an operating threshold using held-out outcomes and the intended intervention.',
  },
  demand: {
    title: 'Return one forecast for each supplied future week',
    description: 'The shared input supplies two horizon records with known weeks and promotion flags. Each receives its own numeric units_sold result, in the same collection order.',
    output: {
      predictions: {
        horizon: [
          { units_sold: prediction(23.5) },
          { units_sold: prediction(18.2) },
        ],
      },
    },
    takeaway: 'The model fills the target in each supplied forecast slot. It does not generate the future-week list, and Number outputs remain continuous even when the source describes units sold.',
  },
  credit: {
    title: 'Read the requested default probability',
    description: 'The application, account list, and recent inquiries contribute context to default_12m. The Boolean decoder returns a probability inside its rich result.',
    output: { predictions: { default_12m: prediction({ probability: 0.08 }) } },
    takeaway: 'content.probability is the model’s estimate for the named event. The state probabilities describe the field’s value state and have a different meaning.',
  },
  maintenance: {
    title: 'Score the asset, rather than every sensor row',
    description: 'The shared schema places failure_30d at the asset root. Sensor readings and work orders inform that one target; the output does not invent a failure label for each reading.',
    output: { predictions: { failure_30d: prediction({ probability: 0.12 }) } },
    takeaway: 'Target placement determines output placement. A different per-reading task would require a target declared inside sensor_readings and matching training answers.',
  },
  claims: {
    title: 'Keep payout and fraud results inside each claim',
    description: 'The shared input contains one submitted claim, whose nested invoices supply context. Its numeric payout and Boolean fraud predictions stay together under the claims list.',
    output: {
      predictions: {
        claims: [{
          paid_amount: prediction(890),
          is_fraudulent: prediction({ probability: 0.06 }),
        }],
      },
    },
    takeaway: 'Nested mixed-type targets remain aligned with their supplied claim. The model returns fields for existing records; this schema does not generate additional claims or invoices.',
  },
};

export const predictions: Concept = {
  id: 'predictions',
  title: 'Predictions',
  description: 'Read typed predictions, field states, and aligned nested results.',
  intro: [
    'A fitted model returns an eager polars.DataFrame with one row per processed observation. Its predictions column follows the names and nesting of configured targets; repeated branches remain lists of records.',
    'Each example assumes a model already trained for the selected schema and a Polars or Arrow input containing its known records. The tree visualizes one illustrative prediction row. Expand Returned fields to inspect its complete state, content, and inferred values; these examples are not executed results or evidence of application quality.',
  ],
  principles: [
    { title: 'Read the type’s content contract', text: 'A rich leaf contains state probabilities, extension-owned content, and an inferred flag. Number content is a scalar in original units; Boolean content.probability estimates True. These are different readouts even when both use numbers.' },
    { title: 'Keep state separate from confidence in the answer', text: 'state describes valued, null, padded, masked, and other states. For a Boolean, state.valued does not mean the event is likely; content.probability answers that question. inferred records whether the result was inferred.' },
    { title: 'Preserve the supplied collection structure', text: 'Targets inside branches produce aligned list records, including nested structures and structural nulls. Supply the requested records before inference; a nested target is not an instruction to generate a new collection.' },
  ],
  examples: scenarios.map((scenario): Concept['examples'][number] => {
    const example = examples[scenario.id];
    return {
      caseId: scenario.id,
      title: example.title,
      description: example.description,
      blocks: [
        {
          label: 'Predict from the known inputs', lang: 'python',
          code: `import polars as pl

# model is a fitted ${scenario.modelClass}.
# inputs is a Polars frame or Arrow table matching that schema.
results: pl.DataFrame = model.predict(inputs, batch_size=256)
row = results.row(0, named=True)`,
          note: 'predict returns an eager Polars frame. Extracting one named row is optional, for inspecting the application boundary; nested prediction fields retain their lists and records.',
        },
        {
          kind: 'prediction', label: 'One illustrative prediction row',
          value: example.output, inputs: scenario.inputs,
          note: 'Illustrative values only. This row corresponds to the known-input record in the shared scenario; the state values are shown to make the rich output contract visible.',
        },
      ],
      takeaway: example.takeaway,
    };
  }),
  takeaway: 'Read each target through its declared type and preserve the supplied collection alignment. Field-state probabilities and answer probabilities describe different things.',
  links: [
    { label: 'Export learned embeddings', href: '/advanced/embeddings/' },
    { label: 'Choose prediction targets', href: '/concepts/targets/' },
    { label: 'Train the model', href: '/advanced/training/' },
    { label: 'Number output contract', href: '/data-types/number/' },
    { label: 'Boolean output contract', href: '/data-types/boolean/' },
    { label: 'Prediction API', href: '/api-map/#api-model-predict' },
  ],
};
