import type { Value, ValueNotes } from '../values/types';

interface Output {
  value: Record<string, Value>;
  notes: ValueNotes;
}

const state = { valued: 0.96, null: 0.01, padded: 0.01, masked: 0.01, other: 0.01 };
const decodedNotes = {
  '/predictions': 'The field is named value here; outputs use your schema’s actual field names.',
  '/predictions/value/state': 'Probabilities of the field states, not answer confidence.',
  '/predictions/value/inferred': 'Whether the returned value was inferred.',
};

/** A public decoded leaf, independent of any dataset or application scenario. */
function decoded(content: Value, note: string): Output {
  return {
    value: { predictions: { value: { state, content, inferred: true } } },
    notes: { ...decodedNotes, '/predictions/value': note },
  };
}

/** Representation-only leaves require embed=True and have no decoded content writer. */
function representation(note: string): Output {
  return {
    value: {
      representations: [
        { path: ['value'], parents: [], output_index: 0, embedding: [0.12, -0.34, 0.08, 0.2] },
      ],
    },
    notes: {
      '/representations': `With embed=True. ${note}`,
      '/representations/0/path': 'Schema path segments; this field is /value.',
      '/representations/0/parents': 'Repeated ancestor positions; empty for a root-level field.',
      '/representations/0/output_index': 'Index within this field’s intrinsic outputs.',
      '/representations/0/embedding': 'One learned vector; four coordinates for d_model=4.',
    },
  };
}

/** Illustrative output shapes, not measurements or results from executed models. */
export const outputs = {
  number: decoded(42.5, 'Numeric content in the training labels’ original units.'),
  quantile: decoded(
    42.5,
    'A source-unit scalar within the learned distribution’s support; not a percentile or prediction interval.',
  ),
  vector: decoded(
    [0.12, -0.34, 0.08],
    'A fixed-width numeric vector matching n_dim=3; coordinates are not probabilities.',
  ),
  coordinates: decoded(
    [40.7128, -74.006],
    'A latitude and longitude pair in degrees, in that order.',
  ),
  ordinal: decoded(
    'medium',
    'A declared ordered label, preserving its scalar type. With no declared labels, content is an integer rank. No label probability is returned.',
  ),
  category: decoded(
    {
      value: 'label_a',
      probability: 0.72,
      topk: [
        { value: 'label_a', probability: 0.72 },
        { value: 'label_b', probability: 0.18 },
      ],
    },
    'The selected label and its probability. With topk=[2], topk contains up to two candidates; it is an empty list when top-k output is not requested.',
  ),
  boolean: decoded(
    { probability: 0.82 },
    'Probability of True; apply your decision threshold separately.',
  ),
  enum: decoded(
    {
      value: 'low',
      probability: 0.72,
      topk: [
        { value: 'low', probability: 0.72 },
        { value: 'high', probability: 0.28 },
      ],
    },
    'A label from Enum("low", "high") and its probability, shown with topk=[2]. Written labels are strings; topk is empty when not requested.',
  ),
  hash: representation(
    'Hash does not decode the original identifier; reconstruction operates on fingerprint buckets.',
  ),
  set: decoded(
    [
      { value: 'label_a', probability: 0.86 },
      { value: 'label_b', probability: 0.64 },
    ],
    'Independent membership probabilities. threshold filters returned labels; an empty result is [].',
  ),
  group: {
    value: {
      predictions: { value: { state, group: { value: 1, probability: 0.38 }, inferred: true } },
    },
    notes: {
      ...decodedNotes,
      '/predictions/value/group':
        'Decoder group readout, without an identity label. The built-in reconstruction loss does not train this group head; this is not a learned input assignment or calibrated confidence. Inspect input assignments with rf.Group.assignments.',
    },
  },
  period: representation(
    'Period reconstructs calendar-part representations internally; it does not write a predicted datetime.',
  ),
  duration: decoded(
    3.5,
    'Elapsed time in the configured granularity, for example days; not a timestamp or formatted duration.',
  ),
  image: representation(
    'Image uses the frozen vision vector as an internal reconstruction target.',
  ),
  text: representation(
    'Text reconstructs frozen encoder vectors internally; it does not generate strings or completions.',
  ),
} satisfies Record<string, Output>;
