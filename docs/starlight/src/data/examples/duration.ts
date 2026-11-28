import { prediction, type TypeExample } from '../type-example.ts';

export const durationExamples: TypeExample[] = [
  {
    id: 'account-age',
    label: 'Time between events',
    summary: 'Use account age as context',
    title: 'Does this account need a closer review?',
    description: 'Calculate account age from two known timestamps, then use that elapsed time alongside recent activity.',
    schema: `import relflow as rf


class AccountReview(rf.Model):
    """An account assessed at an explicit review time."""

    account_age_days = rf.Duration(
        query=("created_at", "reviewed_at"),
        granularity="days",
    )
    recent_declines = rf.Number
    needs_review = rf.Boolean(predict=True)`,
    dataset: {
      id: 'account-age', label: 'Account review',
      observation: 'One account at an explicit review time.',
      description: 'Two known timestamps supply the elapsed account age used with recent activity to predict a review outcome.',
      entities: [{
        id: 'account', name: 'Account review', description: 'Raw source values for an account assessment; Duration derives account_age_days from the timestamp pair.',
        columns: [
          { name: 'created_at', type: 'datetime', description: 'Known account creation timestamp. The Duration query uses this as the start of account_age_days.' },
          { name: 'reviewed_at', type: 'datetime', description: 'Explicit review cutoff. Duration subtracts created_at from this timestamp and expresses the elapsed value in days.' },
          { name: 'recent_declines', type: 'number', description: 'Recent decline count known at the review cutoff.' },
          { name: 'needs_review', type: 'boolean', role: 'target', description: 'Historical review outcome. This answer is hidden while learning from account age and activity, and omitted from prediction requests.' },
        ],
      }],
      relations: [],
    },
    inputs: {
      created_at: '2026-11-01T12:00:00Z',
      reviewed_at: '2026-11-03T00:00:00Z',
      recent_declines: 2,
    },
    answers: { needs_review: true },
    output: { predictions: { needs_review: prediction({ probability: 0.82 }) } },
    outputNote: 'The Boolean target returns its probability of true. The derived account age supplies input context and is not included as a prediction.',
    takeaway: 'The ordered query paths produce reviewed_at minus created_at: 1.5 days here. Supply the endpoint timestamps; Duration derives and normalizes the elapsed value.',
  },
  {
    id: 'shipment-recency',
    label: 'A shared cutoff',
    summary: 'Measure every shipment’s age',
    title: 'How much longer might each shipment take?',
    description: 'Measure each shipment’s elapsed time against the same review cutoff supplied on its parent record.',
    schema: `import relflow as rf


class Shipment(rf.Branch):
    """A shipment with elapsed and remaining delivery time."""

    carrier = rf.Category
    elapsed_hours = rf.Duration(
        query=("shipped_at", "../reviewed_at"),
        granularity="hours",
    )
    delivery_hours_remaining = rf.Number(predict=True)


class ShipmentReview(rf.Model):
    """Shipments assessed together at one review cutoff."""

    destination_region = rf.Category
    shipments = Shipment(length=16, overflow="error")`,
    dataset: {
      id: 'shipment-recency', label: 'Shipment review',
      observation: 'One review containing shipments assessed at a shared cutoff.',
      description: 'A parent review timestamp is combined with each shipment’s departure time to derive elapsed hours.',
      entities: [
        {
          id: 'review', name: 'Shipment review', description: 'The common cutoff and destination context for all supplied shipments.',
          columns: [
            { name: 'reviewed_at', type: 'datetime', description: 'Explicit review cutoff shared by the shipments. Their ../reviewed_at query reads this parent source value.' },
            { name: 'destination_region', type: 'string', description: 'Known destination-region label shared by the supplied shipments.' },
            { name: 'shipments', type: 'records[]', role: 'collection', description: 'Shipment records supplied for individual remaining-delivery-time predictions.' },
          ],
        },
        {
          id: 'shipment', name: 'Shipment', description: 'One record in shipments. Duration derives elapsed_hours using its shipped_at and the parent reviewed_at.',
          columns: [
            { name: 'carrier', type: 'string', description: 'Carrier responsible for this shipment.' },
            { name: 'shipped_at', type: 'datetime', description: 'Known departure timestamp for this shipment. Subtracting it from the parent cutoff gives elapsed_hours.' },
            { name: 'delivery_hours_remaining', type: 'number', role: 'target', description: 'Observed hours still required for delivery after the review cutoff. A separate hidden Number answer for this shipment.' },
          ],
        },
      ],
      relations: [{ from: 'review', to: 'shipment', fromColumn: 'shipments', cardinality: 'many', description: 'Each review contains the supplied shipments, which all use its reviewed_at cutoff.' }],
    },
    inputs: {
      reviewed_at: '2026-11-13T12:00:00Z',
      destination_region: 'north',
      shipments: [
        { carrier: 'parcel_a', shipped_at: '2026-11-12T06:00:00Z' },
        { carrier: 'parcel_b', shipped_at: '2026-11-13T00:00:00Z' },
      ],
    },
    answers: {
      shipments: [
        { delivery_hours_remaining: 8 },
        { delivery_hours_remaining: 30 },
      ],
    },
    output: {
      predictions: {
        shipments: [
          { delivery_hours_remaining: prediction(9.4) },
          { delivery_hours_remaining: prediction(28.7) },
        ],
      },
    },
    outputNote: 'Each Number answer is an estimated remaining duration in hours, aligned with its supplied shipment. The two Duration inputs are 30 hours and 12 hours.',
    takeaway: 'The ../ path reads reviewed_at from this observation’s parent source and broadcasts it across the shipment branch. Every recency uses an explicit cutoff, without reading the wall clock.',
  },
  {
    id: 'repair-target',
    label: 'A duration target',
    summary: 'Learn from completion timestamps',
    title: 'How long might this repair take to resolve?',
    description: 'Derive a hidden training label from the recorded opening and resolution timestamps, then predict its duration for new jobs.',
    schema: `import relflow as rf


class RepairJob(rf.Model):
    """A repair job whose resolution duration is the target."""

    asset_type = rf.Category
    urgency = rf.Enum("routine", "urgent")
    affected_units = rf.Number
    resolution_hours = rf.Duration(
        query=("opened_at", "resolved_at"),
        granularity="hours",
        predict=True,
    )`,
    dataset: {
      id: 'repair-target', label: 'Repair duration',
      observation: 'One repair job when its duration is to be estimated.',
      description: 'Known repair context is paired with historical endpoint timestamps that derive a hidden resolution-hours answer.',
      entities: [{
        id: 'job', name: 'Repair job', description: 'Prediction requests contain the repair context. Labeled training records also contain the duration’s two source timestamps.',
        columns: [
          { name: 'asset_type', type: 'string', description: 'Known equipment type, such as conveyor.' },
          { name: 'urgency', type: 'string', description: 'Known urgency label: routine or urgent.' },
          { name: 'affected_units', type: 'number', description: 'Number of affected units known when assessing the repair.' },
          { name: 'opened_at', type: 'datetime', role: 'training', description: 'Training-only start timestamp for resolution_hours. The Duration query subtracts this from resolved_at; new requests omit both endpoints.' },
          { name: 'resolved_at', type: 'datetime', role: 'training', description: 'Training-only completion timestamp. Together with opened_at it derives resolution_hours, the hidden numeric target in hours, rather than a timestamp prediction.' },
        ],
      }],
      relations: [],
    },
    inputs: { asset_type: 'conveyor', urgency: 'urgent', affected_units: 2 },
    answers: { opened_at: '2026-11-13T09:00:00Z', resolved_at: '2026-11-13T15:30:00Z' },
    answerNote: 'These training-only timestamps derive resolution_hours = 6.5. The target declaration keeps that derived number out of model inputs. Neither endpoint is declared as another input field; new requests omit both timestamps.',
    output: { predictions: { resolution_hours: prediction(6.2) } },
    outputNote: 'The Duration target returns numeric content in hours, with state and inferred. It does not return a resolution timestamp or an endpoint pair.',
    takeaway: 'Historical endpoints supply the training answer through the query. New jobs supply only the declared repair context; the fitted model estimates elapsed hours without knowing the completion time.',
  },
];
