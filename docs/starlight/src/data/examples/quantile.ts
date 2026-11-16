import { prediction, type TypeExample } from '../type-example.ts';

export const quantileExamples: TypeExample[] = [
  {
    id: 'processing',
    label: 'Predict a measurement',
    summary: 'Learn duration by relative position',
    title: 'How long will this processing job take?',
    description: 'Represent payload size and duration by their positions in their respective training distributions.',
    schema: `import relflow as rf


class ProcessingJob(rf.Model):
    """A queued job with its measured processing duration."""

    payload_mb = rf.Quantile
    worker_count = rf.Number
    duration_ms = rf.Quantile(mask=True)`,
    dataset: {
      id: 'processing', label: 'Processing job',
      observation: 'One queued processing job before it runs.',
      description: 'Payload and worker context paired with the job’s measured duration in source units.',
      entities: [{
        id: 'job', name: 'Processing job', description: 'The job whose duration is being estimated.',
        columns: [
          { name: 'payload_mb', type: 'number', description: 'Payload size in megabytes. The caller supplies the measurement, not its percentile.' },
          { name: 'worker_count', type: 'number', description: 'Number of workers assigned to the processing job.' },
          { name: 'duration_ms', type: 'number', role: 'target', description: 'Measured processing time in milliseconds. Quantile learns its distribution and returns predictions in these same units.' },
        ],
      }],
      relations: [],
    },
    inputs: { payload_mb: 480, worker_count: 4 },
    answers: { duration_ms: 1250 },
    output: { predictions: { duration_ms: prediction(1190) } },
    outputNote: 'The duration content is 1,190 milliseconds, not a percentile. The target distribution converts the learned percentile coordinate back to source units.',
    takeaway: 'Each Quantile field learns its own distribution. A relatively large payload and a relatively long duration refer to different distributions and units.',
  },
  {
    id: 'claim-review',
    label: 'Use rank as context',
    summary: 'Review unusually large claims',
    title: 'Does this claim need a closer review?',
    description: 'Use a claim amount’s relative position as one input to a Boolean review target.',
    schema: `import relflow as rf


class Claim(rf.Model):
    """An insurance claim considered for manual review."""

    amount_usd = rf.Quantile
    incident_age_days = rf.Number
    coverage = rf.Category
    needs_review = rf.Boolean(mask=True)`,
    dataset: {
      id: 'claim-review', label: 'Claim review',
      observation: 'One claim considered for manual review.',
      description: 'Known claim details paired with a historical review decision.',
      entities: [{
        id: 'claim', name: 'Claim', description: 'The claim assessed using its amount, incident age, and coverage.',
        columns: [
          { name: 'amount_usd', type: 'number', description: 'Claim amount in US dollars. Quantile learns its relative position from training values.' },
          { name: 'incident_age_days', type: 'number', description: 'Age of the incident in days at the review cutoff.' },
          { name: 'coverage', type: 'string', description: 'Coverage category known for the claim, such as electronics.' },
          { name: 'needs_review', type: 'boolean', role: 'target', description: 'Historical flag indicating whether the claim needs closer review. Absent from prediction inputs.' },
        ],
      }],
      relations: [],
    },
    inputs: { amount_usd: 4200, incident_age_days: 2, coverage: 'electronics' },
    answers: { needs_review: true },
    output: { predictions: { needs_review: prediction({ probability: 0.83 }) } },
    outputNote: 'The output estimates the probability that needs_review is true. The amount supplies context and does not appear as a predicted percentile.',
    takeaway: 'Quantile can be an input without being a target. Its relative position is learned from this field’s training values; it is not a percentile supplied by the caller.',
  },
  {
    id: 'resale',
    label: 'Every listing',
    summary: 'Estimate nested sale prices',
    title: 'What sale price can we estimate for each listing?',
    description: 'Give every supplied listing its own source-unit price prediction while sharing the marketplace context.',
    schema: `import relflow as rf


class Listing(rf.Branch):
    """A resale listing with its eventual sale price."""

    product_category = rf.Category
    condition = rf.Enum("good", "fair", "poor")
    asking_price_usd = rf.Quantile
    sale_price_usd = rf.Quantile(mask=True)


class ResaleBatch(rf.Model):
    """Listings assessed together in one marketplace."""

    marketplace = rf.Category
    listings = Listing(length=16, overflow="error")`,
    dataset: {
      id: 'resale', label: 'Resale listings',
      observation: 'One batch of supplied listings in a marketplace.',
      description: 'Marketplace context shared by listing records, each with its own eventual sale-price answer.',
      entities: [
        {
          id: 'batch', name: 'Resale batch', description: 'The marketplace context containing the listings assessed together.',
          columns: [
            { name: 'marketplace', type: 'string', description: 'Marketplace label shared by the supplied listings.' },
            { name: 'listings', type: 'records[]', role: 'collection', description: 'Listing records supplied for individual sale-price predictions.' },
          ],
        },
        {
          id: 'listing', name: 'Listing', description: 'One record in listings, retaining its product, condition, and price context.',
          columns: [
            { name: 'product_category', type: 'string', description: 'Category of the listed product, such as camera or bicycle.' },
            { name: 'condition', type: 'string', description: 'Known item condition: good, fair, or poor.' },
            { name: 'asking_price_usd', type: 'number', description: 'Asking price in US dollars when the listing is assessed.' },
            { name: 'sale_price_usd', type: 'number', role: 'target', description: 'Eventual sale price in US dollars. A hidden training answer for this listing, returned in source units when predicted.' },
          ],
        },
      ],
      relations: [{ from: 'batch', to: 'listing', fromColumn: 'listings', cardinality: 'many', description: 'Each batch contains the supplied listings collection.' }],
    },
    inputs: {
      marketplace: 'north',
      listings: [
        { product_category: 'camera', condition: 'good', asking_price_usd: 340 },
        { product_category: 'bicycle', condition: 'fair', asking_price_usd: 120 },
      ],
    },
    answers: { listings: [{ sale_price_usd: 310 }, { sale_price_usd: 95 }] },
    output: {
      predictions: {
        listings: [
          { sale_price_usd: prediction(302.5) },
          { sale_price_usd: prediction(102) },
        ],
      },
    },
    outputNote: 'Both results are dollar amounts, aligned with the two supplied listings. The branch adds record positions; it does not turn the target into a vector of quantile levels.',
    takeaway: 'The sale_price_usd field shares one learned distribution across its listing positions. Predictions remain within that field’s learned price range.',
  },
];
