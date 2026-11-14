import { prediction, type TypeExample } from '../type-example.ts';

/** Static contract illustrations, never executed predictions or measurements. */
export const booleanExamples: TypeExample[] = [
  {
    id: 'purchase',
    label: 'One probability',
    summary: 'Anticipate a repeat purchase',
    title: 'Will this customer buy again in the next 30 days?',
    description: 'Use known Boolean and numeric facts to learn the probability of one future outcome.',
    schema: `import relflow as rf


class Customer(rf.Model):
    """A customer whose next purchase we want to anticipate."""

    account_age_days = rf.Number
    has_subscription = rf.Boolean
    buys_again_in_30d = rf.Boolean(mask=True)`,
    dataset: {
      id: 'purchase', label: 'Repeat purchase',
      observation: 'One customer at a prediction cutoff.',
      description: 'Known account context paired with a future purchase outcome from a complete 30-day label window.',
      entities: [{
        id: 'customer', name: 'Customer', description: 'The customer whose next purchase is being anticipated.',
        columns: [
          { name: 'account_age_days', type: 'number', description: 'Account age in days at the prediction cutoff.' },
          { name: 'has_subscription', type: 'boolean', description: 'Whether the customer has a subscription at the cutoff. This is known input context.' },
          { name: 'buys_again_in_30d', type: 'boolean', role: 'target', description: 'Whether another purchase occurred during the 30 days after the cutoff. A hidden training answer, absent from prediction inputs.' },
        ],
      }],
      relations: [],
    },
    inputs: { account_age_days: 240, has_subscription: true },
    answers: { buys_again_in_30d: true },
    output: { predictions: { buys_again_in_30d: prediction({ probability: 0.82 }) } },
    outputNote: 'The probability is for True. It is a score for the application to use, rather than an automatically thresholded decision.',
    takeaway: 'The subscription flag is known context. The future purchase flag is a hidden training answer and is omitted when requesting a prediction.',
  },
  {
    id: 'items',
    label: 'Every item',
    summary: 'Predict nested outcomes',
    title: 'Which items are likely to be returned?',
    description: 'Place the Boolean target inside a branch to learn one return probability for each supplied item.',
    schema: `import relflow as rf


class Item(rf.Branch):
    """An ordered item whose return outcome we want to predict."""

    product_category = rf.Category
    price = rf.Number
    is_gift = rf.Boolean
    returned = rf.Boolean(mask=True)


class Order(rf.Model):
    """Items purchased together by one customer."""

    customer_age_days = rf.Number
    items = Item(length=16, overflow="error")`,
    dataset: {
      id: 'items', label: 'Order returns',
      observation: 'One order containing its supplied items.',
      description: 'Order context and per-item purchase details, with a return outcome attached to each item.',
      entities: [
        {
          id: 'order', name: 'Order', description: 'Items purchased together by one customer.',
          columns: [
            { name: 'customer_age_days', type: 'number', description: 'Customer age in days as represented by this example’s source field.' },
            { name: 'items', type: 'records[]', role: 'collection', description: 'Purchased item records supplied for individual return predictions.' },
          ],
        },
        {
          id: 'item', name: 'Item', description: 'One record in items; the return answer stays with its item.',
          columns: [
            { name: 'product_category', type: 'string', description: 'Category of this purchased item, such as clothing or books.' },
            { name: 'price', type: 'number', description: 'Item price known at purchase, in the dataset’s currency units.' },
            { name: 'is_gift', type: 'boolean', description: 'Whether the item was marked as a gift at purchase.' },
            { name: 'returned', type: 'boolean', role: 'target', description: 'Observed return outcome for this item. Omitted when requesting its return probability.' },
          ],
        },
      ],
      relations: [{ from: 'order', to: 'item', fromColumn: 'items', cardinality: 'many', description: 'Each order contains the supplied items collection.' }],
    },
    inputs: {
      customer_age_days: 380,
      items: [
        { product_category: 'clothing', price: 64, is_gift: true },
        { product_category: 'books', price: 18, is_gift: false },
      ],
    },
    answers: { items: [{ returned: true }, { returned: false }] },
    output: {
      predictions: {
        items: [
          { returned: prediction({ probability: 0.67 }) },
          { returned: prediction({ probability: 0.12 }) },
        ],
      },
    },
    outputNote: 'The two result records align with the two supplied items. Each content.probability refers to that item’s returned target.',
    takeaway: 'Nesting places each decision at the level where it belongs. The model predicts outcomes for the supplied records without generating new items.',
  },
  {
    id: 'service',
    label: 'Several answers',
    summary: 'Plan a service visit',
    title: 'Does this job need a visit, and will parts be required?',
    description: 'Learn two Boolean outcomes from shared context, with decision metrics evaluated at explicitly selected thresholds.',
    schema: `import relflow as rf


class ServiceJob(rf.Model):
    """A service request with visit and parts requirements."""

    asset_age_years = rf.Number
    has_remote_access = rf.Boolean
    fault_code = rf.Category
    needs_visit = rf.Boolean(mask=True, threshold=[0.2, 0.5, 0.8])
    needs_parts = rf.Boolean(mask=True)`,
    dataset: {
      id: 'service', label: 'Service requirements',
      observation: 'One service request before planning the response.',
      description: 'Known asset and fault context paired with two separate service requirements.',
      entities: [{
        id: 'job', name: 'Service job', description: 'The shared job context used to predict a visit and parts requirements.',
        columns: [
          { name: 'asset_age_years', type: 'number', description: 'Asset age in years when the service request is assessed.' },
          { name: 'has_remote_access', type: 'boolean', description: 'Whether remote access is available when planning the response.' },
          { name: 'fault_code', type: 'string', description: 'Fault category reported for this service request.' },
          { name: 'needs_visit', type: 'boolean', role: 'target', description: 'Historical flag indicating whether an on-site visit was required. Its configured thresholds affect metrics, not the source field count.' },
          { name: 'needs_parts', type: 'boolean', role: 'target', description: 'Historical flag indicating whether parts were required. Learned as a separate answer from the visit outcome.' },
        ],
      }],
      relations: [],
    },
    inputs: { asset_age_years: 7.5, has_remote_access: false, fault_code: 'pump_warning' },
    answers: { needs_visit: true, needs_parts: false },
    output: {
      predictions: {
        needs_visit: prediction({ probability: 0.89 }),
        needs_parts: prediction({ probability: 0.34 }),
      },
    },
    outputNote: 'Each target returns one probability of True. The needs_visit thresholds configure decision metrics and do not create three prediction fields.',
    takeaway: 'Several Boolean answers can share context while remaining separate objectives. Their probabilities are not competing classes and need not sum to one.',
  },
];
