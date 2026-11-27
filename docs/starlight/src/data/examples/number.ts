import { prediction, type TypeExample } from '../type-example.ts';

export const numberExamples: TypeExample[] = [
  {
    id: 'delivery',
    label: 'One value',
    summary: 'Estimate delivery time',
    title: 'How many days will delivery take?',
    description: 'Learn one numeric answer from the order information available when it is placed.',
    schema: `import relflow as rf


class Delivery(rf.Model):
    """An order whose delivery time we want to estimate."""

    order_total = rf.Number
    item_count = rf.Number
    delivery_days = rf.Number(predict=True)`,
    dataset: {
      id: 'delivery', label: 'Delivery order',
      observation: 'One order when it is placed.',
      description: 'Order details available at checkout, paired with a historical delivery-time label.',
      entities: [{
        id: 'order', name: 'Order', description: 'The order whose delivery time is being estimated.',
        columns: [
          { name: 'order_total', type: 'number', description: 'Order total known at checkout, in the dataset’s currency units.' },
          { name: 'item_count', type: 'number', description: 'Number of items included in the order.' },
          { name: 'delivery_days', type: 'number', role: 'target', description: 'Observed days until delivery. A historical training answer, omitted from prediction requests.' },
        ],
      }],
      relations: [],
    },
    inputs: { order_total: 64, item_count: 2 },
    answers: { delivery_days: 3 },
    output: { predictions: { delivery_days: prediction(2.8) } },
    outputNote: 'The numeric content is in days, the same unit as the training answer.',
    takeaway: 'The known delivery time teaches the target during fitting. New requests contain the order details without that answer.',
  },
  {
    id: 'inventory',
    label: 'Every item',
    summary: 'Predict inside a collection',
    title: 'How much should we replenish for each item?',
    description: 'Place a numeric target inside a branch to learn a separate answer for each supplied item.',
    schema: `import relflow as rf


class Item(rf.Branch):
    """An inventory item and its replenishment target."""

    product = rf.Category
    on_hand = rf.Number
    sold_last_7d = rf.Number
    replenish_units = rf.Number(predict=True)


class Inventory(rf.Model):
    """Items reviewed together for the next replenishment."""

    lead_time_days = rf.Number
    items = Item(length=16, overflow="error")`,
    dataset: {
      id: 'inventory', label: 'Inventory review',
      observation: 'One replenishment review containing its supplied items.',
      description: 'A shared lead time and per-item stock context, with a replenishment answer for each item.',
      entities: [
        {
          id: 'review', name: 'Inventory review', description: 'Items assessed together for the next replenishment.',
          columns: [
            { name: 'lead_time_days', type: 'number', description: 'Expected replenishment lead time in days, shared by the supplied items.' },
            { name: 'items', type: 'records[]', role: 'collection', description: 'Item records supplied together. Each keeps its own stock values and replenishment answer.' },
          ],
        },
        {
          id: 'item', name: 'Item', description: 'One record in items; predictions retain the supplied record order.',
          columns: [
            { name: 'product', type: 'string', description: 'Product label for this inventory item, such as notebook.' },
            { name: 'on_hand', type: 'number', description: 'Units currently in stock for this item.' },
            { name: 'sold_last_7d', type: 'number', description: 'Units sold during the seven days before the review.' },
            { name: 'replenish_units', type: 'number', role: 'target', description: 'Historical replenishment quantity for this item. Hidden while learning from the known stock context.' },
          ],
        },
      ],
      relations: [{ from: 'review', to: 'item', fromColumn: 'items', cardinality: 'many', description: 'Each review contains the supplied items collection.' }],
    },
    inputs: {
      lead_time_days: 5,
      items: [
        { product: 'notebook', on_hand: 12, sold_last_7d: 24 },
        { product: 'pencil', on_hand: 8, sold_last_7d: 5 },
      ],
    },
    answers: { items: [{ replenish_units: 36 }, { replenish_units: 0 }] },
    output: {
      predictions: {
        items: [
          { replenish_units: prediction(33.5) },
          { replenish_units: prediction(0.8) },
        ],
      },
    },
    outputNote: 'The two result records follow the two supplied items in order. Each target retains a numeric content value.',
    takeaway: 'A branch target gives each retained record its own answer. Number does not enforce integer counts; the application decides how a score becomes a purchase quantity.',
  },
  {
    id: 'service',
    label: 'Several answers',
    summary: 'Learn time and cost together',
    title: 'How long will the job take, and what will parts cost?',
    description: 'Two numeric targets share the same job context while keeping separate meanings and units.',
    schema: `import relflow as rf


class ServiceJob(rf.Model):
    """A service job with labor and parts estimates."""

    job_type = rf.Enum("installation", "repair", "inspection")
    asset_age_years = rf.Number
    planned_units = rf.Number
    technician_hours = rf.Number(predict=True)
    parts_cost_usd = rf.Number(predict=True)`,
    dataset: {
      id: 'service', label: 'Service job',
      observation: 'One service job at planning time.',
      description: 'Known job requirements paired with separate labor-time and parts-cost answers.',
      entities: [{
        id: 'job', name: 'Service job', description: 'The job context shared by both prediction targets.',
        columns: [
          { name: 'job_type', type: 'string', description: 'Known job category: installation, repair, or inspection.' },
          { name: 'asset_age_years', type: 'number', description: 'Age of the asset in years when the job is planned.' },
          { name: 'planned_units', type: 'number', description: 'Number of units included in the planned job.' },
          { name: 'technician_hours', type: 'number', role: 'target', description: 'Observed technician labor time in hours. Learned jointly with parts cost as a separate answer.' },
          { name: 'parts_cost_usd', type: 'number', role: 'target', description: 'Observed parts cost in US dollars. Omitted from new prediction requests along with the labor-time answer.' },
        ],
      }],
      relations: [],
    },
    inputs: { job_type: 'repair', asset_age_years: 7.5, planned_units: 3 },
    answers: { technician_hours: 5, parts_cost_usd: 180 },
    output: {
      predictions: {
        technician_hours: prediction(4.7),
        parts_cost_usd: prediction(164.2),
      },
    },
    outputNote: 'Each named field has its own numeric content, expressed in the units used for that field’s labels.',
    takeaway: 'Several answers can learn from shared context. Evaluate time and cost separately; a useful result on one target does not establish quality on the other.',
  },
];

