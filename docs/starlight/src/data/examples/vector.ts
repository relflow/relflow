import { prediction, type TypeExample } from '../type-example.ts';

export const vectorExamples: TypeExample[] = [
  {
    id: 'sensor',
    label: 'Existing coordinates',
    summary: 'Use a sensor signature',
    title: 'Does this machine need an inspection?',
    description: 'Read an existing four-coordinate sensor signature alongside operating context.',
    dataset: {
      id: 'sensor', label: 'Sensor assessment',
      observation: 'One machine assessment with a supplied sensor signature.',
      description: 'A fixed-width numeric signature and operating history teach a Boolean inspection decision.',
      entities: [{ id: 'assessment', name: 'Machine assessment', description: 'Known sensor context and the historical inspection label.', columns: [
        { name: 'signature', type: 'number[]', description: 'Four sensor coordinates in the source’s fixed order and scale. The example does not assign physical meanings or units to those coordinates.' },
        { name: 'operating_hours', type: 'number', description: 'Machine operating hours known at assessment.' },
        { name: 'needs_inspection', type: 'boolean', role: 'target', description: 'Historical inspection label. New assessments omit this answer and receive a Boolean probability.' },
      ] }],
      relations: [],
    },
    schema: `import relflow as rf


class SensorAssessment(rf.Model):
    """A machine assessment with a fixed-width sensor signature."""

    signature = rf.Vector(n_dim=4)
    operating_hours = rf.Number
    needs_inspection = rf.Boolean(mask=True)`,
    inputs: { signature: [0.12, -0.4, 0.8, 0.03], operating_hours: 2400 },
    answers: { needs_inspection: true },
    output: { predictions: { needs_inspection: prediction({ probability: 0.87 }) } },
    outputNote: 'The signature is input context. This schema predicts a Boolean probability, rather than writing another vector.',
    takeaway: 'The source defines what the four coordinate positions mean. Every signature must use that same width, order, and scale; Vector learns a projection without fitting a normalizer.',
  },
  {
    id: 'position',
    label: 'A vector target',
    summary: 'Predict three spatial coordinates',
    title: 'Where will the tracked object be next?',
    description: 'Learn a future position from the current position, velocity, and requested time interval.',
    dataset: {
      id: 'position', label: 'Position forecast',
      observation: 'One tracked object at a requested future time interval.',
      description: 'Current coordinates and velocity provide context for a future three-coordinate position target.',
      entities: [{ id: 'forecast', name: 'Position forecast', description: 'Known motion context with the observed future position label.', columns: [
        { name: 'position_m', type: 'number[]', description: 'Current x, y, and z coordinates in metres, in that order.' },
        { name: 'velocity_m_per_s', type: 'number[]', description: 'Current velocity along x, y, and z in metres per second, using the same coordinate order as position_m.' },
        { name: 'seconds_ahead', type: 'number', description: 'Requested forecast interval in seconds, known when making the prediction.' },
        { name: 'next_position_m', type: 'number[]', role: 'target', description: 'Observed x, y, and z coordinates in metres after seconds_ahead. This fixed-width vector is one target, omitted from new forecast requests.' },
      ] }],
      relations: [],
    },
    schema: `import relflow as rf


class PositionForecast(rf.Model):
    """A tracked object with a future three-coordinate position."""

    position_m = rf.Vector(n_dim=3)
    velocity_m_per_s = rf.Vector(n_dim=3)
    seconds_ahead = rf.Number
    next_position_m = rf.Vector(n_dim=3, mask=True, objective="l1")`,
    inputs: { position_m: [10, 20, 5], velocity_m_per_s: [1, -0.5, 0], seconds_ahead: 2 },
    answers: { next_position_m: [12, 19, 5] },
    output: { predictions: { next_position_m: prediction([11.8, 19.2, 5.1]) } },
    outputNote: 'The target content is one fixed-width vector: x, y, and z in metres. It is not three state/probability results or a variable collection of positions.',
    takeaway: 'A Vector target learns from known coordinate arrays. The l1 objective averages absolute coordinate errors in the supplied units; it does not impose physical motion constraints.',
  },
  {
    id: 'stock',
    label: 'Nested mixed targets',
    summary: 'Forecast sales and stock risk',
    title: 'What might each product sell over the next three days?',
    description: 'Predict a fixed sales horizon and a Boolean stock-risk answer for every supplied product.',
    dataset: {
      id: 'stock', label: 'Warehouse stock forecast',
      observation: 'One warehouse with the products supplied for forecasting.',
      description: 'Each product keeps a three-day sales history and two aligned targets: its next sales vector and stock-risk label.',
      entities: [
        { id: 'forecast', name: 'Stock forecast', description: 'Warehouse context and the complete product collection.', columns: [
          { name: 'warehouse', type: 'string', description: 'Warehouse being assessed, such as north.' },
          { name: 'products', type: 'records[]', role: 'collection', description: 'Products supplied for review. The number of products may vary while each sales vector remains three coordinates wide.' },
        ] },
        { id: 'product', name: 'Product', description: 'One product in products at the forecast cutoff.', columns: [
          { name: 'sku', type: 'string', description: 'Product identifier for this supplied stock record.' },
          { name: 'on_hand', type: 'number', description: 'Units on hand at the forecast cutoff.' },
          { name: 'last_3_days_units', type: 'number[]', description: 'Three observed daily sales counts in the source’s consistent coordinate order, all known at the cutoff.' },
          { name: 'next_3_days_units', type: 'number[]', role: 'target', description: 'Observed units sold on the next day, second day, and third day, in that order. This historical vector label is omitted when forecasting.' },
          { name: 'runs_out', type: 'boolean', role: 'target', description: 'Historical stock-risk label learned alongside the sales vector for this product. Omitted from new prediction requests.' },
        ] },
      ],
      relations: [{ from: 'forecast', to: 'product', fromColumn: 'products', cardinality: 'many', description: 'Each warehouse observation contains the supplied product records that receive both predictions.' }],
    },
    schema: `import relflow as rf


class Product(rf.Branch):
    """A product with three-day sales and stock-risk targets."""

    sku = rf.Category
    on_hand = rf.Number
    last_3_days_units = rf.Vector(n_dim=3)
    next_3_days_units = rf.Vector(n_dim=3, mask=True)
    runs_out = rf.Boolean(mask=True)


class StockForecast(rf.Model):
    """Products reviewed together at one warehouse."""

    warehouse = rf.Category
    products = Product(length=32, overflow="error")`,
    inputs: {
      warehouse: 'north',
      products: [
        { sku: 'notebook', on_hand: 20, last_3_days_units: [7, 9, 8] },
        { sku: 'pencil', on_hand: 80, last_3_days_units: [4, 5, 3] },
      ],
    },
    answers: {
      products: [
        { next_3_days_units: [9, 10, 8], runs_out: true },
        { next_3_days_units: [4, 4, 5], runs_out: false },
      ],
    },
    output: {
      predictions: {
        products: [
          { next_3_days_units: prediction([8.6, 9.2, 8.1]), runs_out: prediction({ probability: 0.81 }) },
          { next_3_days_units: prediction([4.2, 4.4, 4.7]), runs_out: prediction({ probability: 0.06 }) },
        ],
      },
    },
    outputNote: 'Each supplied product gets a three-coordinate sales vector and a separate Boolean probability. Future coordinates mean the next day, second day, and third day, in that order.',
    takeaway: 'The product collection can vary in length, but every sales vector stays three coordinates wide. The numeric and Boolean objectives learn together without imposing agreement between their answers.',
  },
];
