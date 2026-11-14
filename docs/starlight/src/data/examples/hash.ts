import { prediction, type TypeExample } from '../type-example.ts';

export const hashExamples: TypeExample[] = [
  {
    id: 'account',
    label: 'Match identities',
    summary: 'Relate an actor to an owner',
    title: 'Does this account activity need review?',
    description: 'Keep the account owner and event actors as identifiers so the model can learn from their matching relationships.',
    dataset: {
      id: 'account', label: 'Account activity',
      observation: 'One account with its recent events at review.',
      description: 'The owner identifier and event actor identifiers provide matching context for the review target.',
      entities: [
        { id: 'account', name: 'Account', description: 'The account context and historical review label.', columns: [
          { name: 'owner_id', type: 'string', description: 'Account owner identifier, compared with actor_id in the supplied events through consistently configured Hash fields.' },
          { name: 'events', type: 'records[]', role: 'collection', description: 'Events known at review, supplied oldest to newest so tail retention keeps the most recent activity.' },
          { name: 'needs_review', type: 'boolean', role: 'target', description: 'Historical review label for the account. Hidden from model inputs and omitted from new prediction requests.' },
        ] },
        { id: 'event', name: 'Event', description: 'One activity record in events.', columns: [
          { name: 'actor_id', type: 'string', description: 'Identifier of the actor associated with this event. Equality with owner_id is available as context.' },
          { name: 'action', type: 'string', description: 'Action recorded for this event, such as login or payment.' },
          { name: 'hours_ago', type: 'number', description: 'Hours elapsed between this event and the account review cutoff.' },
        ] },
      ],
      relations: [{ from: 'account', to: 'event', fromColumn: 'events', cardinality: 'many', description: 'Each account observation contains its own recent event records.' }],
    },
    schema: `import relflow as rf


class Event(rf.Branch):
    """An account event attributed to one actor."""

    actor_id = rf.Hash
    action = rf.Category
    hours_ago = rf.Number


class Account(rf.Model):
    """An account and its recent activity."""

    owner_id = rf.Hash
    events = Event(length=64, overflow="tail")
    needs_review = rf.Boolean(mask=True)`,
    inputs: {
      owner_id: 'person-17',
      events: [
        { actor_id: 'person-17', action: 'login', hours_ago: 4 },
        { actor_id: 'person-17', action: 'payment', hours_ago: 1 },
      ],
    },
    answers: { needs_review: false },
    output: { predictions: { needs_review: prediction({ probability: 0.18 }) } },
    outputNote: 'The Boolean target returns the probability that review is needed. The identifier fields provide context and are not decoded into new IDs.',
    takeaway: 'Matching identifiers share hash content within the encoded batch. The model can learn whether the same actor appears as the owner and in the supplied events.',
  },
  {
    id: 'candidates',
    label: 'Across collections',
    summary: 'Connect views with candidates',
    title: 'Which supplied products is this customer likely to buy?',
    description: 'Match identifiers between a viewing history and a separate list of candidate products.',
    dataset: {
      id: 'candidates', label: 'Product candidates',
      observation: 'One viewing history and the supplied products to score.',
      description: 'Two collections retain viewed and candidate product identities, with a separate purchase label on each candidate.',
      entities: [
        { id: 'review', name: 'Product review', description: 'The complete observation scored together.', columns: [
          { name: 'views', type: 'records[]', role: 'collection', description: 'Products viewed before the prediction cutoff, supplied oldest to newest.' },
          { name: 'candidates', type: 'records[]', role: 'collection', description: 'Products selected by the caller for scoring. Prediction preserves these supplied positions.' },
        ] },
        { id: 'view', name: 'View', description: 'One prior interaction in views.', columns: [
          { name: 'product_id', type: 'string', description: 'Identifier of the viewed product. Hash preserves equality with matching product_id values in candidates.' },
          { name: 'minutes_ago', type: 'number', description: 'Minutes elapsed since this view at the prediction cutoff.' },
        ] },
        { id: 'candidate', name: 'Candidate', description: 'One supplied product in candidates.', columns: [
          { name: 'product_id', type: 'string', description: 'Identifier of this candidate product, comparable with the identifiers in views.' },
          { name: 'price', type: 'number', description: 'Candidate price known when scoring, in the dataset’s currency units.' },
          { name: 'purchased', type: 'boolean', role: 'target', description: 'Historical purchase outcome for this candidate. The label is omitted when scoring new candidates.' },
        ] },
      ],
      relations: [
        { from: 'review', to: 'view', fromColumn: 'views', cardinality: 'many', description: 'The review retains its viewing history as a separate collection.' },
        { from: 'review', to: 'candidate', fromColumn: 'candidates', cardinality: 'many', description: 'The review contains the candidate records that receive purchase predictions.' },
      ],
    },
    schema: `import relflow as rf


class View(rf.Branch):
    """A product viewed before the prediction cutoff."""

    product_id = rf.Hash
    minutes_ago = rf.Number


class Candidate(rf.Branch):
    """A supplied product and its purchase target."""

    product_id = rf.Hash
    price = rf.Number
    purchased = rf.Boolean(mask=True)


class ProductReview(rf.Model):
    """A viewing history and products to score."""

    views = View(length=32, overflow="tail")
    candidates = Candidate(length=8, overflow="error")`,
    inputs: {
      views: [
        { product_id: 'product-8', minutes_ago: 20 },
        { product_id: 'product-17', minutes_ago: 3 },
      ],
      candidates: [
        { product_id: 'product-17', price: 24 },
        { product_id: 'product-42', price: 38 },
      ],
    },
    answers: { candidates: [{ purchased: true }, { purchased: false }] },
    output: {
      predictions: {
        candidates: [
          { purchased: prediction({ probability: 0.76 }) },
          { purchased: prediction({ probability: 0.21 }) },
        ],
      },
    },
    outputNote: 'The two probabilities stay aligned with the two supplied candidates. No additional products or identifiers are generated.',
    takeaway: 'Hash relationships can cross branch boundaries. Keep record order and configure matching Hash fields consistently; the target remains an ordinary Boolean field.',
  },
  {
    id: 'stock',
    label: 'Numeric answer',
    summary: 'Read quantities in context',
    title: 'How many units are available for the requested part?',
    description: 'Use a requested identifier and a collection of stock records as context for a numeric target.',
    dataset: {
      id: 'stock', label: 'Part availability',
      observation: 'One requested part with relevant stock records.',
      description: 'The request and stock records retain their part identifiers while a numeric target teaches available quantity.',
      entities: [
        { id: 'availability', name: 'Availability request', description: 'The requested identifier, supplied stock context, and historical answer.', columns: [
          { name: 'requested_part', type: 'string', description: 'Identifier of the requested part, comparable with part_id in each stock record.' },
          { name: 'stock', type: 'records[]', role: 'collection', description: 'Stock quantities supplied as context, including the part identifier associated with each quantity.' },
          { name: 'available_units', type: 'number', role: 'target', description: 'Known available quantity used as a training label. New requests omit this answer; Number predicts an estimate.' },
        ] },
        { id: 'stock', name: 'Stock record', description: 'One part quantity in stock.', columns: [
          { name: 'part_id', type: 'string', description: 'Identifier of the part associated with this stock quantity. Equality with requested_part supplies context.' },
          { name: 'units', type: 'number', description: 'Available units reported by this stock record.' },
        ] },
      ],
      relations: [{ from: 'availability', to: 'stock', fromColumn: 'stock', cardinality: 'many', description: 'The request contains the stock records supplied for its availability estimate.' }],
    },
    schema: `import relflow as rf


class Stock(rf.Branch):
    """A part quantity at one stock location."""

    part_id = rf.Hash
    units = rf.Number


class Availability(rf.Model):
    """A requested part and relevant stock records."""

    requested_part = rf.Hash
    stock = Stock(length=32, overflow="error")
    available_units = rf.Number(mask=True)`,
    inputs: {
      requested_part: 'part-17',
      stock: [
        { part_id: 'part-17', units: 6 },
        { part_id: 'part-24', units: 8 },
      ],
    },
    answers: { available_units: 6 },
    output: { predictions: { available_units: prediction(5.7) } },
    outputNote: 'The numeric target returns an estimated quantity in units. A Hash field does not turn the prediction into an exact database lookup.',
    takeaway: 'Identity matching supplies one kind of context; the answer can have a different type. Number learns the quantity while Hash supplies identifier relationships.',
  },
];
