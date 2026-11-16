import { prediction, type TypeExample } from '../type-example.ts';

export const textExamples: TypeExample[] = [
  {
    id: 'support',
    label: 'One decision',
    summary: 'Classify a support request',
    title: 'Does this message need escalation?',
    description: 'Combine the meaning of a customer message with numeric account context to learn a Boolean outcome.',
    dataset: {
      id: 'support', label: 'Support message',
      observation: 'One customer message when deciding whether to escalate.',
      description: 'The message and account context teach a Boolean escalation target.',
      entities: [{ id: 'ticket', name: 'Support ticket', description: 'The known message and account age with its historical decision label.', columns: [
        { name: 'message', type: 'string', description: 'Customer message available when scoring. Text supplies its frozen-encoder representation as input context.' },
        { name: 'account_age_days', type: 'number', description: 'Age of the customer’s account in days at the decision time.' },
        { name: 'needs_escalation', type: 'boolean', role: 'target', description: 'Historical escalation label. New prediction requests supply the message and account context without this answer.' },
      ] }],
      relations: [],
    },
    schema: `import relflow as rf


class SupportTicket(rf.Model):
    """A message and the escalation decision to learn."""

    message = rf.Text
    account_age_days = rf.Number
    needs_escalation = rf.Boolean(mask=True)`,
    inputs: {
      message: 'My account is locked and the reset email never arrives.',
      account_age_days: 240,
    },
    answers: { needs_escalation: true },
    output: { predictions: { needs_escalation: prediction({ probability: 0.84 }) } },
    outputNote: 'The answer is a Boolean probability. The text remains input context; this model does not generate a reply.',
    takeaway: 'The frozen text encoder provides a representation of the message. relflow learns how to use it with the rest of the record for the declared prediction target.',
  },
  {
    id: 'relevance',
    label: 'Every candidate',
    summary: 'Compare query and descriptions',
    title: 'Which supplied descriptions fit the request?',
    description: 'Place text at the root and inside a collection, then learn an answer for each candidate record.',
    dataset: {
      id: 'relevance', label: 'Search candidates',
      observation: 'One search request and its supplied candidate descriptions.',
      description: 'The request is shared context; each candidate retains its own description, price, and relevance label.',
      entities: [
        { id: 'search', name: 'Search review', description: 'The request and complete candidate set scored together.', columns: [
          { name: 'request', type: 'string', description: 'The caller’s search request, such as a description of the bag they want.' },
          { name: 'candidates', type: 'records[]', role: 'collection', description: 'Candidate records selected for scoring. Their order determines the aligned prediction positions.' },
        ] },
        { id: 'candidate', name: 'Candidate', description: 'One supplied candidate in candidates.', columns: [
          { name: 'description', type: 'string', description: 'Candidate description available when scoring. Its Text representation can be compared with the request context.' },
          { name: 'price', type: 'number', description: 'Known candidate price in the dataset’s currency units.' },
          { name: 'relevant', type: 'boolean', role: 'target', description: 'Historical relevance label for this candidate and request. Omitted when scoring a new candidate set.' },
        ] },
      ],
      relations: [{ from: 'search', to: 'candidate', fromColumn: 'candidates', cardinality: 'many', description: 'Each request contains the supplied candidates that receive relevance predictions.' }],
    },
    schema: `import relflow as rf


class Candidate(rf.Branch):
    """A candidate description and relevance label."""

    description = rf.Text
    price = rf.Number
    relevant = rf.Boolean(mask=True)


class SearchReview(rf.Model):
    """A search request and the candidates to assess."""

    request = rf.Text
    candidates = Candidate(length=8, overflow="error")`,
    inputs: {
      request: 'A small waterproof bag for a day hike.',
      candidates: [
        { description: 'Compact roll-top dry bag with shoulder straps.', price: 34 },
        { description: 'Large open woven beach tote.', price: 28 },
      ],
    },
    answers: { candidates: [{ relevant: true }, { relevant: false }] },
    output: {
      predictions: {
        candidates: [
          { relevant: prediction({ probability: 0.88 }) },
          { relevant: prediction({ probability: 0.19 }) },
        ],
      },
    },
    outputNote: 'Each candidate receives its own Boolean probability in the supplied order. The output does not generate descriptions or new candidates.',
    takeaway: 'Text can appear at different levels of a schema. A branch retains which description, price, and prediction belong to each candidate.',
  },
  {
    id: 'service',
    label: 'Several answers',
    summary: 'Learn time and urgency',
    title: 'How long might this repair take, and is it urgent?',
    description: 'Use a written request and equipment context to learn numeric and Boolean targets together.',
    dataset: {
      id: 'service', label: 'Repair request',
      observation: 'One reported equipment fault when requesting service.',
      description: 'Written fault context and equipment age teach labor-time and urgency targets together.',
      entities: [{ id: 'repair', name: 'Repair request', description: 'Known repair context with two historical outcome labels.', columns: [
        { name: 'description', type: 'string', description: 'Written fault report available before service. The Text field may reuse its frozen vectors from the configured store.' },
        { name: 'asset_age_years', type: 'number', description: 'Equipment age in years at the service request.' },
        { name: 'labor_hours', type: 'number', role: 'target', description: 'Observed labor hours used as a numeric training label. Omitted from new repair requests.' },
        { name: 'urgent', type: 'boolean', role: 'target', description: 'Historical urgency label learned with labor_hours from the same request context. Omitted when predicting.' },
      ] }],
      relations: [],
    },
    schema: `import relflow as rf


class RepairRequest(rf.Model):
    """A reported fault and the service outcomes to learn."""

    description = rf.Text(max_length=256, store="./request-vectors")
    asset_age_years = rf.Number
    labor_hours = rf.Number(mask=True)
    urgent = rf.Boolean(mask=True)`,
    inputs: {
      description: 'The compressor stops after ten minutes and restarts after cooling.',
      asset_age_years: 6,
    },
    answers: { labor_hours: 3.5, urgent: true },
    output: {
      predictions: {
        labor_hours: prediction(3.2),
        urgent: prediction({ probability: 0.81 }),
      },
    },
    outputNote: 'The numeric content is in labor hours, and urgent contains a Boolean probability. Both targets use the same supplied context.',
    takeaway: 'Text automatically reuses prepared text and frozen vectors; store chooses their persistent directory. The predictive heads still learn their own task from labeled examples; assess each answer separately.',
  },
];
