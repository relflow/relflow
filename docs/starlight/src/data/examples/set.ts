import { prediction, type TypeExample } from '../type-example.ts';

export const setExamples: TypeExample[] = [
  {
    id: 'interests',
    label: 'Several labels',
    summary: 'Learn overlapping interests',
    title: 'Which product categories might interest this customer?',
    description: 'Use label membership as context and learn several labels that can apply to the same customer.',
    dataset: {
      id: 'interests', label: 'Customer interests',
      observation: 'One customer before the future interests being predicted.',
      description: 'Past category memberships provide context for a future set of potentially overlapping interests.',
      entities: [{
        id: 'customer', name: 'Customer', description: 'One customer with known purchase categories and a multi-label answer.',
        columns: [
          { name: 'region', type: 'string', description: 'Customer region known when predicting, such as north.' },
          { name: 'purchased_categories', type: 'string[]', description: 'Known category memberships from past purchases. This scalar list is a Set input, not a collection of child records.' },
          { name: 'next_interests', type: 'string[]', role: 'target', description: 'Historical set of relevant future category labels. Several labels may apply together; new prediction requests omit the set.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf


class Customer(rf.Model):
    """A customer whose future product interests can overlap."""

    region = rf.Category
    purchased_categories = rf.Set
    next_interests = rf.Set(predict=True, threshold=0.5)`,
    inputs: { region: 'north', purchased_categories: ['books', 'home'] },
    answers: { next_interests: ['home', 'sports'] },
    output: {
      predictions: {
        next_interests: prediction([
          { value: 'books', probability: 0.61 },
          { value: 'home', probability: 0.87 },
          { value: 'sports', probability: 0.72 },
        ]),
      },
    },
    outputNote: 'Each pair is a known vocabulary label and its independent membership probability. These probabilities need not sum to one.',
    takeaway: 'One customer can have several interests. The 0.5 threshold filters the written pairs; it does not change the training objective.',
  },
  {
    id: 'support',
    label: 'Overlapping actions',
    summary: 'Triage a support request',
    title: 'Which follow-up actions does this ticket need?',
    description: 'Combine text meaning and account context to predict several useful actions rather than a single destination.',
    dataset: {
      id: 'support', label: 'Support follow-up actions',
      observation: 'One support request at triage.',
      description: 'The known request text and account tier support a historical set of follow-up actions.',
      entities: [{
        id: 'ticket', name: 'Support request', description: 'One request that can need several actions at once.',
        columns: [
          { name: 'message', type: 'string', description: 'Request text available at triage, used as Text input.' },
          { name: 'account_tier', type: 'string', description: 'Known account tier: free, standard, or enterprise.' },
          { name: 'actions', type: 'string[]', role: 'target', description: 'Historical follow-up action labels, such as billing_review and invoice_help. This Set answer can contain several labels and is absent from prediction requests.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf


class SupportTicket(rf.Model):
    """A support request that may need several follow-up actions."""

    message = rf.Text
    account_tier = rf.Enum("free", "standard", "enterprise")
    actions = rf.Set(predict=True, threshold=0.5)`,
    inputs: {
      message: 'I was charged twice and cannot download my invoices.',
      account_tier: 'standard',
    },
    answers: { actions: ['billing_review', 'invoice_help'] },
    output: {
      predictions: {
        actions: prediction([
          { value: 'billing_review', probability: 0.91 },
          { value: 'invoice_help', probability: 0.78 },
        ]),
      },
    },
    outputNote: 'Several actions can be relevant together. The model returns membership probabilities for learned action labels.',
    takeaway: 'Text supplies context; Set supplies the multi-label answer. Constructing this schema requires the relflow[text] dependencies and an available text encoder.',
  },
  {
    id: 'claims',
    label: 'Every record',
    summary: 'Predict nested label sets',
    title: 'Which checks does each submitted claim need?',
    description: 'Place the Set target inside a branch so each supplied claim has its own collection of predicted labels.',
    dataset: {
      id: 'claims', label: 'Claim review checks',
      observation: 'One policy review containing its submitted claims.',
      description: 'Policy context is shared, while each supplied claim has a separate set of historical review checks.',
      entities: [
        {
          id: 'policy', name: 'Policy review', description: 'Known policy context and the supplied claims.',
          columns: [
            { name: 'policy_age_days', type: 'number', description: 'Policy age in days at review.' },
            { name: 'claims', type: 'records[]', role: 'collection', description: 'Submitted claims retained as separate records, each with its own review-check answer.' },
          ],
        },
        {
          id: 'claim', name: 'Submitted claim', description: 'One item in the claims collection.',
          columns: [
            { name: 'claimed_amount', type: 'number', description: 'Amount requested by this claim, in the dataset’s currency units.' },
            { name: 'item_category', type: 'string', description: 'Category of the claimed item, such as electronics or clothing.' },
            { name: 'incident_age_days', type: 'number', description: 'Days since the incident when reviewing the claim.' },
            { name: 'review_checks', type: 'string[]', role: 'target', description: 'Historical check labels for this claim. An empty list is a supplied empty Set answer; it does not remove the claim record.' },
          ],
        },
      ],
      relations: [{ from: 'policy', to: 'claim', fromColumn: 'claims', description: 'Each policy review supplies claims whose review-check sets remain aligned with those claims.' }],
    },
    schema: `import relflow as rf


class Claim(rf.Branch):
    """A submitted claim and its required review checks."""

    claimed_amount = rf.Number
    item_category = rf.Category
    incident_age_days = rf.Number
    review_checks = rf.Set(predict=True, threshold=0.5)


class ClaimReview(rf.Model):
    """Claims reviewed together under one insurance policy."""

    policy_age_days = rf.Number
    claims = Claim(length=8, overflow="error")`,
    inputs: {
      policy_age_days: 240,
      claims: [
        { claimed_amount: 750, item_category: 'electronics', incident_age_days: 2 },
        { claimed_amount: 40, item_category: 'clothing', incident_age_days: 1 },
      ],
    },
    answers: {
      claims: [
        { review_checks: ['receipt_check', 'coverage_review'] },
        { review_checks: [] },
      ],
    },
    output: {
      predictions: {
        claims: [
          {
            review_checks: prediction([
              { value: 'receipt_check', probability: 0.89 },
              { value: 'coverage_review', probability: 0.67 },
            ]),
          },
          { review_checks: prediction([]) },
        ],
      },
    },
    outputNote: 'Each output record stays aligned with its supplied claim. The second claim’s content list is empty because no known check reaches the 0.5 threshold.',
    takeaway: 'The branch preserves claim boundaries, and each Set holds that claim’s label memberships. The model does not generate a new list of claims.',
  },
];
