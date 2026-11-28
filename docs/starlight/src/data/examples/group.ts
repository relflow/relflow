import { prediction, type TypeExample } from '../type-example.ts';

export const groupExamples: TypeExample[] = [
  {
    id: 'payment',
    label: 'Shared behavior',
    summary: 'Use a merchant as context',
    title: 'Will this payment be disputed?',
    description: 'Learn shared representations for repeated merchant identities through a downstream prediction target.',
    dataset: {
      id: 'payment', label: 'Merchant payment context',
      observation: 'One payment at its prediction cutoff.',
      description: 'Known merchant identity and payment context support a historical dispute outcome.',
      entities: [{
        id: 'payment', name: 'Payment', description: 'A payment whose visible merchant identity supplies Group context.',
        columns: [
          { name: 'merchant', type: 'string', description: 'Known merchant identity, such as merchant-17. Group uses the identity as context; it is not a generated target.' },
          { name: 'amount', type: 'number', description: 'Known payment amount in the dataset’s currency units.' },
          { name: 'channel', type: 'string', description: 'Known payment channel, either web or store.' },
          { name: 'disputed', type: 'boolean', role: 'target', description: 'Recorded dispute outcome for a historical payment. New prediction requests omit this answer.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf


class Payment(rf.Model):
    """A payment and its observed dispute outcome."""

    merchant = rf.Group(n_groups=12)
    amount = rf.Number
    channel = rf.Enum("web", "store")
    disputed = rf.Boolean(predict=True)`,
    inputs: { merchant: 'merchant-17', amount: 84.5, channel: 'web' },
    answers: { disputed: false },
    output: { predictions: { disputed: prediction({ probability: 0.12 }) } },
    outputNote: 'The output is the dispute target’s probability of true. The merchant Group supplies input context and is not a decoded target in this schema.',
    takeaway: 'The dispute objective can train the visible merchant field’s assignment logits. Identities share group representations; the resulting groups need not have human-readable meanings.',
  },
  {
    id: 'history',
    label: 'Within a history',
    summary: 'Predict future spending',
    title: 'How much might this customer spend next month?',
    description: 'Keep merchant identities inside purchase records and learn a numeric outcome from their shared context.',
    dataset: {
      id: 'history', label: 'Customer purchase history',
      observation: 'One customer at a prediction cutoff.',
      description: 'Known past purchases provide context for the customer’s spending during the following 30 days.',
      entities: [
        {
          id: 'customer', name: 'Customer review', description: 'One cutoff with its past purchases and future spending answer.',
          columns: [
            { name: 'purchases', type: 'records[]', role: 'collection', description: 'Past purchases supplied oldest to newest, excluding purchases after the prediction cutoff.' },
            { name: 'spend_next_30d', type: 'number', role: 'target', description: 'Recorded spending in the 30 days after the cutoff, in the dataset’s currency units. This historical answer is absent from prediction requests.' },
          ],
        },
        {
          id: 'purchase', name: 'Past purchase', description: 'One item in the purchases history.',
          columns: [
            { name: 'merchant', type: 'string', description: 'Merchant identity for this past purchase, used as visible Group context.' },
            { name: 'amount', type: 'number', description: 'Known amount of this past purchase, in the dataset’s currency units.' },
            { name: 'days_ago', type: 'number', description: 'Days from this purchase to the prediction cutoff.' },
          ],
        },
      ],
      relations: [{ from: 'customer', to: 'purchase', fromColumn: 'purchases', description: 'Each customer review retains its own purchase history before the cutoff.' }],
    },
    schema: `import relflow as rf


class Purchase(rf.Branch):
    """A purchase known at the prediction cutoff."""

    merchant = rf.Group(n_groups=8)
    amount = rf.Number
    days_ago = rf.Number


class Customer(rf.Model):
    """A purchase history and future spending target."""

    purchases = Purchase(length=32, overflow="tail")
    spend_next_30d = rf.Number(predict=True)`,
    inputs: {
      purchases: [
        { merchant: 'merchant-17', amount: 46, days_ago: 20 },
        { merchant: 'merchant-31', amount: 18, days_ago: 4 },
      ],
    },
    answers: { spend_next_30d: 120 },
    output: { predictions: { spend_next_30d: prediction(112.4) } },
    outputNote: 'The numeric content uses the spending labels’ currency units. No merchant labels or group assignments are generated in this prediction output.',
    takeaway: 'A downstream objective can learn through a Group field inside a branch. Supply purchase history oldest to newest when retaining its tail, and keep future purchases out of the context.',
  },
  {
    id: 'offers',
    label: 'Every candidate',
    summary: 'Score supplied seller offers',
    title: 'Which seller offers is a customer likely to buy?',
    description: 'Use grouped seller identities alongside price and delivery context to learn an outcome for each supplied offer.',
    dataset: {
      id: 'offers', label: 'Seller offer review',
      observation: 'One product-category review containing the candidate offers to score.',
      description: 'Shared product context accompanies each seller offer and its historical purchase outcome.',
      entities: [
        {
          id: 'review', name: 'Offer review', description: 'Shared product context and the supplied candidate offers.',
          columns: [
            { name: 'product_category', type: 'string', description: 'Product category shared by these candidate offers, such as stationery.' },
            { name: 'offers', type: 'records[]', role: 'collection', description: 'Candidate offers supplied by the caller, each retaining its own purchase answer.' },
          ],
        },
        {
          id: 'offer', name: 'Seller offer', description: 'One item in the offers collection.',
          columns: [
            { name: 'seller', type: 'string', description: 'Known seller identity used as Group context; no seller identity is generated by this model.' },
            { name: 'price', type: 'number', description: 'Offered price in the dataset’s currency units.' },
            { name: 'delivery_days', type: 'number', description: 'Delivery time quoted for this offer, in days.' },
            { name: 'purchased', type: 'boolean', role: 'target', description: 'Historical purchase outcome for this candidate offer. Each supplied offer has its own Boolean answer, omitted from new requests.' },
          ],
        },
      ],
      relations: [{ from: 'review', to: 'offer', fromColumn: 'offers', description: 'Each review supplies candidate offers with purchase answers aligned to those offers.' }],
    },
    schema: `import relflow as rf


class Offer(rf.Branch):
    """A seller offer with a purchase target."""

    seller = rf.Group(n_groups=6)
    price = rf.Number
    delivery_days = rf.Number
    purchased = rf.Boolean(predict=True)


class OfferReview(rf.Model):
    """Offers scored together for one product category."""

    product_category = rf.Category
    offers = Offer(length=8, overflow="error")`,
    inputs: {
      product_category: 'stationery',
      offers: [
        { seller: 'seller-17', price: 24, delivery_days: 2 },
        { seller: 'seller-31', price: 20, delivery_days: 5 },
      ],
    },
    answers: { offers: [{ purchased: false }, { purchased: true }] },
    output: {
      predictions: {
        offers: [
          { purchased: prediction({ probability: 0.42 }) },
          { purchased: prediction({ probability: 0.63 }) },
        ],
      },
    },
    outputNote: 'The probabilities follow the supplied offer order. Each score belongs to its own seller, price, and delivery context.',
    takeaway: 'The group count stays fixed as the identity vocabulary grows. Per-identity assignment storage still grows with that vocabulary; compare held-out quality and resource use with Category.',
  },
];
