import { prediction, type TypeExample } from '../type-example.ts';

/** Static contract illustrations, never executed predictions or measurements. */
export const categoryExamples: TypeExample[] = [
  {
    id: 'shopping',
    label: 'Learned labels',
    summary: 'Anticipate a shopping category',
    title: 'Which category will this customer shop next?',
    description: 'Learn a vocabulary from training labels and return ranked alternatives for one mutually exclusive outcome.',
    dataset: {
      id: 'shopping', label: 'Customer shopping',
      observation: 'One customer at the prediction cutoff.',
      description: 'Known customer context and the next observed shopping category used as a historical answer.',
      entities: [{
        id: 'customer', name: 'Customer', description: 'A customer snapshot before their next shopping event.',
        columns: [
          { name: 'region', type: 'string', description: 'Customer region known at the cutoff, such as north.' },
          { name: 'preferred_channel', type: 'string', description: 'Preferred shopping channel known at the cutoff, such as app.' },
          { name: 'account_age_days', type: 'number', description: 'Account age in days when the prediction is made.' },
          { name: 'next_category', type: 'string', role: 'target', description: 'Category of the next observed shopping event. Historical labels teach the vocabulary; prediction requests omit this answer.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf


class Customer(rf.Model):
    """A customer whose next shopping category we want to predict."""

    region = rf.Category
    preferred_channel = rf.Category
    account_age_days = rf.Number
    next_category = rf.Category(mask=True, topk=[3])`,
    inputs: { region: 'north', preferred_channel: 'app', account_age_days: 240 },
    answers: { next_category: 'home' },
    output: {
      predictions: {
        next_category: prediction({
          value: 'home',
          probability: 0.62,
          topk: [
            { value: 'home', probability: 0.62 },
            { value: 'books', probability: 0.25 },
            { value: 'sports', probability: 0.13 },
          ],
        }),
      },
    },
    outputNote: 'This illustration assumes exactly three learned target labels, so its complete top-k distribution sums to one. The selected value repeats the highest-probability candidate.',
    takeaway: 'The schema does not list the shopping labels. Training discovers them; new prediction requests provide customer context without next_category.',
  },
  {
    id: 'tickets',
    label: 'Every request',
    summary: 'Route a collection of tickets',
    title: 'Which queue should receive each ticket?',
    description: 'Keep tickets as separate records and place a classification target inside each one.',
    dataset: {
      id: 'tickets', label: 'Ticket routing batch',
      observation: 'One customer account with the tickets being reviewed together.',
      description: 'Account context accompanies each supplied ticket, whose historical destination is a separate answer.',
      entities: [
        {
          id: 'account', name: 'Account review', description: 'The account-level context and its supplied tickets.',
          columns: [
            { name: 'account_tier', type: 'string', description: 'Service tier shared by the tickets in this account review.' },
            { name: 'tickets', type: 'records[]', role: 'collection', description: 'The supplied tickets to route, preserving one record and answer per ticket.' },
          ],
        },
        {
          id: 'ticket', name: 'Ticket', description: 'One item in the tickets collection.',
          columns: [
            { name: 'topic', type: 'string', description: 'Topic known when reviewing this ticket, such as duplicate_charge.' },
            { name: 'hours_open', type: 'number', description: 'Hours this ticket has been open at review.' },
            { name: 'destination', type: 'string', role: 'target', description: 'Historical destination queue for this ticket. The label is learned as a Category and omitted from prediction requests.' },
          ],
        },
      ],
      relations: [{ from: 'account', to: 'ticket', fromColumn: 'tickets', description: 'Each account review supplies a collection of tickets with aligned destination answers.' }],
    },
    schema: `import relflow as rf


class Ticket(rf.Branch):
    """A support ticket assigned to a learned destination queue."""

    topic = rf.Category
    hours_open = rf.Number
    destination = rf.Category(mask=True, topk=[3])


class SupportBatch(rf.Model):
    """Tickets reviewed together for one customer account."""

    account_tier = rf.Category
    tickets = Ticket(length=8, overflow="error")`,
    inputs: {
      account_tier: 'enterprise',
      tickets: [
        { topic: 'duplicate_charge', hours_open: 3 },
        { topic: 'password_reset', hours_open: 1 },
      ],
    },
    answers: { tickets: [{ destination: 'billing' }, { destination: 'account' }] },
    output: {
      predictions: {
        tickets: [
          {
            destination: prediction({
              value: 'billing', probability: 0.86,
              topk: [
                { value: 'billing', probability: 0.86 },
                { value: 'account', probability: 0.09 },
                { value: 'technical', probability: 0.05 },
              ],
            }),
          },
          {
            destination: prediction({
              value: 'account', probability: 0.91,
              topk: [
                { value: 'account', probability: 0.91 },
                { value: 'technical', probability: 0.06 },
                { value: 'billing', probability: 0.03 },
              ],
            }),
          },
        ],
      },
    },
    outputNote: 'Each ticket retains its own rich classification result, in input order. This example assumes a learned vocabulary containing exactly the three displayed queues.',
    takeaway: 'Shared account context and per-ticket facts contribute to predictions at the ticket level. Choose Enum when these queues should instead be a fixed schema contract.',
  },
  {
    id: 'codes',
    label: 'Numeric labels',
    summary: 'Predict a diagnostic code',
    title: 'Which fault code fits these sensor readings?',
    description: 'Represent integer codes as identities when their numerical spacing carries no meaning.',
    dataset: {
      id: 'codes', label: 'Machine diagnostics',
      observation: 'One machine assessment with its known sensor readings.',
      description: 'Known machine context and measurements are paired with a historical diagnostic identity.',
      entities: [{
        id: 'machine', name: 'Machine assessment', description: 'A diagnostic assessment before the fault code is supplied as an answer.',
        columns: [
          { name: 'machine_family', type: 'string', description: 'Equipment family, such as pump, known at assessment.' },
          { name: 'vibration_mm_s', type: 'number', description: 'Measured vibration in millimeters per second.' },
          { name: 'temperature_c', type: 'number', description: 'Measured temperature in degrees Celsius.' },
          { name: 'fault_code', type: 'number', role: 'target', description: 'Historical integer diagnostic code. Category treats it as an identity rather than a quantity; decoded labels are written as strings.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf


class Machine(rf.Model):
    """A machine whose diagnostic code is a categorical identity."""

    machine_family = rf.Category
    vibration_mm_s = rf.Number
    temperature_c = rf.Number
    fault_code = rf.Category(mask=True, topk=[3])`,
    inputs: { machine_family: 'pump', vibration_mm_s: 7.2, temperature_c: 84 },
    answers: { fault_code: 17 },
    output: {
      predictions: {
        fault_code: prediction({
          value: '17',
          probability: 0.74,
          topk: [
            { value: '17', probability: 0.74 },
            { value: '42', probability: 0.19 },
            { value: '93', probability: 0.07 },
          ],
        }),
      },
    },
    outputNote: 'Training labels here are integers, but decoded labels are written as strings. The illustration assumes exactly three learned fault codes.',
    takeaway: 'Category treats 17, 42, and 93 as different names rather than ordered measurements. Use Number for quantities whose magnitude matters.',
  },
];
