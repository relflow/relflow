import { prediction, type TypeExample } from '../type-example.ts';

/** Static contract illustrations, never executed predictions or measurements. */
export const enumExamples: TypeExample[] = [
  {
    id: 'routing',
    label: 'Declared labels',
    summary: 'Route to a known queue',
    title: 'Which of our three queues should receive this ticket?',
    description: 'Declare the complete destination contract before training, while learning which context supports each outcome.',
    dataset: {
      id: 'routing', label: 'Ticket queue assignment',
      observation: 'One support ticket at routing time.',
      description: 'Known ticket context and a historical destination chosen from the three declared queues.',
      entities: [{
        id: 'ticket', name: 'Support ticket', description: 'One ticket whose queue is the prediction target.',
        columns: [
          { name: 'account_tier', type: 'string', description: 'Known account tier: free, standard, or enterprise.' },
          { name: 'topic', type: 'string', description: 'Topic known at routing time, such as api_timeout.' },
          { name: 'hours_open', type: 'number', description: 'Hours the ticket has been open when it is routed.' },
          { name: 'destination', type: 'string', role: 'target', description: 'Historical destination: billing, technical, or account. New prediction requests omit this answer.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf


class Ticket(rf.Model):
    """A support ticket routed to one of three declared queues."""

    account_tier = rf.Enum("free", "standard", "enterprise")
    topic = rf.Category
    hours_open = rf.Number
    destination = rf.Enum("billing", "technical", "account", mask=True, topk=[3])`,
    inputs: { account_tier: 'enterprise', topic: 'api_timeout', hours_open: 6 },
    answers: { destination: 'technical' },
    output: {
      predictions: {
        destination: prediction({
          value: 'technical', probability: 0.68,
          topk: [
            { value: 'technical', probability: 0.68 },
            { value: 'billing', probability: 0.21 },
            { value: 'account', probability: 0.11 },
          ],
        }),
      },
    },
    outputNote: 'All three declared labels appear in this full ranked distribution. Their probabilities sum to one, and the selected value is the highest-probability label.',
    takeaway: 'Declaring a label makes it an allowed class. Training examples still teach the model when that class is appropriate; unknown non-null labels are rejected.',
  },
  {
    id: 'priorities',
    label: 'Every task',
    summary: 'Assign nested priorities',
    title: 'What priority should each maintenance task receive?',
    description: 'Reuse the same declared priority levels for the target on every supplied task.',
    dataset: {
      id: 'priorities', label: 'Maintenance work queue',
      observation: 'One site review containing the maintenance tasks to prioritize.',
      description: 'Available staffing is shared context; each supplied task has its own priority answer.',
      entities: [
        {
          id: 'site', name: 'Site review', description: 'Shared staffing context and the supplied work queue.',
          columns: [
            { name: 'staffed_technicians', type: 'number', description: 'Number of technicians staffed when the queue is reviewed.' },
            { name: 'tasks', type: 'records[]', role: 'collection', description: 'Tasks being prioritized together, each retaining its own context and answer.' },
          ],
        },
        {
          id: 'task', name: 'Maintenance task', description: 'One item in the tasks collection.',
          columns: [
            { name: 'asset_family', type: 'string', description: 'Equipment family associated with this task, such as pump or fan.' },
            { name: 'hours_since_report', type: 'number', description: 'Hours elapsed since this task was reported, as known at review.' },
            { name: 'asset_available', type: 'boolean', description: 'Whether the asset is available when priorities are assigned.' },
            { name: 'priority', type: 'string', role: 'target', description: 'Historical assigned priority: low, normal, or urgent. These are declared class labels, with a separate answer for each task.' },
          ],
        },
      ],
      relations: [{ from: 'site', to: 'task', fromColumn: 'tasks', description: 'Each site review supplies maintenance tasks with aligned priority answers.' }],
    },
    schema: `import relflow as rf


class Task(rf.Branch):
    """A maintenance task assigned a declared priority."""

    asset_family = rf.Category
    hours_since_report = rf.Number
    asset_available = rf.Boolean
    priority = rf.Enum("low", "normal", "urgent", mask=True, topk=[3])


class WorkQueue(rf.Model):
    """Maintenance tasks considered together at one site."""

    staffed_technicians = rf.Number
    tasks = Task(length=16, overflow="error")`,
    inputs: {
      staffed_technicians: 4,
      tasks: [
        { asset_family: 'pump', hours_since_report: 8, asset_available: false },
        { asset_family: 'fan', hours_since_report: 2, asset_available: true },
      ],
    },
    answers: { tasks: [{ priority: 'urgent' }, { priority: 'normal' }] },
    output: {
      predictions: {
        tasks: [
          {
            priority: prediction({
              value: 'urgent', probability: 0.83,
              topk: [
                { value: 'urgent', probability: 0.83 },
                { value: 'normal', probability: 0.13 },
                { value: 'low', probability: 0.04 },
              ],
            }),
          },
          {
            priority: prediction({
              value: 'normal', probability: 0.63,
              topk: [
                { value: 'normal', probability: 0.63 },
                { value: 'low', probability: 0.29 },
                { value: 'urgent', probability: 0.08 },
              ],
            }),
          },
        ],
      },
    },
    outputNote: 'The two priority results follow the two supplied tasks. Each task has its own distribution over the same three declared labels.',
    takeaway: 'The words low, normal, and urgent are class labels. Their declaration does not impose a numeric distance between the priorities.',
  },
  {
    id: 'codes',
    label: 'Reusable codes',
    summary: 'Share a fixed declaration',
    title: 'Which configured service code fits the shipment?',
    description: 'Unpack reusable integer labels into a fixed classifier, without treating their values as measurements.',
    dataset: {
      id: 'codes', label: 'Shipment service selection',
      observation: 'One shipment when its service is selected.',
      description: 'Known shipment properties and a historical service code from the configured set.',
      entities: [{
        id: 'shipment', name: 'Shipment', description: 'A shipment assigned one declared service identity.',
        columns: [
          { name: 'weight_kg', type: 'number', description: 'Shipment weight in kilograms.' },
          { name: 'distance_km', type: 'number', description: 'Shipment distance in kilometers.' },
          { name: 'is_fragile', type: 'boolean', description: 'Whether the shipment is marked fragile when selecting a service.' },
          { name: 'service_code', type: 'number', role: 'target', description: 'Historical code 10, 20, or 30. The code is a declared identity rather than a measurement; decoded labels are written as strings.' },
        ],
      }],
      relations: [],
    },
    schema: `import relflow as rf

SERVICE_CODES = (10, 20, 30)


class Shipment(rf.Model):
    """A shipment assigned one of the configured service codes."""

    weight_kg = rf.Number
    distance_km = rf.Number
    is_fragile = rf.Boolean
    service_code = rf.Enum(*SERVICE_CODES, mask=True, topk=[3])`,
    inputs: { weight_kg: 2.4, distance_km: 680, is_fragile: true },
    answers: { service_code: 20 },
    output: {
      predictions: {
        service_code: prediction({
          value: '20', probability: 0.71,
          topk: [
            { value: '20', probability: 0.71 },
            { value: '30', probability: 0.18 },
            { value: '10', probability: 0.11 },
          ],
        }),
      },
    },
    outputNote: 'The declared integer labels are written as strings in decoded results. All three allowed codes appear because topk=[3] covers the whole declaration.',
    takeaway: 'The label tuple is reusable application configuration. Every non-null service_code must belong to it, and adding a declaration does not supply training examples.',
  },
];
