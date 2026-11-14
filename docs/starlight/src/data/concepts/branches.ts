import { stringify } from 'yaml';
import type { Concept, CaseId } from './types';
import { scenarios } from './scenarios';

const lessons: Record<CaseId, { title: string; description: string; placement: string; fields: string[]; note: string; takeaway: string }> = {
  tickets: {
    title: 'Keep each message’s author, body, and timing together',
    description: 'The messages collection contains whole message records. A branch keeps each body attached to the author and elapsed time that belong to it.',
    placement: 'messages = TicketMessage(length=64, overflow="tail")',
    fields: ['messages'],
    note: 'This placement belongs inside TicketModel. Supply messages oldest to newest when the tail should retain the latest conversation; relflow does not sort them.',
    takeaway: 'The cap retains up to 64 whole messages. It does not take 64 independent authors, bodies, and timestamps that could lose their alignment.',
  },
  churn: {
    title: 'Give each history its own collection',
    description: 'Monthly usage and support tickets have different record shapes and lengths. Their positions do not imply a one-to-one match across the two collections.',
    placement: 'monthly_usage = MonthlyUsage(length=12, overflow="tail")\nsupport_tickets = SupportTicket(length=32, overflow="tail")',
    fields: ['monthly_usage', 'support_tickets'],
    note: 'These placements belong inside ChurnModel. Each branch retains its own tail; the two usage rows below are not zipped with the single support ticket.',
    takeaway: 'Branches preserve relationships within a record and expose each collection as context for the parent. Cross-history joins remain an explicit data-preparation decision.',
  },
  demand: {
    title: 'Separate observed history from supplied forecast slots',
    description: 'The history branch contains known sales. The horizon branch contains future weeks whose numeric answers are hidden targets.',
    placement: 'history = SalesWeek(length=104, overflow="tail")\nhorizon = ForecastWeek(length=12, overflow="error")',
    fields: ['history', 'horizon'],
    note: 'These placements belong inside DemandModel. The history keeps its final records; a horizon longer than 12 supplied weeks raises an error rather than silently losing requested forecasts.',
    takeaway: 'A branch can supply context, contain targets, or do both. The model returns forecasts aligned with the retained horizon records; the cap does not request 12 generated weeks.',
  },
  credit: {
    title: 'Use different overflow choices for different records',
    description: 'Existing accounts form an inventory, while recent inquiries form a history. Their limits and overflow policies express distinct application choices.',
    placement: 'accounts = CreditAccount(length=64, overflow="error")\ninquiries = CreditInquiry(length=32, overflow="tail")',
    fields: ['accounts', 'inquiries'],
    note: 'These placements belong inside CreditModel. More than 64 accounts rejects the observation. Supply inquiries oldest to newest if the final 32 should be the most recent.',
    takeaway: 'Choose a cap according to the context the decision requires and the resources it can use. A branch policy chooses which supplied records survive; it does not assemble or order them.',
  },
  maintenance: {
    title: 'A reading count is a resource bound, not a time window',
    description: 'Sensor readings and work orders retain different amounts of context. Each record keeps its own measurements or maintenance fields intact.',
    placement: 'sensor_readings = SensorReading(length=168, overflow="tail")\nwork_orders = WorkOrder(length=32, overflow="tail")',
    fields: ['sensor_readings', 'work_orders'],
    note: 'These placements belong inside MaintenanceModel. A limit of 168 means 168 readings, even if sampling is irregular. Apply a time-window filter in data preparation when a wall-clock interval is required.',
    takeaway: 'A branch’s cap counts records. The data source supplies their timing, ordering, and observation cutoff.',
  },
  claims: {
    title: 'Keep the nesting that gives each amount its owner',
    description: 'Billed items belong to invoices, invoices belong to claims, and claims belong to a policy observation. Every level has a separate collection contract.',
    placement: 'line_items = LineItem(length=32, overflow="error")\ninvoices = Invoice(length=16, overflow="error")\nclaims = Claim(length=32, overflow="error")',
    fields: ['claims'],
    note: 'These three lines belong to Invoice, Claim, and ClaimsModel respectively. Each cap applies within its own parent record, and an excess collection raises an error at that level.',
    takeaway: 'Nested branches preserve which provider billed each item for which claim. Resource use depends on the combined retained structure, not just the outer claim count.',
  },
};

export const branches: Concept = {
  id: 'branches',
  title: 'Branches',
  description: 'Keep repeated records aligned and control how their context reaches the rest of a model.',
  intro: [
    'A Branch describes a collection of records. Each record keeps its related fields and nested children together, so an amount stays attached to its merchant, a message to its author, and an invoice item to its claim.',
    'The branch builds local context from those records. Its reduction controls how that context is passed to the parent; targets can also live inside a branch to request an answer for each supplied record.',
  ],
  principles: [
    { title: 'Record boundaries carry meaning', text: 'Cropping and batching retain whole records with their children aligned. Separate branches describe separate collections; matching array positions across branches do not create a join.' },
    { title: 'Local context has a parent-facing representation', text: 'Attention reduction uses learned context vectors, while Mean averages present encoded coordinates. Reduction is separate from attention within the branch; changing the reducer is an architecture choice to evaluate on your task.' },
    { title: 'Bound the context you intend to use', text: 'A positive length caps the retained records in each collection. Omitting the cap allows width to follow the largest collection in a batch. Finite caps still execute at the observed width and do not pad prediction lists to the cap.' },
    { title: 'Selection follows an explicit policy', text: 'tail keeps final source records, while error rejects overflow. With a finite cap, random takes a contiguous training window and the tail during evaluation and prediction; the source must supply a larger, separately bounded candidate pool for that window to vary.' },
  ],
  examples: scenarios.map((scenario) => {
    const lesson = lessons[scenario.id];
    const input = Object.fromEntries(lesson.fields.map((field) => [field, scenario.inputs[field]]));
    return {
      caseId: scenario.id,
      title: lesson.title,
      description: lesson.description,
      blocks: [
        { label: 'Branch placements from the schema', lang: 'python' as const, code: lesson.placement, note: lesson.note },
        { label: 'The supplied record collections', lang: 'yaml' as const, records: true, code: stringify(input, { indent: 2, lineWidth: 0 }), note: 'This is an excerpt of the shared illustrative input record. The complete schema shows the record classes used by these placements.' },
      ],
      takeaway: lesson.takeaway,
    };
  }),
  takeaway: 'Use branches to preserve the records that belong together, then choose explicit retention and context policies for each collection.',
  links: [
    { label: 'Choose the values inside each record', href: '/concepts/data-types/' },
    { label: 'Declare prediction targets', href: '/concepts/targets/' },
    { label: 'Assemble your observation', href: '/data/' },
    { label: 'Look up Branch and reduction APIs', href: '/api-map/' },
  ],
};
