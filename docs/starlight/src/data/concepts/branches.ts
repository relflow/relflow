import { stringify } from 'yaml';
import type { Concept, CaseId } from './types';
import { scenarios } from './scenarios';

const lessons: Record<CaseId, { title: string; description: string; fields: string[]; note: string; takeaway: string }> = {
  tickets: {
    title: 'Keep each message’s author, body, and timing together',
    description: 'The messages collection contains whole message records. A branch keeps each body attached to the author and elapsed time that belong to it.',
    fields: ['messages'],
    note: 'The messages branch retains the latest 64 records when supplied oldest to newest. Each TicketMessage keeps its author, body, and elapsed time together.',
    takeaway: 'The branch retains whole message records, keeping each author, body, and elapsed time aligned.',
  },
  churn: {
    title: 'Give each history its own collection',
    description: 'Monthly usage and support tickets have different record shapes and lengths. Their positions do not imply a one-to-one match across the two collections.',
    fields: ['monthly_usage', 'support_tickets'],
    note: 'Monthly usage and support tickets each retain their own history. The two usage rows below and the single support ticket belong to separate collections.',
    takeaway: 'Branches preserve relationships within a record and expose each collection as context for the parent. Cross-history joins remain an explicit data-preparation decision.',
  },
  demand: {
    title: 'Separate observed history from supplied forecast slots',
    description: 'The history branch contains known sales. The horizon branch contains future weeks whose numeric answers are hidden targets.',
    fields: ['history', 'horizon'],
    note: 'The history branch retains the latest observed weeks. The horizon branch accepts up to 12 supplied future weeks, with an error for excess records.',
    takeaway: 'A branch can supply context, contain targets, or do both. The model returns forecasts aligned with the retained horizon records; the cap does not request 12 generated weeks.',
  },
  credit: {
    title: 'Combine accounts and inquiries in one application',
    description: 'Existing accounts form an inventory, while recent inquiries form a history. Their limits and overflow policies express distinct application choices.',
    fields: ['accounts', 'inquiries'],
    note: 'The accounts branch accepts up to 64 records, with an error for excess accounts. The inquiries branch retains the latest 32 when supplied oldest to newest.',
    takeaway: 'Choose a cap according to the context the decision requires and the resources it can use. A branch policy chooses which supplied records survive; it does not assemble or order them.',
  },
  maintenance: {
    title: 'Keep telemetry and maintenance in their own histories',
    description: 'Sensor readings and work orders retain different amounts of context. Each record keeps its own measurements or maintenance fields intact.',
    fields: ['sensor_readings', 'work_orders'],
    note: 'The sensor_readings and work_orders branches have separate history limits. A limit of 168 means 168 readings, even if sampling is irregular.',
    takeaway: 'A branch’s cap counts records. The data source supplies their timing, ordering, and observation cutoff.',
  },
  claims: {
    title: 'Keep the nesting that gives each amount its owner',
    description: 'Billed items belong to invoices, invoices belong to claims, and claims belong to a policy observation. Every level has a separate collection contract.',
    fields: ['claims'],
    note: 'ClaimsModel contains Claim records; Claim contains Invoice records; Invoice contains LineItem records. Each cap applies within its own parent record.',
    takeaway: 'Nested branches preserve which provider billed each item for which claim. Resource use depends on the combined retained structure, not just the outer claim count.',
  },
};

export const branches: Concept = {
  id: 'branches',
  title: 'Context branches',
  description: 'Organize collections of related records and their nested context.',
  intro: [],
  principles: [
    { title: 'Related values stay together', text: 'A branch keeps the fields of each record aligned. An amount stays attached to its invoice item, a message to its author, and a reading to its timestamp.' },
    { title: 'Collections can have different structures', text: 'A model can combine several branches with different fields and lengths. Monthly usage and support tickets each keep their own records and contribute context to the same subscriber observation.' },
    { title: 'Nested limits apply within each parent', text: 'An invoice limit applies separately to each claim. A line-item limit applies separately to each invoice. Each level of the hierarchy retains its own record boundaries.' },
  ],
  examples: scenarios.map((scenario) => {
    const lesson = lessons[scenario.id];
    const input = Object.fromEntries(lesson.fields.map((field) => [field, scenario.inputs[field]]));
    return {
      caseId: scenario.id,
      title: lesson.title,
      description: lesson.description,
      blocks: [
        { label: 'Branches in the class schema', lang: 'python' as const, code: scenario.schema, note: lesson.note },
        { label: 'The supplied record collections', lang: 'yaml' as const, records: true, code: stringify(input, { indent: 2, lineWidth: 0 }), note: 'This is an excerpt of the shared illustrative input record. The class schema shows how these collections fit within the complete observation.' },
      ],
      takeaway: lesson.takeaway,
    };
  }),
  takeaway: 'Use branches to describe the collections in your data, nest them where records belong inside other records, and choose how much context to retain.',
  links: [
    { label: 'Choose the values inside each record', href: '/concepts/data-types/' },
    { label: 'Declare prediction targets', href: '/concepts/targets/' },
    { label: 'Assemble and load observations', href: '/advanced/data-modules/' },
    { label: 'Look up Branch and reduction APIs', href: '/api-map/' },
  ],
};
