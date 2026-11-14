import type { Concept, CaseId } from './types';
import { scenarios } from './scenarios';

const lessons: Record<CaseId, { title: string; description: string; schemaNote: string; takeaway: string }> = {
  tickets: {
    title: 'A ticket with a branch of messages',
    description: 'TicketModel is the root. Its messages branch groups the author, text, and timing of each message within the conversation.',
    schemaNote: 'TicketMessage declares the message leaves. Placing it under messages makes those records part of the ticket’s context.',
    takeaway: 'The tree keeps each message’s fields together while making the conversation available to the ticket’s prediction targets.',
  },
  churn: {
    title: 'A subscriber with two histories',
    description: 'ChurnModel combines subscriber leaves with two branches: monthly usage and support tickets. Each history has its own record structure.',
    schemaNote: 'MonthlyUsage and SupportTicket define separate branches beneath the same subscriber root.',
    takeaway: 'Several collections can contribute context to one observation while retaining their own fields and record boundaries.',
  },
  demand: {
    title: 'A forecast with history and horizon branches',
    description: 'DemandModel groups a store-product observation with past sales and supplied future weeks. The units target is a leaf inside each forecast week.',
    schemaNote: 'SalesWeek describes observed history; ForecastWeek describes the supplied weeks that need predictions.',
    takeaway: 'Branches can contain known context, prediction targets, or both. A target’s position determines which record its answer belongs to.',
  },
  credit: {
    title: 'An application with account and inquiry branches',
    description: 'CreditModel is one application. Its root leaves describe the request, while account and inquiry branches preserve the applicant’s related records.',
    schemaNote: 'CreditAccount and CreditInquiry define different record types beneath the same application root.',
    takeaway: 'Each branch gives its related records a place in the model tree and contributes context to the application’s targets.',
  },
  maintenance: {
    title: 'An asset with readings and work orders',
    description: 'MaintenanceModel places sensor readings and completed work orders beneath one asset. Their leaves represent different kinds of measurement and history.',
    schemaNote: 'SensorReading contains calendar and numeric leaves. WorkOrder contains categories, sets, and elapsed days.',
    takeaway: 'Typed leaves let different kinds of data share a model, while branches preserve the records those values describe.',
  },
  claims: {
    title: 'A policy with several levels of nested records',
    description: 'ClaimsModel contains claims, each claim contains invoices, and each invoice contains line items. Prediction targets belong to the individual claims.',
    schemaNote: 'The nested declarations describe policy → claims → invoices → line items, with typed leaves at each level.',
    takeaway: 'Nested branches keep each amount attached to its item, invoice, and claim as context moves through the model.',
  },
};

export const models: Concept = {
  id: 'models',
  title: 'Model trees',
  description: 'Understand the branches, typed leaves, and nested context of a model.',
  intro: [],
  principles: [
    { title: 'Context follows the hierarchy', text: 'Within a branch, attention combines representations of its fields and child records. Cross-attention can reduce a collection to learned summaries for its parent and supply context to prediction heads. Nesting lets this process carry information across several levels of related records.' },
    { title: 'Typed leaves define how values are represented', text: 'Each leaf type owns its representation of a value. A number, a category, and a set can therefore enter the same tree through components suited to their meanings.' },
    { title: 'Targets keep their place in the tree', text: 'A prediction target can belong to the root or to a nested record. Its location determines where its answer belongs: one probability for a ticket, one forecast for each supplied week, or several answers for each claim.' },
  ],
  examples: scenarios.map((scenario) => {
    const lesson = lessons[scenario.id];
    return {
      caseId: scenario.id,
      title: lesson.title,
      description: lesson.description,
      blocks: [
        { kind: 'model' as const, label: 'Explore the model tree', caseId: scenario.id, note: `${lesson.schemaNote} The walkthrough keeps the task’s primary answer in place while adding context and another objective.` },
      ],
      takeaway: lesson.takeaway,
    };
  }),
  takeaway: 'Branches describe which records belong together. Typed leaves describe their values. Together, they define the structure through which the model learns and predicts.',
  links: [
    { label: 'Describe repeated records with branches', href: '/concepts/branches/' },
    { label: 'Choose field representations', href: '/concepts/data-types/' },
    { label: 'Find model constructors and methods', href: '/api-map/' },
  ],
};
