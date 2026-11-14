import type { Concept, CaseId } from './types';
import { scenarios } from './scenarios';

const lessons: Record<CaseId, { title: string; description: string; schemaNote: string; trainingNote: string; takeaway: string }> = {
  tickets: {
    title: 'One conversation, two questions',
    description: 'A ticket brings together product context and repeated messages. Its two targets ask whether it escalates and how many hours it takes to resolve.',
    schemaNote: 'TicketMessage declares the records in the conversation. TicketModel combines that context with a Boolean target and a numeric target.',
    trainingNote: 'The data module supplies ticket records and their completed outcomes. Fitting learns both objectives from that shared context.',
    takeaway: 'A model can learn several kinds of answer from the same observation. Each target keeps its own field type and meaning.',
  },
  churn: {
    title: 'Two histories, one subscriber decision',
    description: 'Usage and support tickets describe different parts of a subscriber’s history. The model combines them with the current plan and tenure to predict cancellation.',
    schemaNote: 'MonthlyUsage and SupportTicket are separate reusable record declarations. Their placements give each history its own retained-length policy.',
    trainingNote: 'Training rows need complete 90-day cancellation outcomes and context available at the chosen subscriber cutoff.',
    takeaway: 'The observation is a subscriber at a decision time. Its histories become parts of the model contract without first becoming one aggregate feature row.',
  },
  demand: {
    title: 'Put the prediction on each future week',
    description: 'Sales history supplies observed context, while the horizon supplies the future weeks and planned promotions that need forecasts.',
    schemaNote: 'The prediction target belongs to ForecastWeek. DemandModel has no root-level units target: it learns one value for each supplied horizon record.',
    trainingNote: 'Historical training cutoffs include the later units for their supplied forecast weeks as hidden targets. Prediction requests omit those units.',
    takeaway: 'Where a target appears is part of the learning problem. This schema predicts values on supplied weeks; it does not invent the forecast calendar.',
  },
  credit: {
    title: 'Keep application and account context together',
    description: 'A loan application contains its requested amount, existing credit accounts, and recent inquiries. The root target describes the applicant’s later outcome.',
    schemaNote: 'CreditAccount and CreditInquiry preserve different record types. CreditModel specifies how those collections belong to one application.',
    trainingNote: 'The data module supplies applications with decision-time account snapshots and completed 12-month labels. The application owns those cutoffs and splits.',
    takeaway: 'A schema describes the assembled observation. It does not discover which accounts belong to an applicant or which facts existed when the decision was made.',
  },
  maintenance: {
    title: 'Combine readings with completed maintenance',
    description: 'An asset’s sensor readings and work orders carry different information. The model combines them with asset context to learn a 30-day failure outcome.',
    schemaNote: 'SensorReading mixes calendar and numeric values; WorkOrder mixes categories, label sets, and elapsed days. Both are context for the same root target.',
    trainingNote: 'Fit uses completed failure labels while keeping later readings and maintenance actions outside each observation’s known context.',
    takeaway: 'Heterogeneous histories can belong to one predictive model. The contract keeps their meanings and record boundaries visible.',
  },
  claims: {
    title: 'A policy context with answers for every claim',
    description: 'Claims contain invoices, and invoices contain billed items. Each claim needs a payout estimate and a fraud probability within the enclosing policy context.',
    schemaNote: 'The nested declarations describe policy → claims → invoices → line items. Both targets live inside Claim, so their answers belong to each supplied claim.',
    trainingNote: 'Fitting needs the actual settled answers for each training claim. Neither target becomes an input clue for the other because both are masked.',
    takeaway: 'The same model declaration can describe deep input structure and mixed targets below the root. Those supplied records determine where the answers belong.',
  },
};

export const models: Concept = {
  id: 'models',
  title: 'Models',
  description: 'Declare the records and predictions that define a trainable neural model.',
  intro: [
    'A relflow model starts with a schema: typed values, related records, and the answers you want to learn. relflow uses that contract to construct field representations, branch context encoders, decoders, and learning objectives.',
    'Keep the class declarations at module scope. Construct a model when you want its neural parameters, then fit it on examples. These are separate steps, even when the complete workflow takes only a few calls.',
  ],
  principles: [
    { title: 'The schema shapes the architecture', text: 'Field types choose representations and output contracts. Branches describe repeated context, and target locations determine where predictions are decoded. A model class therefore describes a learning structure as well as an input contract.' },
    { title: 'Declaration, construction, and fitting are separate', text: 'Declaring a Model subclass records the schema. Calling that class constructs a new, untrained model. Calling fit learns from the supplied data; choosing an architecture preset does not load pretrained task knowledge.' },
    { title: 'Configure a node where it is placed', text: 'Class attributes declare children. Calling a declared branch configures that placement, while constructor settings such as preset configure the enclosing model. Add or replace declared children through a subclass rather than constructor field arguments.' },
    { title: 'Define what one example means', text: 'The root represents one assembled observation, which may contain many nested records. Your application supplies its relationships, decision cutoff, and labels; the model contract describes the observation you chose.' },
  ],
  examples: scenarios.map((scenario) => {
    const lesson = lessons[scenario.id];
    return {
      caseId: scenario.id,
      title: lesson.title,
      description: lesson.description,
      blocks: [
        { kind: 'model' as const, label: 'See what the schema builds', caseId: scenario.id, note: `${lesson.schemaNote} The walkthrough keeps the task’s primary answer in place while adding context and another objective.` },
        { label: 'Fit the constructed model', lang: 'python' as const, code: 'model.fit(datamodule, batch_size=128, max_epochs=8)', note: `${lesson.trainingNote} Construct the class shown in the selected step and supply matching training and validation data, including any additional target labels. These settings illustrate the API rather than an optimized recipe.` },
      ],
      takeaway: lesson.takeaway,
    };
  }),
  takeaway: 'Describe the learning contract, construct its neural model, and fit it on your observations. Keep those three steps distinct as the schema grows.',
  links: [
    { label: 'Describe repeated records with branches', href: '/concepts/branches/' },
    { label: 'Choose field representations', href: '/concepts/data-types/' },
    { label: 'Train a complete first model', href: '/quickstart/' },
    { label: 'Find model constructors and methods', href: '/api-map/' },
  ],
};
