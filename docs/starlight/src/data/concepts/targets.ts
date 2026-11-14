import { stringify } from 'yaml';
import { scenarios } from './scenarios';
import type { CaseId, Concept } from './types';

const tasks: Record<CaseId, { title: string; description: string; answersNote: string; takeaway: string }> = {
  tickets: {
    title: 'Learn escalation and resolution time together',
    description: 'One ticket has two answers: whether it escalates and how many hours it takes to resolve. The conversation available at triage supplies context for both.',
    answersNote: 'Both root fields are hidden targets. The training record contains their settled values; the prediction request contains neither.',
    takeaway: 'Boolean learns escalation and Number learns hours. Keep messages written after triage out of the input, even when preparing a historical ticket whose eventual outcome is known.',
  },
  churn: {
    title: 'Predict one subscriber outcome from two histories',
    description: 'Usage and support records are visible context. The single answer is whether this subscriber cancels within 90 days of the observation cutoff.',
    answersNote: 'churn_90d belongs at the subscriber level. It is not repeated for every usage month or support ticket.',
    takeaway: 'Declaring a visible field does not create a reconstruction objective. monthly_usage.minutes and support_tickets.resolved remain context; churn_90d is the supervised answer.',
  },
  demand: {
    title: 'Separate observed sales from future sales targets',
    description: 'Past weeks contain known sales. Future weeks contain known dates and planned promotions, plus hidden units_sold answers during training.',
    answersNote: 'The first answer belongs to the first horizon record and the second to the second record. Merge them by their retained positions, without moving labels into history.',
    takeaway: 'history.units_sold is input; horizon.units_sold is a target. The same leaf name has different roles at different addresses. Predictions fill the supplied forecast slots rather than generating new weeks.',
  },
  credit: {
    title: 'Keep the default label separate from credit history',
    description: 'The application and its known accounts and inquiries provide inputs. default_12m describes the defined default outcome in the following 12 months.',
    answersNote: 'This illustration has a settled negative answer. A loan whose outcome window has not finished must not be labeled false merely because no default has been reported yet.',
    takeaway: 'Past missed payments are visible account facts; the application-level future default is hidden. Decide what constitutes default and which applications have complete labels before fitting.',
  },
  maintenance: {
    title: 'Learn failure risk from readings and completed work',
    description: 'Sensor observations and maintenance known at the assessment time supply context for a failure-within-30-days target.',
    answersNote: 'failure_30d is hidden at the asset level. Sensor values and parts_replaced are inputs, not additional prediction targets in this schema.',
    takeaway: 'A future repair report can help establish the label but cannot become evidence available at the earlier assessment. Add sensor-reconstruction objectives only through an explicit masking policy.',
  },
  claims: {
    title: 'Give every submitted claim two typed answers',
    description: 'Each retained claim receives a payout target and a fraud target, using its invoice items and the policy context. Neither answer is placed on the policy root.',
    answersNote: 'The answer list aligns with the supplied claims. This record has one claim, so it has one paid_amount and one is_fraudulent label.',
    takeaway: 'The amount is a numeric objective and fraud is a Boolean objective. Both are hidden for each claim. Invoice amounts stay attached to their items and can support both targets through learned context.',
  },
};

export const targets: Concept = {
  id: 'targets',
  title: 'Targets and masking',
  description: 'Choose the answers to learn, keep labels out of inputs, and place objectives at the right depth.',
  intro: [
    'A target is a field whose answer you want the model to learn. Its type determines the kind of prediction and objective; its address determines which record that answer describes.',
    'Training records combine known context with observed answers. A prediction request supplies the context and omits the answers. The selected examples show the same schema, inputs, and labels together.',
  ],
  principles: [
    {
      title: 'A target has a type and a location',
      text: 'A Boolean target learns a probability of true; a Number target learns a numeric value in the label’s units. A target at the root describes the observation. A target inside a branch describes each retained record at that depth.',
    },
    {
      title: 'mask=True excludes labels and adds supervision',
      text: 'The shorthand means Mask(skip=True, dropout=False, reconstruct=True). The label is never embedded as an input, and its observed training values supply a reconstruction objective. Keep it out of new requests.',
    },
    {
      title: 'Context dropout is a different choice',
      text: 'A float such as mask=0.1 drops context positions during training without creating a reconstruction objective for them. A Mask with reconstruct=True asks the model to learn the selected values. Choose the visibility and learning policy deliberately; hiding a field alone does not make it an answer.',
    },
    {
      title: 'Keep nested answers aligned',
      text: 'A branch target follows the records supplied to that branch. Preserve their positions when combining context and answers. Several target types can share context, while keeping separate meanings, objectives, and evaluation metrics.',
    },
  ],
  examples: scenarios.map((scenario) => ({
    caseId: scenario.id,
    title: tasks[scenario.id].title,
    description: tasks[scenario.id].description,
    blocks: [
      {
        label: 'Declare the context and hidden targets',
        lang: 'python',
        code: scenario.schema,
        note: 'mask=True marks the supervised fields. Constructing the model creates an untrained network; fitting supplies the learning step.',
      },
      {
        label: 'Known inputs for a new prediction',
        lang: 'yaml', records: true,
        code: stringify(scenario.inputs, { indent: 2, lineWidth: 0 }),
      },
      {
        label: 'Answers added to historical training records',
        lang: 'yaml', records: true,
        code: stringify(scenario.answers, { indent: 2, lineWidth: 0 }),
        note: tasks[scenario.id].answersNote,
      },
    ],
    takeaway: tasks[scenario.id].takeaway,
  })),
  takeaway: 'Choose the answer type and address, declare how it is hidden and learned, and evaluate each target against its own task. Shared context does not establish that every objective benefits equally.',
  links: [
    { label: 'Build masking and pretraining tasks', href: '/advanced/masking-and-pretraining/' },
    { label: 'Read structured prediction results', href: '/predictions/' },
    { label: 'Choose the type of each answer', href: '/data-types/' },
    { label: 'Find Mask and model methods in the API map', href: '/api-map/' },
    { label: 'Inspect controlled masking experiments', href: '/evidence/' },
  ],
};
