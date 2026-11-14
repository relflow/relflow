import { stringify } from 'yaml';
import type { Concept, CaseId } from './types';
import { scenarios } from './scenarios';

const lessons: Record<CaseId, { title: string; description: string; fields: string[]; note: string; takeaway: string }> = {
  tickets: {
    title: 'A label, a message, and elapsed time mean different things',
    description: 'An author label can identify a category, a message body carries language meaning, and elapsed minutes have numeric magnitude. Their source values need different representations.',
    fields: ['priority', 'messages'],
    note: 'Category learns label-specific context; Text uses a frozen text encoder; Number keeps elapsed minutes as a numeric value. Text requires its optional dependencies and an available encoder.',
    takeaway: 'The type supplies meaning and a representation. Masking supplies the prediction role: a probability for escalation and a numeric value for resolution time.',
  },
  churn: {
    title: 'Calendar position and elapsed tenure are separate values',
    description: 'Usage records contain recurring calendar context and measured quantities. Subscriber tenure describes elapsed months rather than a position in the calendar.',
    fields: ['tenure_months', 'monthly_usage'],
    note: 'Period extracts the requested recurring calendar part from a timestamp. The numeric fields retain their own units: minutes, gigabytes, and currency.',
    takeaway: 'A timestamp and a duration are different modeling choices. Month-of-year context does not provide elapsed tenure or all the information in the original timestamp.',
  },
  demand: {
    title: 'The same value type can be context or an answer',
    description: 'Past units sold are observed numeric context. Future units sold are numeric targets. The representation remains Number while masking changes the field’s role.',
    fields: ['history', 'horizon'],
    note: 'The historical branch reads known units. The horizon branch learns hidden units for supplied future weeks. Planned promotions can remain known Boolean inputs.',
    takeaway: 'Type and role are separate decisions. Number represents the quantity; the target mask says which values must be learned rather than read as context.',
  },
  credit: {
    title: 'Keep category identities separate from financial magnitudes',
    description: 'An account type names a class of account. Balance, months open, and missed-payment counts have numeric magnitude that should survive their encoding.',
    fields: ['requested_amount', 'accounts'],
    note: 'Number manages numeric normalization from training data. Keep units consistent in each field; a recorded zero missed payments remains a known value.',
    takeaway: 'Choose a field’s representation from what its value means. Converting every source column to a label would discard the numeric distances this problem supplies.',
  },
  maintenance: {
    title: 'A label set and a history are different kinds of collection',
    description: 'A work order’s replaced parts are unordered memberships. The work orders themselves are records with their own type and elapsed age, while sensor readings carry measurements.',
    fields: ['sensor_readings', 'work_orders'],
    note: 'Set represents which part labels are present, without order or duplicate counts. Branch preserves each job’s related fields. The empty parts_replaced list below is known empty membership.',
    takeaway: 'The fact that two source values are lists does not give them the same meaning. Use a Set for membership and a Branch for repeated structured records.',
  },
  claims: {
    title: 'Identity, label, and amount carry distinct signals',
    description: 'A provider identifier supplies matching relationships, a billed-item code names a category, and its amount supplies numeric magnitude within the invoice.',
    fields: ['claims'],
    note: 'Hash provides identifier relationships without a persistent identity vocabulary. Category can learn label-specific representations; Number preserves the billed amount’s magnitude.',
    takeaway: 'The schema can combine these representations at every nesting level. Choose them for their intended semantics, then evaluate whether they support the decisions your application needs.',
  },
};

export const dataTypes: Concept = {
  id: 'data-types',
  title: 'Leaf types',
  description: 'See how typed leaves bind to values and define their representation and prediction roles.',
  intro: [],
  principles: [
    { title: 'Start with the meaning of the value', text: 'Use Number when numeric distance matters, Category for a learned label vocabulary, Enum for a fixed declaration, and Hash for identifier relationships. Different choices can be valid for the same source representation when the modeling intent differs.' },
    { title: 'Structure describes where values belong', text: 'Branch keeps repeated records and their child fields aligned. Set represents unordered label membership, while Vector represents a fixed number of numeric coordinates. Those collection meanings can coexist in a nested schema.' },
    { title: 'The same type can be input or target', text: 'A Boolean can represent a known fact or a hidden outcome. Number can represent an observed measurement or an answer to learn. The type defines the representation; the field configuration selects its role.' },
  ],
  examples: scenarios.map((scenario) => {
    const lesson = lessons[scenario.id];
    const input = Object.fromEntries(lesson.fields.map((field) => [field, scenario.inputs[field]]));
    return {
      caseId: scenario.id,
      title: lesson.title,
      description: lesson.description,
      blocks: [
        { label: 'Typed leaves in the class schema', lang: 'python' as const, code: scenario.schema, note: lesson.note },
        { label: 'Matching known input values', lang: 'yaml' as const, records: true, code: stringify(input, { indent: 2, lineWidth: 0 }), note: 'These values are selected from the shared illustrative input record. Their names match the corresponding fields inside each enclosing record.' },
      ],
      takeaway: lesson.takeaway,
    };
  }),
  takeaway: 'Choose the representation from the value’s meaning, keep its structural owner, and declare separately whether the value is known context or an answer to learn.',
  links: [
    { label: 'Register a custom data type', href: '/advanced/custom-data-types/' },
    { label: 'Choose among the field types', href: '/data-types/' },
    { label: 'Find field options and extension contracts', href: '/api-map/' },
  ],
};
