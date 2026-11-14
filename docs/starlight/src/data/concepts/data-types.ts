import { stringify } from 'yaml';
import type { Concept, CaseId } from './types';
import { scenarios } from './scenarios';

const lessons: Record<CaseId, { title: string; description: string; declarations: string; fields: string[]; note: string; extra?: { code: string; note: string }; takeaway: string }> = {
  tickets: {
    title: 'A label, a message, and elapsed time mean different things',
    description: 'An author label can identify a category, a message body carries language meaning, and elapsed minutes have numeric magnitude. Their source values need different representations.',
    declarations: `import relflow as rf


class TicketMessage(rf.Branch):
    """One message in a support ticket conversation."""

    author = rf.Category
    body = rf.Text
    minutes_since_previous = rf.Number`,
    fields: ['priority', 'messages'],
    note: 'Category learns label-specific context; Text uses a frozen text encoder; Number keeps elapsed minutes as a numeric value. Text requires its optional dependencies and an available encoder.',
    extra: {
      code: 'priority = rf.Enum("low", "normal", "high")\nescalated = rf.Boolean(mask=True)\nhours_to_resolve = rf.Number(mask=True)',
      note: 'These fields belong inside TicketModel. Enum fixes the accepted priority labels; the masked Boolean and Number fields define two different kinds of answer.',
    },
    takeaway: 'The type supplies meaning and a representation. Masking supplies the prediction role: a probability for escalation and a numeric value for resolution time.',
  },
  churn: {
    title: 'Calendar position and elapsed tenure are separate values',
    description: 'Usage records contain recurring calendar context and measured quantities. Subscriber tenure describes elapsed months rather than a position in the calendar.',
    declarations: `import relflow as rf


class MonthlyUsage(rf.Branch):
    """One month of a subscriber's product usage."""

    month = rf.Period("month_of_year")
    minutes = rf.Number
    data_gb = rf.Number
    overage_fee = rf.Number`,
    fields: ['tenure_months', 'monthly_usage'],
    note: 'Period extracts the requested recurring calendar part from a timestamp. The numeric fields retain their own units: minutes, gigabytes, and currency.',
    extra: {
      code: 'tenure_months = rf.Number\nchurn_90d = rf.Boolean(mask=True)',
      note: 'These fields belong inside ChurnModel. Tenure supplies elapsed magnitude; the hidden cancellation flag supplies the future outcome to learn.',
    },
    takeaway: 'A timestamp and a duration are different modeling choices. Month-of-year context does not provide elapsed tenure or all the information in the original timestamp.',
  },
  demand: {
    title: 'The same value type can be context or an answer',
    description: 'Past units sold are observed numeric context. Future units sold are numeric targets. The representation remains Number while masking changes the field’s role.',
    declarations: `import relflow as rf


class SalesWeek(rf.Branch):
    """One week of observed sales for a store and product."""

    week = rf.Period("week_of_year")
    units_sold = rf.Number
    promotion = rf.Boolean
    stockout = rf.Boolean


class ForecastWeek(rf.Branch):
    """One supplied future week to forecast."""

    week = rf.Period("week_of_year")
    promotion = rf.Boolean
    units_sold = rf.Number(mask=True)`,
    fields: ['history', 'horizon'],
    note: 'The historical branch reads known units. The horizon branch learns hidden units for supplied future weeks. Planned promotions can remain known Boolean inputs.',
    takeaway: 'Type and role are separate decisions. Number represents the quantity; the target mask says which values must be learned rather than read as context.',
  },
  credit: {
    title: 'Keep category identities separate from financial magnitudes',
    description: 'An account type names a class of account. Balance, months open, and missed-payment counts have numeric magnitude that should survive their encoding.',
    declarations: `import relflow as rf


class CreditAccount(rf.Branch):
    """One of the applicant's existing credit accounts."""

    account_type = rf.Category
    balance = rf.Number
    months_open = rf.Number
    missed_payments = rf.Number`,
    fields: ['requested_amount', 'accounts'],
    note: 'Number manages numeric normalization from training data. Keep units consistent in each field; a recorded zero missed payments remains a known value.',
    extra: {
      code: 'product = rf.Category\nrequested_amount = rf.Number\ndefault_12m = rf.Boolean(mask=True)',
      note: 'These fields belong inside CreditModel. Product is an input label, the amount has magnitude, and the default target asks for a true-or-false probability.',
    },
    takeaway: 'Choose a field’s representation from what its value means. Converting every source column to a label would discard the numeric distances this problem supplies.',
  },
  maintenance: {
    title: 'A label set and a history are different kinds of collection',
    description: 'A work order’s replaced parts are unordered memberships. The work orders themselves are records with their own type and elapsed age, while sensor readings carry measurements.',
    declarations: `import relflow as rf


class SensorReading(rf.Branch):
    """One sensor observation from an asset."""

    hour = rf.Period("hour_of_day")
    vibration = rf.Number
    temperature = rf.Number
    load = rf.Number


class WorkOrder(rf.Branch):
    """One completed maintenance job on an asset."""

    type = rf.Category
    parts_replaced = rf.Set
    age_days = rf.Number`,
    fields: ['sensor_readings', 'work_orders'],
    note: 'Set represents which part labels are present, without order or duplicate counts. Branch preserves each job’s related fields. The empty parts_replaced list below is known empty membership.',
    takeaway: 'The fact that two source values are lists does not give them the same meaning. Use a Set for membership and a Branch for repeated structured records.',
  },
  claims: {
    title: 'Identity, label, and amount carry distinct signals',
    description: 'A provider identifier supplies matching relationships, a billed-item code names a category, and its amount supplies numeric magnitude within the invoice.',
    declarations: `import relflow as rf


class LineItem(rf.Branch):
    """One billed item within a claim invoice."""

    code = rf.Category
    amount = rf.Number


class Invoice(rf.Branch):
    """One provider's invoice and its billed items."""

    provider = rf.Hash
    line_items = LineItem(length=32, overflow="error")`,
    fields: ['claims'],
    note: 'Hash provides identifier relationships without a persistent identity vocabulary. Category can learn label-specific representations; Number preserves the billed amount’s magnitude.',
    extra: {
      code: 'paid_amount = rf.Number(mask=True)\nis_fraudulent = rf.Boolean(mask=True)',
      note: 'These targets belong inside Claim. Payout is a numeric answer; fraud is a Boolean probability. Neither type makes that answer available as model input.',
    },
    takeaway: 'The schema can combine these representations at every nesting level. Choose them for their intended semantics, then evaluate whether they support the decisions your application needs.',
  },
};

export const dataTypes: Concept = {
  id: 'data-types',
  title: 'Data types',
  description: 'Choose representations according to what your values mean, then assign their input or target roles.',
  intro: [
    'A field type tells relflow what a value means and how to encode or learn it. The integer 17 could be a measured count, a category code, or an opaque identifier. Its storage type alone does not settle that choice.',
    'Types provide their own accepted values, learned state, and output contracts. You describe the meaning; relflow handles the associated representation machinery, such as numeric normalization or categorical vocabularies.',
  ],
  principles: [
    { title: 'Start with the meaning of the value', text: 'Use Number when numeric distance matters, Category for a learned label vocabulary, Enum for a fixed declaration, and Hash for identifier relationships. Different choices can be valid for the same source representation when the modeling intent differs.' },
    { title: 'A type owns a learning contract', text: 'A field type controls more than conversion from source data: it defines its representation, supported objectives, and written results. Follow its accepted values and missingness semantics rather than forcing every field into one encoding.' },
    { title: 'Structure describes where values belong', text: 'Branch keeps repeated records and their child fields aligned. Set represents unordered label membership, while Vector represents a fixed number of numeric coordinates. Those collection meanings can coexist in a nested schema.' },
    { title: 'Type and prediction role are separate', text: 'A Boolean can be a known fact or a hidden outcome; Number can be a measurement or a target. mask=True selects a supervised target, whose type determines the answer contract. Text, Hash, and Period context does not imply generation of original strings, identifiers, or timestamps.' },
  ],
  examples: scenarios.map((scenario) => {
    const lesson = lessons[scenario.id];
    const input = Object.fromEntries(lesson.fields.map((field) => [field, scenario.inputs[field]]));
    return {
      caseId: scenario.id,
      title: lesson.title,
      description: lesson.description,
      blocks: [
        { label: 'Choose meanings inside a record', lang: 'python' as const, code: lesson.declarations, note: lesson.note },
        ...(lesson.extra ? [{ label: 'Other fields in the same model', lang: 'python' as const, code: lesson.extra.code, note: lesson.extra.note }] : []),
        { label: 'Matching known input values', lang: 'yaml' as const, records: true, code: stringify(input, { indent: 2, lineWidth: 0 }), note: 'These values are selected from the shared illustrative input record. The full schema supplies the enclosing branches and any fields omitted from this excerpt.' },
      ],
      takeaway: lesson.takeaway,
    };
  }),
  takeaway: 'Choose the representation from the value’s meaning, keep its structural owner, and declare separately whether the value is known context or an answer to learn.',
  links: [
    { label: 'Register a custom data type', href: '/advanced/custom-data-types/' },
    { label: 'Assign prediction roles with targets', href: '/concepts/targets/' },
    { label: 'Choose among the field types', href: '/data-types/' },
    { label: 'See collection membership with Set', href: '/data-types/set/' },
    { label: 'Inspect the learning experiment library', href: '/evidence/' },
  ],
};
