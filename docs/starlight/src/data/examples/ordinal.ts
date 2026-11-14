import { prediction, type TypeExample } from '../type-example.ts';

/** Static contract illustrations, never executed predictions or measurements. */
export const ordinalExamples: TypeExample[] = [
  {
    id: 'priority',
    label: 'Ordered labels',
    summary: 'Predict a review priority',
    title: 'What priority should this review receive?',
    description: 'Declare labels in their intended order and learn a numeric rank for the answer.',
    schema: `import relflow as rf


class Review(rf.Model):
    """A submitted review assigned an ordered priority."""

    amount_usd = rf.Number
    hours_waiting = rf.Number
    customer_tier = rf.Category
    priority = rf.Ordinal(
        values=("low", "normal", "high"),
        mask=True,
    )`,
    dataset: {
      id: 'priority', label: 'Review priority',
      observation: 'One submitted review awaiting a priority assignment.',
      description: 'Known review context paired with an answer from an explicitly ordered set of labels.',
      entities: [{
        id: 'review', name: 'Review', description: 'The submitted review receiving one priority label.',
        columns: [
          { name: 'amount_usd', type: 'number', description: 'Amount associated with the review, in US dollars.' },
          { name: 'hours_waiting', type: 'number', description: 'Hours the review has been waiting at assessment time.' },
          { name: 'customer_tier', type: 'string', description: 'Known customer tier, such as enterprise.' },
          { name: 'priority', type: 'string', role: 'target', description: 'Historical priority label: low, normal, or high, in that declared order. New requests omit the answer.' },
        ],
      }],
      relations: [],
    },
    inputs: { amount_usd: 750, hours_waiting: 18, customer_tier: 'enterprise' },
    answers: { priority: 'high' },
    output: { predictions: { priority: prediction('high') } },
    outputNote: 'Content is the decoded label itself. The declaration maps low, normal, and high to numeric ranks 0, 1, and 2; no class probabilities are returned.',
    takeaway: 'An error from low to high is two rank steps. The declared order supplies that distance instead of treating every incorrect label as an unrelated class.',
  },
  {
    id: 'integer-ranks',
    label: 'Integer ranks',
    summary: 'Predict an open integer position',
    title: 'How should this task move in the schedule?',
    description: 'Use integer positions directly when the rank space is open and positions can be negative.',
    schema: `import relflow as rf


class ScheduledTask(rf.Model):
    """A task positioned relative to today's scheduling slot."""

    current_offset = rf.Ordinal
    remaining_hours = rf.Number
    available_hours = rf.Number
    next_offset = rf.Ordinal(mask=True)`,
    dataset: {
      id: 'integer-ranks', label: 'Scheduled task',
      observation: 'One task being positioned in a schedule.',
      description: 'Current scheduling context paired with the task’s next integer offset.',
      entities: [{
        id: 'task', name: 'Scheduled task', description: 'A task positioned relative to today’s scheduling slot.',
        columns: [
          { name: 'current_offset', type: 'number', description: 'Current integer scheduling offset. Negative positions are valid; the source supplies a rank directly.' },
          { name: 'remaining_hours', type: 'number', description: 'Estimated work remaining on the task, in hours.' },
          { name: 'available_hours', type: 'number', description: 'Available scheduling capacity in hours.' },
          { name: 'next_offset', type: 'number', role: 'target', description: 'Historical next integer scheduling offset. Ordinal predicts an integer rank rather than choosing from a fixed vocabulary.' },
        ],
      }],
      relations: [],
    },
    inputs: { current_offset: -2, remaining_hours: 6, available_hours: 8 },
    answers: { next_offset: -1 },
    output: { predictions: { next_offset: prediction(-1) } },
    outputNote: 'Without values, content is a native integer rank. The numeric readout is rounded to the nearest integer before being written.',
    takeaway: 'Negative ranks and ranks absent from training are valid without vocabulary admission. Numeric gaps keep their size; accepting an unseen rank does not establish prediction quality there.',
  },
  {
    id: 'inspection',
    label: 'Every component',
    summary: 'Combine ordered and numeric targets',
    title: 'What grade and repair time does each component need?',
    description: 'Predict an ordered grade and a measured repair duration for each supplied component.',
    schema: `import relflow as rf


class Component(rf.Branch):
    """An inspected component with grade and repair targets."""

    family = rf.Category
    wear_percent = rf.Number
    grade = rf.Ordinal(values=(10, 20, 30), mask=True)
    repair_hours = rf.Number(mask=True)


class Inspection(rf.Model):
    """Components assessed together during a maintenance inspection."""

    equipment_age_years = rf.Number
    components = Component(length=16, overflow="error")`,
    dataset: {
      id: 'inspection', label: 'Component inspection',
      observation: 'One maintenance inspection containing its supplied components.',
      description: 'Shared equipment context and per-component measurements, with separate grade and repair-time answers.',
      entities: [
        {
          id: 'inspection', name: 'Inspection', description: 'Components assessed together during one maintenance inspection.',
          columns: [
            { name: 'equipment_age_years', type: 'number', description: 'Equipment age in years when the inspection occurs.' },
            { name: 'components', type: 'records[]', role: 'collection', description: 'Inspected component records, each retaining its own grade and repair-time targets.' },
          ],
        },
        {
          id: 'component', name: 'Component', description: 'One supplied component in components.',
          columns: [
            { name: 'family', type: 'string', description: 'Component family, such as bearing or seal.' },
            { name: 'wear_percent', type: 'number', description: 'Observed component wear expressed as a percentage.' },
            { name: 'grade', type: 'number', role: 'target', description: 'Historical grade label from the ordered values 10, 20, and 30. These are adjacent ordinal labels, not ten-step rank gaps.' },
            { name: 'repair_hours', type: 'number', role: 'target', description: 'Observed repair duration in hours for this component, learned separately from the grade.' },
          ],
        },
      ],
      relations: [{ from: 'inspection', to: 'component', fromColumn: 'components', cardinality: 'many', description: 'Each inspection contains the supplied components collection.' }],
    },
    inputs: {
      equipment_age_years: 7,
      components: [
        { family: 'bearing', wear_percent: 38 },
        { family: 'seal', wear_percent: 82 },
      ],
    },
    answers: {
      components: [
        { grade: 20, repair_hours: 1.5 },
        { grade: 10, repair_hours: 3 },
      ],
    },
    output: {
      predictions: {
        components: [
          { grade: prediction(20), repair_hours: prediction(1.7) },
          { grade: prediction(10), repair_hours: prediction(2.8) },
        ],
      },
    },
    outputNote: 'Each result stays aligned with its supplied component. Grade returns a native integer label, while repair_hours returns a continuous measurement.',
    takeaway: 'The declared labels 10, 20, and 30 occupy adjacent ranks 0, 1, and 2. Their label gaps do not become ten-step distances, and their integer types survive prediction writing.',
  },
];
