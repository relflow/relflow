/** Experiment-owned presentation. This module validates structure, never experiment membership. */
export type ComparisonKind =
  | 'Controlled comparison'
  | 'Baseline reference'
  | 'Before and after'
  | 'Recorded conditions'
  | 'Recorded measurement';
export interface Review {
  question: string;
  description?: string;
  modeling: string;
  note: string;
  comparison: {
    kind: ComparisonKind;
    metric: string;
    definition: string;
    direction: string;
    domain?: [number, number];
    conditions: { label: string; path: (string | number)[]; format: string }[];
  };
  criteria?: {
    prefixes?: Record<string, string>;
    operations?: 'group' | 'request';
  };
}

function mapping(value: unknown, location: string): Record<string, unknown> {
  if (!value || typeof value !== 'object' || Array.isArray(value))
    throw new Error(`${location}: expected a mapping.`);
  return value as Record<string, unknown>;
}

/** Every active experiment must declare its complete presentation contract. */
export function readReview(value: unknown, location: string): Review {
  const entry = mapping(value, `${location}: evidence-review`);
  for (const field of ['question', 'modeling', 'note']) {
    if (typeof entry[field] !== 'string' || (field !== 'note' && !entry[field].trim()))
      throw new Error(`${location}: evidence-review.${field} must be a string`);
  }
  if (
    entry.description !== undefined &&
    (typeof entry.description !== 'string' || !entry.description.trim())
  )
    throw new Error(`${location}: evidence-review.description must be a nonempty string`);
  const plan = mapping(entry.comparison, `${location}: evidence-review.comparison`);
  const kinds: ComparisonKind[] = [
    'Controlled comparison',
    'Baseline reference',
    'Before and after',
    'Recorded conditions',
    'Recorded measurement',
  ];
  if (!kinds.includes(plan.kind as ComparisonKind))
    throw new Error(`${location}: evidence-review needs a supported comparison kind`);
  for (const field of ['metric', 'definition', 'direction'])
    if (typeof plan[field] !== 'string' || !(plan[field] as string).trim())
      throw new Error(`${location}: comparison.${field} must be a nonempty string`);
  if (
    plan.domain !== undefined &&
    (!Array.isArray(plan.domain) ||
      plan.domain.length !== 2 ||
      !plan.domain.every((value) => typeof value === 'number' && Number.isFinite(value)) ||
      plan.domain[0] >= plan.domain[1])
  )
    throw new Error(`${location}: comparison.domain must contain two finite increasing bounds`);
  if (!Array.isArray(plan.conditions) || !plan.conditions.length)
    throw new Error(`${location}: evidence-review.comparison.conditions must be nonempty`);
  for (const condition of plan.conditions) {
    const point = mapping(condition, `${location}: comparison condition`);
    if (
      typeof point.label !== 'string' ||
      !point.label.trim() ||
      typeof point.format !== 'string' ||
      !['number', 'percent', 'error', 'auc'].includes(point.format)
    )
      throw new Error(`${location}: comparison conditions need a label and format`);
    if (
      !Array.isArray(point.path) ||
      !point.path.length ||
      !point.path.every(
        (part) =>
          (typeof part === 'string' && part.length > 0) ||
          (typeof part === 'number' && Number.isSafeInteger(part)),
      )
    )
      throw new Error(`${location}: comparison paths need nonempty keys or integer indices`);
  }
  if (entry.criteria !== undefined) {
    const criteria = mapping(entry.criteria, `${location}: evidence-review.criteria`);
    if (
      criteria.prefixes !== undefined &&
      Object.values(mapping(criteria.prefixes, `${location}: criterion prefixes`)).some(
        (label) => typeof label !== 'string' || !label.trim(),
      )
    )
      throw new Error(`${location}: criterion prefixes must map to nonempty labels`);
    if (
      criteria.operations !== undefined &&
      !['group', 'request'].includes(criteria.operations as string)
    )
      throw new Error(`${location}: criterion operations must be group or request`);
  }
  return entry as unknown as Review;
}
