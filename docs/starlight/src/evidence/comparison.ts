import type { EvidenceEntry } from './data';
import { metricAt } from './metrics.ts';
import { review } from './catalog.ts';
import type { ComparisonKind } from './metadata.ts';

export interface Comparison {
  kind: ComparisonKind;
  metric: string;
  definition: string;
  direction: string;
  points: { label: string; path: (string | number)[]; format: string; value: number | null }[];
  domain: [number, number];
  available: boolean;
}

/** Resolve the reviewed comparison directly from the selected record, without outcome inference. */
export function comparison(entry: EvidenceEntry): Comparison {
  const content = review(entry);
  const plan = content.comparison;
  const usable =
    !!entry.result &&
    entry.result.mode === 'full' &&
    entry.result.outcome !== 'error' &&
    entry.sourceMatches;
  const points = plan.conditions.map((condition) => ({
    ...condition,
    value: usable ? metricAt(entry.result?.metrics, condition.path) : null,
  }));
  const values = points.flatMap((point) => (point.value === null ? [] : [point.value]));
  const low = Math.min(0, plan.domain?.[0] ?? 0, ...values);
  const high = Math.max(plan.domain?.[1] ?? 0, ...values);
  return {
    ...plan,
    points,
    domain: [low, high === low ? low + 1 : high],
    available: usable && points.length > 0 && points.every((point) => point.value !== null),
  };
}
