import type { EvidenceEntry } from './data';
import { metricAt } from './metrics.ts';
import { review } from './catalog.ts';
import type { ComparisonKind, Metric } from './reviews.ts';

export interface Comparison {
  kind: ComparisonKind;
  metric: Metric;
  definition: string;
  direction: string;
  points: { label: string; path: (string | number)[]; format: string; value: number | null }[];
  domain: [number, number];
  available: boolean;
}

const definitions: Record<Metric, string> = {
  RMSE: 'Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.',
  MAE: 'Mean absolute prediction error, in the target’s original units. Zero is an exact prediction.',
  MSE: 'Mean squared prediction error, in squared target units. Zero is an exact prediction.',
  'Normalized RMSE':
    'Root mean squared error divided by the reference scale defined in this experiment. Zero is exact; one equals that reference scale.',
  'Normalized MAE':
    'Mean absolute error divided by the reference scale defined in this experiment. Zero is exact; one equals that reference scale.',
  AUC: 'Ranking score: 0.5 is chance and 1 is perfect separation. This is not the percentage of correct predictions.',
  Accuracy:
    'Fraction of predictions matching the expected answer, over the population named in each condition.',
  Probability:
    'Predicted probability of the named outcome or membership. The task determines the appropriate value; higher is not always better.',
  Proportion:
    'Recorded fraction of the population named in each condition. See the method for its denominator.',
  'Prediction drift':
    'Absolute change in predictions after an operation, on the scale named in each condition. Zero means no change.',
  'Assigned groups':
    'Number of assigned groups at the recorded epoch. Group count alone does not establish correct memberships.',
  'Predicted value':
    'Model output for each input condition, in target units. These are predictions, not prediction errors.',
  'Adjusted Rand index':
    'Agreement with the true groups, ignoring group names. One is exact agreement; independent groupings average around zero and can score below zero.',
  'Macro F1':
    'Average of each class’s F1 score, giving every class equal weight. One is perfect; this is not overall accuracy.',
  'False positive rate':
    'Fraction of legitimate records incorrectly flagged as anomalies in the named group. Lower means fewer false alarms.',
  'Probability RMSE':
    'Root mean squared difference from the true probability allowed by the visible information. This scores probability estimates, not individual yes/no answers.',
};

/** Resolve the reviewed comparison directly from the selected record, without outcome inference. */
export function comparison(entry: EvidenceEntry): Comparison {
  const plan = review(entry).comparison;
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
  const unitInterval = [
    'AUC',
    'Accuracy',
    'Probability',
    'Proportion',
    'Adjusted Rand index',
    'Macro F1',
    'False positive rate',
  ].includes(plan.metric);
  const low = Math.min(0, ...values);
  const high = Math.max(unitInterval ? 1 : 0, ...values);
  const lower = /RMSE|MAE|MSE/.test(plan.metric) || plan.metric === 'False positive rate';
  const higher = ['AUC', 'Accuracy', 'Adjusted Rand index', 'Macro F1'].includes(plan.metric);
  return {
    ...plan,
    points,
    definition: definitions[plan.metric],
    direction: lower
      ? 'Lower error is better'
      : higher
        ? 'Higher is better'
        : 'Task-specific interpretation',
    domain: [low, high === low ? low + 1 : high],
    available: usable && points.length > 0 && points.every((point) => point.value !== null),
  };
}
