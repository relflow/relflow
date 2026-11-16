import type { EvidenceEntry } from './data';
import { metricAt } from './metrics.ts';

type Path = (string | number)[];
type Kind =
  | 'Controlled comparison'
  | 'Baseline reference'
  | 'Before and after'
  | 'Recorded conditions'
  | 'Recorded measurement';
interface Measure {
  label: string;
  path: Path;
  format: string;
}
export interface Comparison {
  kind: Kind;
  metric: string;
  definition: string;
  direction: string;
  points: (Measure & { value: number | null })[];
  domain: [number, number];
  available: boolean;
}

// Select comparisons, never values or verdicts. Unselected diagnostics stay in Measurements.
const controls: Record<string, [number, number]> = {
  P006: [1, 0],
  P010: [1, 0],
  P011: [1, 0],
  P012: [1, 0],
  P014: [1, 0],
  P016: [1, 0],
  P017: [1, 0],
  P021: [1, 0],
  P022: [1, 0],
  P023: [1, 0],
  P024: [1, 0],
  P025: [1, 0],
  P026: [1, 0],
  P028: [1, 0],
  P029: [1, 0],
  P030: [1, 0],
  P031: [1, 0],
  P032: [1, 0],
  P033: [1, 0],
  P036: [0, 1],
  P037: [1, 0],
  P038: [1, 0],
  P042: [0, 2],
  P043: [0, 1],
  P044: [1, 0],
  P045: [0, 1],
  P052: [1, 0],
  P064: [1, 0],
  P067: [1, 0],
  P069: [1, 0],
  P071: [0, 1],
  P073: [1, 0],
  P074: [1, 0],
  P076: [1, 0],
  P077: [0, 1],
  P078: [1, 0],
};
const conditions: Record<string, number[]> = {
  P002: [0, 1],
  P035: [0, 1],
  P053: [0, 1],
  P068: [0, 1, 2],
  P072: [0, 1],
  P056: [0, 1],
  P057: [0, 1],
  P058: [0, 1],
  P060: [0, 1, 2],
  P061: [0, 1],
  P062: [0, 2],
  P063: [0, 1],
  P065: [0, 2],
};
const beforeAfter: Record<string, [number, number]> = { P059: [0, 1], P070: [0, 1] };

function metricName(path: Path, format: string) {
  const key = String(path.at(-1)).toLowerCase();
  if (/nrmse|normalized_rmse/.test(key)) return 'Normalized RMSE';
  if (/rmse/.test(key)) return 'RMSE';
  if (/normalized_mae/.test(key)) return 'Normalized MAE';
  if (/mae/.test(key)) return 'MAE';
  if (/mse/.test(key)) return 'MSE';
  if (/auc/.test(key) || format === 'auc') return 'AUC';
  if (/accuracy/.test(key)) return 'Accuracy';
  if (/probability|probabilities|prior|confidence/.test(path.join(' '))) return 'Probability';
  if (/drift/.test(key)) return 'Prediction drift';
  if (/committed/.test(path.join(' '))) return 'Assigned groups';
  if (/boundary_predictions/.test(path.join(' '))) return 'Predicted value';
  return format === 'percent'
    ? 'Proportion'
    : format === 'error'
      ? 'Recorded error'
      : 'Recorded value';
}

/** Use raw source-unit error only when every compared condition actually records it. */
function rawErrors(entry: EvidenceEntry, measures: Measure[]): Measure[] {
  const raw = measures.map((measure) => {
    const path = [...measure.path];
    const last = path.at(-1);
    if (typeof last !== 'string') return measure;
    path[path.length - 1] = last
      .replace(/normalized_rmse|nrmse/g, 'rmse')
      .replace(/normalized_mae/g, 'mae');
    return { ...measure, path };
  });
  return raw.every(({ path }) => metricAt(entry.result?.metrics, path) !== null) ? raw : measures;
}

export function comparison(entry: EvidenceEntry): Comparison {
  const usable =
    !!entry.result &&
    entry.result.mode === 'full' &&
    entry.result.outcome !== 'error' &&
    entry.sourceMatches;
  const sourceMeasures: Measure[] = entry.readouts.map((readout) => ({
    label: readout.label.replace(/\b(?:nRMSE|RMSE|MSE|MAE|AUC)\b/g, '').trim() || readout.label,
    path: readout.metric,
    format: readout.format,
  }));
  let kind: Kind = 'Recorded measurement';
  let measures: Measure[] = [];
  const pair = controls[entry.id] || conditions[entry.id] || beforeAfter[entry.id];
  if (pair && pair.every((index) => sourceMeasures[index])) {
    measures = pair.map((index) => sourceMeasures[index]);
    kind = controls[entry.id]
      ? 'Controlled comparison'
      : beforeAfter[entry.id]
        ? 'Before and after'
        : 'Recorded conditions';
  } else {
    const primary = ({ P013: 1, P041: 1, P075: 1 } as Record<string, number>)[entry.id];
    const familiar =
      sourceMeasures[primary] ||
      sourceMeasures.find((measure) =>
        /RMSE|MAE|MSE|AUC|Accuracy/.test(metricName(measure.path, measure.format)),
      );
    if (familiar || sourceMeasures[0]) measures = [familiar || sourceMeasures[0]];
  }

  if (entry.id === 'P046') {
    measures = ['Metadata edit', 'Inactive extension', 'Inactive deletion'].map((label, index) => ({
      label,
      path: ['edits', index, 'max_prediction_drift'],
      format: 'number',
    }));
    kind = 'Recorded conditions';
  } else if (entry.id === 'P047' || entry.id === 'P048') {
    const arm = entry.id === 'P047' ? 'extended' : 'selective_reset';
    const target = entry.id === 'P047' ? 'w' : 'u';
    measures = ['before', 'after'].map((stage) => ({
      label: stage === 'before' ? 'Before continued training' : 'After continued training',
      path: ['arms', arm, stage, target, 'rmse'],
      format: 'error',
    }));
    kind = 'Before and after';
  } else if (entry.id === 'P049') {
    measures = [
      { label: 'Original model', path: ['source', 'rmse'], format: 'error' },
      {
        label: 'Input disabled · first cycle',
        path: ['edits', 0, 'scores', 'rmse'],
        format: 'error',
      },
      {
        label: 'Input restored · first cycle',
        path: ['edits', 1, 'scores', 'rmse'],
        format: 'error',
      },
    ];
    kind = 'Before and after';
  } else if (entry.id === 'P055') {
    measures = [sourceMeasures[1], sourceMeasures[0]].filter(Boolean);
    kind = 'Baseline reference';
  } else if (entry.id === 'P040') {
    measures = [
      { label: 'Best flattened-data reference', path: ['flat_oracle_rmse'], format: 'error' },
      { label: 'Model with nested sessions', path: ['test_rmse'], format: 'error' },
    ];
    kind = 'Baseline reference';
  } else if (entry.id === 'P079') {
    measures = [
      {
        label: 'Number inputs · Number target',
        path: ['heavy', 'number_number', 'mae'],
        format: 'error',
      },
      {
        label: 'Quantile inputs · Number target',
        path: ['heavy', 'rank_number', 'mae'],
        format: 'error',
      },
    ];
    kind = 'Controlled comparison';
  }
  measures = rawErrors(entry, measures);
  // A single error readout can have a genuine recorded constant-prediction reference.
  if (measures.length === 1 && metricName(measures[0].path, measures[0].format) === 'RMSE') {
    const base = [...measures[0].path.slice(0, -1), 'baseline_rmse'];
    if (metricAt(entry.result?.metrics, base) !== null) {
      measures = [
        { label: 'Constant-prediction baseline', path: base, format: 'error' },
        { ...measures[0], label: 'Trained model' },
      ];
      kind = 'Baseline reference';
    }
  }
  // Never put quantities with different units on a shared numerical axis.
  if (new Set(measures.map((measure) => metricName(measure.path, measure.format))).size > 1) {
    measures = measures.slice(0, 1);
    kind = 'Recorded measurement';
  }
  // The Set fallback experiment records RMSE under condition names without a metric suffix.
  const metric =
    entry.id === 'P064'
      ? 'RMSE'
      : measures[0]
        ? metricName(measures[0].path, measures[0].format)
        : 'Recorded value';
  const lower = /RMSE|MAE|MSE|Recorded error/.test(metric);
  const probability = ['AUC', 'Accuracy', 'Probability', 'Proportion'].includes(metric);
  const definitions: Record<string, string> = {
    RMSE: 'Root mean squared prediction error, in the target’s original units. Zero is an exact prediction.',
    MAE: 'Mean absolute prediction error, in the target’s original units. Zero is an exact prediction.',
    MSE: 'Mean squared prediction error, in squared target units. Zero is an exact prediction.',
    'Normalized RMSE':
      'Root mean squared error divided by the reference scale defined in the method. Values retain the recorded normalization.',
    'Normalized MAE':
      'Mean absolute error divided by the reference scale defined in the method. Values retain the recorded normalization.',
    AUC: 'Ranking score: 0.5 is chance and 1 is perfect separation. This is not the percentage of correct predictions.',
    Accuracy: 'Fraction of predictions that match the recorded test answers.',
    Probability:
      'Recorded probability or confidence. The task and its reference determine the appropriate value.',
    Proportion: 'Recorded fraction. See the method for the population and denominator.',
    'Prediction drift':
      'Change in predictions after the operation, on the scale recorded by the method. Zero means no change.',
    'Assigned groups':
      'Number of assigned groups at the recorded epoch. A matching group count alone does not establish recovery of the true memberships.',
    'Predicted value':
      'Model output for each input condition, in target units. These are predictions, not prediction errors.',
    'Recorded error':
      'Prediction error on the scale defined by the experiment’s method. No conversion to a different error metric is applied.',
    'Recorded value':
      'This is a recorded diagnostic, not an accuracy score. Its meaning and units are specified in the method.',
  };
  const points = measures.map((measure) => ({
    ...measure,
    value: usable ? metricAt(entry.result?.metrics, measure.path) : null,
  }));
  const values = points.flatMap(({ value }) => (value === null ? [] : [value]));
  const low = Math.min(0, ...values);
  const high = Math.max(probability ? 1 : 0, ...values);
  return {
    kind,
    metric,
    definition: definitions[metric],
    direction: lower
      ? 'Lower error is better'
      : ['AUC', 'Accuracy'].includes(metric)
        ? 'Higher is better'
        : 'Task-specific interpretation',
    points,
    domain: [low, high === low ? low + 1 : high],
    available: usable && points.length > 0 && points.every(({ value }) => value !== null),
  };
}
