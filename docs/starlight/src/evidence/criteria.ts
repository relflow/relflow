/** Display labels only: original criterion keys and booleans remain authoritative. */
const prefixes: Record<string, Record<string, string>> = {
  P053: {
    hiddenhidden: 'Both views hidden',
    hiddenvisible: 'First view hidden',
    visiblehidden: 'Second view hidden',
    visiblevisible: 'Both views available',
  },
  P060: {
    capacity_2: 'Two-row output allocation',
    capacity_128: '128-row output allocation',
    capacity_2_known: 'Two-row allocation · familiar inputs',
    capacity_2_novel: 'Two-row allocation · unseen inputs',
    capacity_128_known: '128-row allocation · familiar inputs',
    capacity_128_novel: '128-row allocation · unseen inputs',
  },
  P061: { '0.01': '1% input unavailability', '0.2': '20% input unavailability' },
  P062: {
    '0': 'No known targets',
    '102': '102 known targets',
    '512': '512 known targets',
    '1024': '1,024 known targets',
  },
  P063: { '2': 'Two-row output allocation', '512': '512-row output allocation' },
  P065: {
    '0': 'No known targets',
    '102': '102 known targets',
    '512': '512 known targets',
    '1024': '1,024 known targets',
  },
  P066: { '0': 'Initial cohort', '1': 'Second cohort', '2': 'Third cohort' },
  P073: {
    untrained: 'Untrained encoder',
    reconstruction: 'Reconstruction pretraining',
    permuted_views: 'Shuffled-view pretraining',
  },
  P074: {
    '000': 'No sensors visible',
    '001': 'Sensor 3 visible',
    '010': 'Sensor 2 visible',
    '011': 'Sensors 2 and 3 visible',
    '100': 'Sensor 1 visible',
    '101': 'Sensors 1 and 3 visible',
    '110': 'Sensors 1 and 2 visible',
    '111': 'All three sensors visible',
  },
  P077: {
    amount_number_clean: 'Numeric task · Number · clean inputs',
    amount_number_contaminated: 'Numeric task · Number · contaminated inputs',
    amount_quantile_clean: 'Numeric task · Quantile · clean inputs',
    amount_quantile_contaminated: 'Numeric task · Quantile · contaminated inputs',
    rank_number_clean: 'Rank task · Number · clean inputs',
    rank_number_contaminated: 'Rank task · Number · contaminated inputs',
    rank_quantile_clean: 'Rank task · Quantile · clean inputs',
    rank_quantile_contaminated: 'Rank task · Quantile · contaminated inputs',
    number_mae: 'Number target · MAE objective',
    number_mse: 'Number target · MSE objective',
    quantile_mae: 'Quantile target · MAE objective',
    quantile_mse: 'Quantile target · MSE objective',
  },
};

export function criterionLabel(id: string, original: string): string {
  const separator = original.indexOf(':');
  const prefix = separator >= 0 ? original.slice(0, separator) : '';
  let label = prefixes[id]?.[prefix]
    ? `${prefixes[id][prefix]}${original.slice(separator)}`
    : original;
  if (id === 'P027')
    label = label.replace(
      /^(max|min|mean|sum)\/([ABC]) /,
      (_, operation: string, group: string) =>
        `${{ max: 'Maximum', min: 'Minimum', mean: 'Average', sum: 'Total' }[operation]} for group ${group}: `,
    );
  if (id === 'P031')
    label = label.replace(
      /^(max|min|mean|sum) /,
      (_, operation: string) =>
        `${{ max: 'Maximum', min: 'Minimum', mean: 'Average', sum: 'Total' }[operation]} request: `,
    );
  return label
    .replace(/\bnRMSE\b/g, 'normalized RMSE')
    .replace(/\bSD\b/g, 'standard deviations')
    .replace(/\bARI\b/g, 'group agreement (ARI)')
    .replace(/\bNLL\b/g, 'negative log likelihood')
    .replace(/All-OOV/g, 'Unknown-only')
    .replace(/all-OOV/g, 'unknown-only')
    .replace(/\bOOV\b/g, 'unseen-label')
    .replace(/\b([a-z][a-z0-9]*(?:_[a-z0-9]+)+)\b/g, (token) => token.replaceAll('_', ' '))
    .replace(/>=/g, '≥')
    .replace(/<=/g, '≤');
}
