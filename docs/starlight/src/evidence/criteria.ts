import type { EvidenceEntry } from './data';

/** Display labels come from the experiment; original criterion keys and booleans stay authoritative. */
export function criterionLabel(entry: Pick<EvidenceEntry, 'review'>, original: string): string {
  const { prefixes = {}, operations } = entry.review?.criteria || {};
  const separator = original.indexOf(':');
  const prefix = separator >= 0 ? original.slice(0, separator) : '';
  let label = prefixes[prefix] ? prefixes[prefix] + original.slice(separator) : original;
  const names: Record<string, string> = {
    max: 'Maximum',
    min: 'Minimum',
    mean: 'Average',
    sum: 'Total',
  };
  if (operations === 'group')
    label = label.replace(
      /^(max|min|mean|sum)\/([ABC]) /,
      (_, operation: string, group: string) => names[operation] + ' for group ' + group + ': ',
    );
  if (operations === 'request')
    label = label.replace(
      /^(max|min|mean|sum) /,
      (_, operation: string) => names[operation] + ' request: ',
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
