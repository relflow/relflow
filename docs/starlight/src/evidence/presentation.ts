interface ResultEntry {
  result: { mode: string; outcome: string } | null;
}

export function status(entry: ResultEntry) {
  if (!entry.result) return { id: 'missing', label: 'No selected run' };
  if (entry.result.outcome === 'error') return { id: 'error', label: 'Execution error' };
  if (entry.result.mode !== 'full') return { id: 'smoke', label: 'Limited run' };
  if (entry.result.outcome === 'met') return { id: 'met', label: 'Checks met' };
  return { id: 'not_met', label: 'Checks unmet' };
}

export function formatValue(value: unknown, format = 'number'): string {
  if (typeof value !== 'number' || !Number.isFinite(value)) return 'Not recorded';
  if (format === 'percent')
    return `${(value * 100).toLocaleString('en-US', { maximumFractionDigits: 2 })}%`;
  if (format === 'auc') return value.toFixed(4);
  return value.toLocaleString('en-US', { maximumSignificantDigits: 6 });
}

export function recordDate(value: string): string {
  return new Date(value).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    timeZone: 'UTC',
  });
}

export function experimentURL(id: string): string {
  return `/evidence/${id}/`;
}
