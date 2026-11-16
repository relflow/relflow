/** Metric paths preserve literal object keys and support Python-style array indexes. */
export function metricAt(root: unknown, path: readonly (string | number)[]): number | null {
  let value = root;
  for (const part of path) {
    if (Array.isArray(value)) {
      if (typeof part !== 'number' || !Number.isInteger(part)) return null;
      value = value.at(part);
    } else if (value !== null && typeof value === 'object') {
      value = Object.hasOwn(value, part)
        ? (value as Record<string | number, unknown>)[part]
        : undefined;
    } else return null;
  }
  return typeof value === 'number' && Number.isFinite(value) ? value : null;
}
