// @ts-check

/** Keep authored root-relative routes inside the selected documentation version. */
export function sitePath(/** @type {string} */ value, /** @type {string} */ base) {
  if (!value.startsWith('/') || value.startsWith('//')) return value;
  const prefix = base.replace(/\/$/, '');
  if (!prefix || value === prefix || value.startsWith(`${prefix}/`)) return value;
  return prefix + value;
}
