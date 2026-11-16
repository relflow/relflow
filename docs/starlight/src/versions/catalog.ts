export interface Version {
  id: string;
  label: string;
  detail: string;
  badge: string;
  url: string;
}

export interface VersionMenu {
  current: string;
  versions: Version[];
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Use the version URLs supplied by Read the Docs, including custom domains. */
export function catalog(data: unknown, currentURL: string): VersionMenu | undefined {
  if (!object(data) || !object(data.versions)) return;
  const { current, active } = data.versions;
  if (!object(current) || typeof current.slug !== 'string' || !Array.isArray(active)) return;
  const versions = new Map<string, Version>();
  for (const item of [...active, current]) {
    if (!object(item) || typeof item.slug !== 'string' || !item.slug) continue;
    if (versions.has(item.slug)) continue;
    const address = object(item.urls) ? item.urls.documentation : undefined;
    const url =
      typeof address === 'string' ? address : item.slug === current.slug ? currentURL : '';
    let parsed: URL;
    try {
      parsed = new URL(url);
    } catch {
      continue;
    }
    if (!['https:', 'http:'].includes(parsed.protocol) || parsed.username || parsed.password)
      continue;
    const badge =
      item.slug === 'stable'
        ? 'Stable'
        : item.slug === 'latest'
          ? 'Development'
          : item.type === 'external'
            ? 'Preview'
            : '';
    versions.set(item.slug, {
      id: item.slug,
      label:
        typeof item.verbose_name === 'string' && item.verbose_name ? item.verbose_name : item.slug,
      detail:
        item.slug === current.slug
          ? 'You are reading this version'
          : badge === 'Stable'
            ? 'Latest published release'
            : badge === 'Development'
              ? 'Tracks the main branch'
              : badge === 'Preview'
                ? 'Pull request preview'
                : 'Documentation',
      badge,
      url: parsed.href,
    });
  }
  if (!versions.has(current.slug)) return;
  return { current: current.slug, versions: [...versions.values()] };
}
