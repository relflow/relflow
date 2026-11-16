/** Illustrative versions for local previews, separate from hosted version discovery. */
export const versions = [
  {
    id: 'preview',
    label: '0.5.0 preview',
    detail: 'The docs you are reviewing',
    badge: 'PR 102',
    group: 'Development',
  },
  {
    id: 'latest',
    label: 'latest',
    detail: 'Tracks the main branch',
    badge: 'main',
    group: 'Development',
  },
  {
    id: 'v0.4.7',
    label: '0.4.7',
    detail: 'Latest published release',
    badge: 'Stable',
    group: 'Releases',
  },
  { id: 'v0.4.6', label: '0.4.6', detail: 'Previous release', badge: '', group: 'Releases' },
  { id: 'v0.4.5', label: '0.4.5', detail: 'Previous release', badge: '', group: 'Releases' },
] as const;
