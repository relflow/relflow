import type { EvidenceEntry } from './data';

/** Primary subjects, not every leaf used by the generator or prediction target. */
const subjects = {
  Number: [
    'P001',
    'P002',
    'P003',
    'P004',
    'P005',
    'P006',
    'P007',
    'P008',
    'P009',
    'P010',
    'P011',
    'P013',
    'P014',
    'P015',
    'P016',
    'P038',
    'P039',
    'P040',
    'P041',
    'P077',
    'P079',
    'P087',
    'P088',
    'P093',
    'P099',
  ],
  Quantile: ['P067', 'P068', 'P077', 'P079'],
  Vector: [],
  Ordinal: ['P100'],
  Category: [
    'P021',
    'P036',
    'P055',
    'P056',
    'P057',
    'P059',
    'P061',
    'P062',
    'P066',
    'P069',
    'P070',
    'P086',
  ],
  Boolean: ['P017', 'P060'],
  Enum: ['P069', 'P070'],
  Hash: ['P021', 'P025', 'P028', 'P029', 'P037', 'P075', 'P098'],
  Set: ['P058', 'P063', 'P064'],
  Group: ['P108', 'P109', 'P110', 'P111', 'P112'],
  Period: ['P042', 'P043', 'P044', 'P045'],
  Duration: ['P099'],
  Text: ['P092'],
} as const;

export type LeafType = keyof typeof subjects;
export const leafTypes = Object.keys(subjects) as LeafType[];

export function typesFor(id: string): LeafType[] {
  return leafTypes.filter((type) => (subjects[type] as readonly string[]).includes(id));
}

export function experimentsForType<T extends { id: string }>(
  entries: readonly T[],
  type: LeafType,
): T[] {
  return entries.filter((entry) => typesFor(entry.id).includes(type));
}

export function question(entry: Pick<EvidenceEntry, 'title'>): string {
  return entry.title.replace(/\bit\b/, 'relflow');
}

/** Explain the setup without promoting the source's authored conclusions to CI outcomes. */
const setups: Record<string, string> = {
  P001: 'Generate independent lists of one to six amounts between −1 and 1. Give the model the amounts in a repeated branch and hide their sum as a Number target. Training, validation, and test use separate lists. Compare test RMSE with a predictor that always returns a constant.',
  P021: 'Give the model two identifiers and hide a Boolean target indicating whether they match. Training and test use disjoint identifier names. Compare Hash with Category on the same equality task: new Category labels share the unknown representation, whereas Hash retains equality within an encoded batch. A shuffled-label control checks for an accidental signal.',
  P038: 'Generate records in which a measurement is either null or the number zero, and hide the Boolean answer identifying which case occurred. Compare preserving these states with replacing nulls by zero. Filling nulls removes the information needed to distinguish the two cases; the test measures whether the model uses the preserved distinction.',
  P039: 'Generate independent records containing exactly six amounts sampled between −1 and 1. Put the amounts in a repeated branch and hide their sum as a Number target. Score held-out totals with RMSE and compare with a predictor that always returns the training-set average total.',
  P040: 'Pair records containing the same six amounts but arrange their sessions differently: two items in each session, or one, four, and one. Hide the structure-dependent target. A reference that sees only flattened values must return the same answer for both records; the model can use their different nesting.',
};

export function modeling(entry: EvidenceEntry): string {
  if (!entry.sourceMatches || !entry.source) return '';
  if (setups[entry.id]) return setups[entry.id];
  // These paragraphs come from the exact selected source, not the live SDK.
  return (
    entry.sections.method
      .replace(/^###[^\n]*\n+/, '')
      .split(/\n### /)[0]
      .split(/\n\s*\n/)
      .filter((paragraph) => paragraph.trim() && !paragraph.startsWith('#'))
      .slice(0, 2)
      .join('\n\n')
      .trim() || entry.description
  );
}
