import type { EvidenceEntry } from './data';
import { reviews, type Review } from './reviews.ts';

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
    'P100',
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
    'P060',
    'P061',
    'P062',
    'P066',
    'P100',
  ],
  Boolean: ['P017'],
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

export function review(entry: Pick<EvidenceEntry, 'id'>): Review {
  const content = reviews[entry.id];
  if (!content)
    throw new Error(
      `Evidence ${entry.id} needs an individually reviewed explanation and comparison in reviews.ts.`,
    );
  return content;
}

export function question(entry: Pick<EvidenceEntry, 'id' | 'title'>): string {
  return review(entry).question;
}

export function description(entry: Pick<EvidenceEntry, 'id' | 'description'>): string {
  return review(entry).description || entry.description;
}

export function modeling(entry: EvidenceEntry): string {
  return entry.sourceMatches && entry.source ? review(entry).modeling : '';
}
