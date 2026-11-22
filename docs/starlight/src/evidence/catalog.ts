import type { EvidenceEntry } from './data';
import { parse } from 'smol-toml';

/** Shared vocabulary only; each experiment owns its tag assignments. */
export const leafTypes = [
  'Number',
  'Quantile',
  'Vector',
  'Coordinates',
  'Ordinal',
  'Category',
  'Boolean',
  'Enum',
  'Hash',
  'Set',
  'Group',
  'Period',
  'Duration',
  'Text',
  'Image',
  'Audio',
] as const;
export type LeafType = (typeof leafTypes)[number];

export const capabilities = [
  {
    id: 'computation',
    label: 'Computation',
  },
  {
    id: 'structure',
    label: 'Structure',
  },
  {
    id: 'representation',
    label: 'Representation',
  },
  {
    id: 'training',
    label: 'Training',
  },
  {
    id: 'reliability',
    label: 'Reliability',
  },
] as const;

export const conditions = [
  {
    id: 'unseen-identities',
    label: 'Unseen identities',
  },
  {
    id: 'longer-collections',
    label: 'Longer than training',
  },
  {
    id: 'new-combinations',
    label: 'Held-out combinations',
  },
  {
    id: 'unseen-values',
    label: 'Held-out labels or ranks',
  },
  {
    id: 'unseen-inputs',
    label: 'Held-out input values',
  },
  {
    id: 'extrapolation',
    label: 'Outside the training range',
  },
  {
    id: 'missing-inputs',
    label: 'Missing inputs',
  },
  {
    id: 'value-states',
    label: 'Null, empty, or invalid values',
  },
  {
    id: 'changed-associations',
    label: 'Broken associations',
  },
  {
    id: 'transformations',
    label: 'Known transformations',
  },
  {
    id: 'varying-collections',
    label: 'Varying collections',
  },
  {
    id: 'hidden-values',
    label: 'Hidden-value leakage',
  },
  {
    id: 'noise',
    label: 'Noisy or unrelated signals',
  },
  {
    id: 'limited-information',
    label: 'Limited information',
  },
  {
    id: 'training-updates',
    label: 'Training and schema changes',
  },
  {
    id: 'transfer',
    label: 'Transfer learning',
  },
  {
    id: 'objectives',
    label: 'Training objectives',
  },
  {
    id: 'representations',
    label: 'Representation comparisons',
  },
  {
    id: 'numeric-extremes',
    label: 'Numerical extremes',
  },
  {
    id: 'fresh-records',
    label: 'Fresh records',
  },
] as const;

export interface Tags {
  capability: (typeof capabilities)[number]['id'];
  types: LeafType[];
  conditions: (typeof conditions)[number]['id'][];
}

export function tags(source: string, location = 'Evidence source'): Tags | null {
  const starts = source.split(/\r?\n/).filter((line) => line === '# /// script');
  if (!starts.length) return null;
  const blocks = [...source.matchAll(/^# \/\/\/ script\r?\n((?:#(?: .*)?\r?\n)*)# \/\/\/\r?$/gm)];
  if (starts.length !== 1 || blocks.length !== 1)
    throw new Error(`${location}: expected exactly one closed PEP 723 script block.`);
  const document = parse(
    blocks[0][1]
      .split(/\r?\n/)
      .map((line) => line.slice(2))
      .join('\n'),
  );
  const table = (document.tool as Record<string, unknown> | undefined)?.relflow as
    Record<string, unknown> | undefined;
  const declared = table?.evidence as Record<string, unknown> | undefined;
  if (declared === undefined) return null; // Retained release sources may predate tags.
  if (!declared || typeof declared !== 'object' || Array.isArray(declared))
    throw new Error(`${location}: tags must be a [tool.relflow.evidence] table.`);
  if (!capabilities.some((item) => item.id === declared.capability))
    throw new Error(
      `${location}: unknown capability ${JSON.stringify(declared.capability)}; use ${capabilities.map((item) => item.id).join(', ')}.`,
    );
  for (const [name, vocabulary] of [
    ['types', leafTypes],
    ['conditions', conditions.map((item) => item.id)],
  ] as const) {
    const values = declared[name];
    if (
      !Array.isArray(values) ||
      (name === 'conditions' && !values.length) ||
      values.some(
        (value) => typeof value !== 'string' || !(vocabulary as readonly string[]).includes(value),
      ) ||
      new Set(values).size !== values.length
    )
      throw new Error(
        `${location}: ${name} ${JSON.stringify(values)} must be a unique list from: ${vocabulary.join(', ')}.`,
      );
  }
  const unknown = Object.keys(declared).filter(
    (key) => !['capability', 'types', 'conditions'].includes(key),
  );
  if (unknown.length) throw new Error(`${location}: unknown tag fields: ${unknown.join(', ')}.`);
  return declared as unknown as Tags;
}

export function typesFor(entry: Pick<EvidenceEntry, 'tags'>): LeafType[] {
  return entry.tags?.types || [];
}

export function experimentsForType<T extends Pick<EvidenceEntry, 'tags'>>(
  entries: readonly T[],
  type: LeafType,
): T[] {
  return entries.filter((entry) => typesFor(entry).includes(type));
}

export function capabilityFor(entry: Pick<EvidenceEntry, 'id' | 'tags'>) {
  const capability = capabilities.find((item) => item.id === entry.tags?.capability);
  if (!capability)
    throw new Error(`Evidence ${entry.id} needs a capability in its script metadata.`);
  return capability;
}

export function conditionsFor(entry: Pick<EvidenceEntry, 'tags'>) {
  return conditions.filter((item) => entry.tags?.conditions.includes(item.id));
}

export function review(entry: Pick<EvidenceEntry, 'review'>) {
  return entry.review;
}

export function question(entry: Pick<EvidenceEntry, 'review' | 'title'>): string {
  return entry.review.question;
}

export function description(entry: Pick<EvidenceEntry, 'review' | 'description'>): string {
  return entry.review.description || entry.description;
}

export function modeling(entry: EvidenceEntry): string {
  return entry.sourceMatches && entry.source ? entry.review.modeling : '';
}
