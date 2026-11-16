import type { Value, ValueNotes } from './types';

export interface Candidate {
  value: Value;
  probability: number;
}

export type PredictionView =
  | { kind: 'scalar'; value: Value }
  | { kind: 'probability'; probability: number }
  | { kind: 'category'; value: Value; probability: number; candidates: Candidate[] }
  | { kind: 'memberships'; candidates: Candidate[] }
  | { kind: 'vector'; values: number[] }
  | { kind: 'empty' }
  | { kind: 'structured' };

export interface ValueNodeData {
  id: string;
  name: string;
  field?: string;
  context?: string;
  note?: string;
  kind: 'record' | 'list' | 'value' | 'prediction';
  value?: Value;
  children: ValueNodeData[];
  view?: PredictionView;
  returned?: ValueNodeData[];
}

interface TreeOptions {
  id: string;
  predictions?: boolean;
  inputs?: Record<string, Value>;
  notes?: ValueNotes;
}

function record(value: Value | undefined): value is Record<string, Value> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function probability(value: Value | undefined): value is number {
  return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1;
}

function candidate(value: Value): value is Record<string, Value> & Candidate {
  return record(value) && 'value' in value && probability(value.probability);
}

/** Recognize only the public content shapes; unknown outputs retain their full value tree. */
export function predictionView(content: Value): PredictionView {
  if (content === null || typeof content !== 'object') return { kind: 'scalar', value: content };
  if (Array.isArray(content)) {
    if (!content.length) return { kind: 'empty' };
    if (content.every((value) => typeof value === 'number' && Number.isFinite(value)))
      return { kind: 'vector', values: content as number[] };
    if (content.every(candidate))
      return { kind: 'memberships', candidates: content as Candidate[] };
    return { kind: 'structured' };
  }
  if (probability(content.probability) && 'value' in content) {
    if (
      content.topk !== undefined &&
      (!Array.isArray(content.topk) || !content.topk.every(candidate))
    )
      return { kind: 'structured' };
    return {
      kind: 'category',
      value: content.value,
      probability: content.probability,
      candidates: (content.topk as Candidate[]) || [],
    };
  }
  if (probability(content.probability) && Object.keys(content).length === 1)
    return { kind: 'probability', probability: content.probability };
  return { kind: 'structured' };
}

/** Strings stay quoted so decoded codes, empty strings, booleans, and null are distinct. */
export function formatValue(value: Value): string {
  return JSON.stringify(value);
}

export function percent(value: number): string {
  return `${Number((value * 100).toFixed(4))}%`;
}

/** Per-vector relative magnitudes describe coordinates, never prediction uncertainty. */
export function magnitudes(values: readonly number[]): number[] {
  let largest = 0;
  for (const value of values) largest = Math.max(largest, Math.abs(value));
  return values.map((value) => (largest ? (Math.abs(value) / largest) * 100 : 0));
}

/** A lossless display projection; the original values are never normalized or mutated. */
export function tree(value: Value, options: TreeOptions): ValueNodeData {
  if (!options.id || /\s/.test(options.id))
    throw new Error('ValueTree needs a page-unique id without whitespace.');
  let ordinal = 0;
  function visit(
    current: Value,
    name: string,
    path: (string | number)[],
    input?: Value,
    raw = false,
  ): ValueNodeData {
    const node: ValueNodeData = {
      id: `${options.id}-node-${ordinal++}`,
      name,
      kind: 'value',
      value: current,
      children: [],
      note: options.notes?.[
        path.map((part) => `/${String(part).replaceAll('~', '~0').replaceAll('/', '~1')}`).join('')
      ],
    };
    const part = path.at(-1);
    if (typeof part === 'string' && !['predictions', 'representations', 'inputs'].includes(part))
      node.field = part;
    if (typeof part === 'number' && record(input)) {
      const context = Object.entries(input)
        .filter(([, value]) => value !== null && typeof value !== 'object')
        .slice(0, 2);
      if (context.length)
        node.context = context.map(([key, value]) => `${key}: ${formatValue(value)}`).join(' · ');
    }
    if (
      !raw &&
      options.predictions &&
      path[0] === 'predictions' &&
      record(current) &&
      'content' in current &&
      record(current.state) &&
      typeof current.inferred === 'boolean'
    ) {
      node.kind = 'prediction';
      node.view = predictionView(current.content);
      node.returned = Object.entries(current).map(([key, child]) =>
        visit(child, key, [...path, key], undefined, true),
      );
      return node;
    }
    if (Array.isArray(current)) {
      node.kind = 'list';
      node.children = current.map((child, index) =>
        visit(
          child,
          `[${index}]`,
          [...path, index],
          Array.isArray(input) ? input[index] : undefined,
          raw,
        ),
      );
    } else if (record(current)) {
      node.kind = 'record';
      node.children = Object.entries(current).map(([key, child]) => {
        const context =
          !path.length && key === 'predictions'
            ? options.inputs
            : record(input)
              ? input[key]
              : undefined;
        return visit(child, key, [...path, key], context, raw);
      });
    }
    return node;
  }
  return visit(value, 'row', [], options.inputs);
}
