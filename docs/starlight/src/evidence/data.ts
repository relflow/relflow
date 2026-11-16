import { metricAt } from './metrics.ts';
import { createHash } from 'node:crypto';
import { execFileSync } from 'node:child_process';
import { existsSync, readFileSync, readdirSync } from 'node:fs';
import { basename, dirname, join, resolve } from 'node:path';
import { parse } from 'yaml';
import { tags, type Tags } from './catalog.ts';

export type MetricValue =
  string | number | boolean | null | MetricValue[] | { [key: string]: MetricValue };

interface EvidenceProvenance {
  evidence_revision?: string;
  relflow_revision?: string;
  script_sha256?: string;
  packages?: Record<string, string>;
  [key: string]: unknown;
}

export interface EvidenceRecord {
  outcome: 'met' | 'not_met' | 'error';
  mode: 'full';
  checks: Record<string, boolean>;
  metrics: Record<string, MetricValue>;
  provenance: EvidenceProvenance;
  seed: number;
  started_at: string;
  duration_seconds: number;
  accelerator: string;
  threads: number;
  steps_override: number | null;
  [key: string]: unknown;
}

interface EvidenceReport {
  release?: string;
  run_id?: string;
  passed?: boolean;
  github_run?: { html_url?: string; run_id?: number; run_attempt?: number; [key: string]: unknown };
  [key: string]: unknown;
}

interface Publication {
  schema: 1;
  version: string;
  revision: string;
  report: EvidenceReport;
  results: Record<string, EvidenceRecord>;
}

interface Sections {
  intro: string;
  example: string;
  comparison: string;
  result: string;
  method: string;
}

interface Readout {
  label: string;
  metric: (string | number)[];
  format: string;
  value: number | null;
}

export interface EvidenceEntry {
  id: string;
  tags: Tags | null;
  title: string;
  description: string;
  family: string;
  lane: 'core' | 'research';
  requirements: string[];
  sourcePath: string;
  sourceRevision: string;
  sourceUrl: string;
  sourceSha256: string;
  source: string;
  code: string;
  sections: Sections;
  diagram: { caption: string; alt: string };
  readouts: Readout[];
  result: EvidenceRecord | null;
  sourceMatches: boolean;
  archived: boolean;
}

type Header = Record<string, unknown>;

interface Declaration {
  id: string;
  tags: Tags | null;
  header: Header;
  source: string;
  sourcePath: string;
  lane: 'core' | 'research';
  archived?: boolean;
}

const site = process.cwd();
const root = resolve(site, '../..');
const explicit = process.env.RELFLOW_EVIDENCE_PUBLICATION;
const publicationPath = explicit ? resolve(site, explicit) : 'no selected publication';
const raw = explicit
  ? readFileSync(publicationPath, 'utf8')
  : JSON.stringify({
      schema: 1,
      version: 'pending',
      revision:
        process.env.RELFLOW_DOCS_REVISION || git(['rev-parse', 'HEAD'])?.trim() || 'unselected',
      report: {},
      results: {},
    });
const publication: unknown = JSON.parse(raw);

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function digest(source: string): string {
  return createHash('sha256').update(source, 'utf8').digest('hex');
}

function metric(value: unknown): value is MetricValue {
  if (value === null || typeof value === 'string' || typeof value === 'boolean') return true;
  if (typeof value === 'number') return Number.isFinite(value);
  if (Array.isArray(value)) return value.every(metric);
  return object(value) && Object.values(value).every(metric);
}

function validate(value: unknown): asserts value is Publication {
  if (
    !object(value) ||
    value.schema !== 1 ||
    typeof value.version !== 'string' ||
    !value.version.trim() ||
    typeof value.revision !== 'string' ||
    !value.revision.trim() ||
    !object(value.report) ||
    !object(value.results)
  ) {
    throw new Error(
      `Evidence publication ${publicationPath} must contain schema 1, version, revision, report, and results.`,
    );
  }
  const report = value.report;
  if (
    ['release', 'run_id'].some(
      (name) => report[name] !== undefined && typeof report[name] !== 'string',
    ) ||
    (report.passed !== undefined && typeof report.passed !== 'boolean') ||
    (report.github_run !== undefined &&
      (!object(report.github_run) ||
        (report.github_run.html_url !== undefined &&
          typeof report.github_run.html_url !== 'string')))
  ) {
    throw new Error(
      `Evidence publication ${publicationPath} has invalid report links, run identifier, or passed flag.`,
    );
  }
  for (const [id, record] of Object.entries(value.results)) {
    if (!/^P\d+$/.test(id) || !object(record)) {
      throw new Error(
        `Evidence record ${id} in ${publicationPath} must be an object with a stable P identifier.`,
      );
    }
    const fields: [string, boolean, string][] = [
      ['mode', record.mode === 'full', 'a full recorded run'],
      [
        'outcome',
        record.outcome === 'met' || record.outcome === 'not_met' || record.outcome === 'error',
        'met, not_met, or error',
      ],
      [
        'checks',
        object(record.checks) &&
          Object.values(record.checks).every((check) => typeof check === 'boolean'),
        'a dictionary of Boolean check outcomes',
      ],
      [
        'metrics',
        object(record.metrics) && metric(record.metrics),
        'a dictionary of finite JSON measurements',
      ],
      [
        'provenance',
        object(record.provenance) && metric(record.provenance),
        'a dictionary of finite JSON provenance values',
      ],
      ['seed', Number.isSafeInteger(record.seed), 'a safe integer'],
      [
        'started_at',
        typeof record.started_at === 'string' &&
          /(?:Z|[+-]\d{2}:\d{2})$/.test(record.started_at) &&
          Number.isFinite(Date.parse(record.started_at)),
        'a valid timestamp with a UTC offset',
      ],
      [
        'duration_seconds',
        typeof record.duration_seconds === 'number' &&
          Number.isFinite(record.duration_seconds) &&
          record.duration_seconds >= 0,
        'a finite nonnegative number',
      ],
      [
        'accelerator',
        typeof record.accelerator === 'string' &&
          ['cpu', 'gpu', 'mps', 'auto'].includes(record.accelerator),
        'cpu, gpu, mps, or auto',
      ],
      [
        'threads',
        typeof record.threads === 'number' &&
          Number.isSafeInteger(record.threads) &&
          record.threads > 0,
        'a positive safe integer',
      ],
      ['steps_override', record.steps_override === null, 'null for a full-budget run'],
      [
        'error',
        record.error === undefined || typeof record.error === 'string',
        'a string when supplied',
      ],
    ];
    for (const [name, valid, expected] of fields) {
      if (!valid)
        throw new Error(
          `Evidence record ${id} in ${publicationPath}: ${name} must be ${expected}; received ${JSON.stringify(record[name])}.`,
        );
    }
    const provenance = record.provenance as Record<string, unknown>;
    for (const name of ['evidence_revision', 'relflow_revision', 'script_sha256']) {
      if (provenance[name] !== undefined && typeof provenance[name] !== 'string') {
        throw new Error(
          `Evidence record ${id} in ${publicationPath}: provenance.${name} must be a string when supplied.`,
        );
      }
    }
    if (
      provenance.packages !== undefined &&
      (!object(provenance.packages) ||
        Object.values(provenance.packages).some((version) => typeof version !== 'string'))
    ) {
      throw new Error(
        `Evidence record ${id} in ${publicationPath}: provenance.packages must map package names to version strings.`,
      );
    }
  }
}

validate(publication);

/** Every status, check, metric, and readout comes from this one selected publication. */
export const selection = {
  version: publication.version,
  revision: publication.revision,
  report: publication.report,
  raw,
  sha256: digest(raw),
  mode: explicit ? ('selected' as const) : ('pending' as const),
};

function git(args: string[]): string | null {
  try {
    return execFileSync('git', args, {
      cwd: root,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'ignore'],
      maxBuffer: 8 * 1024 * 1024,
    });
  } catch {
    return null;
  }
}

const checkoutRevision = git(['rev-parse', 'HEAD'])?.trim() || selection.revision;

function cells(source: string): { markdown: boolean; text: string }[] {
  const found: { markdown: boolean; text: string }[] = [];
  let markdown = false;
  let buffer = '';
  for (const line of source.match(/[^\n]*\n|[^\n]+$/g) || []) {
    const marker = /^# %%(?:\s+\[(markdown|raw)\])?[^\r\n]*(?:\r?\n)?$/.exec(line);
    if (marker) {
      if (buffer) found.push({ markdown, text: buffer });
      buffer = '';
      markdown = marker[1] === 'markdown';
    } else {
      buffer += markdown ? line.replace(/^# ?/, '') : line;
    }
  }
  if (buffer) found.push({ markdown, text: buffer });
  return found;
}

function header(source: string): Header {
  const first = cells(source)[0];
  const match = first?.markdown && /^\s*---\r?\n([\s\S]*?)\r?\n---(?:\r?\n|$)/.exec(first.text);
  const value: unknown = match ? parse(match[1]) : null;
  if (!object(value))
    throw new Error(
      'Evidence source must begin with commented YAML frontmatter in a Markdown cell.',
    );
  return value;
}

function identify(source: string, metadata: Header): string {
  const matches = [
    ...source.matchAll(/^@experiment\(\s*(['"])(P\d+)\1\s*,\s*seed\s*=\s*\d+\s*\)\s*$/gm),
  ];
  if (matches.length > 1 || (/@experiment\b/.test(source) && matches.length !== 1)) {
    throw new Error(
      'Evidence source must have exactly one literal @experiment("Pxxx", seed=N) declaration.',
    );
  }
  const id = matches[0]?.[2] || metadata['evidence-id'];
  if (
    typeof id !== 'string' ||
    !/^P\d+$/.test(id) ||
    (metadata['evidence-id'] !== undefined && metadata['evidence-id'] !== id)
  ) {
    throw new Error('Evidence source has a missing or conflicting stable P identifier.');
  }
  return id;
}

function declarations(directory = join(root, 'evidence'), required = true): Declaration[] {
  const entries: Declaration[] = [];
  for (const [collection, lane] of [
    ['experiments', 'core'],
    ['research', 'research'],
  ] as const) {
    const parent = join(directory, collection);
    if (!existsSync(parent)) continue;
    for (const family of readdirSync(parent, { withFileTypes: true })
      .filter((entry) => entry.isDirectory())
      .sort((a, b) => a.name.localeCompare(b.name))) {
      for (const file of readdirSync(join(parent, family.name), { withFileTypes: true })
        .filter((entry) => entry.isFile() && entry.name.endsWith('.py'))
        .sort((a, b) => a.name.localeCompare(b.name))) {
        const sourcePath = `evidence/${collection}/${family.name}/${file.name}`;
        const source = readFileSync(join(parent, family.name, file.name), 'utf8');
        const metadata = header(source);
        const id = identify(source, metadata);
        const classified = tags(source, sourcePath);
        if (required && classified === null)
          throw new Error(
            `${sourcePath}: declare [tool.relflow.evidence] capability, types, and conditions.`,
          );
        entries.push({
          id,
          tags: classified,
          header: metadata,
          source,
          sourcePath,
          lane,
        });
      }
    }
  }
  if (new Set(entries.map((entry) => entry.id)).size !== entries.length)
    throw new Error('Evidence declarations contain duplicate P identifiers.');
  return entries.sort((a, b) => Number(a.id.slice(1)) - Number(b.id.slice(1)));
}

const declared = declarations();
const frozen = explicit ? declarations(join(dirname(publicationPath), 'source'), false) : [];
const identifiers = new Set(declared.map((entry) => entry.id));
for (const [id, record] of Object.entries(publication.results)) {
  if (!identifiers.has(id)) {
    const expected = record.provenance.script_sha256;
    const entry = frozen.find((entry) => entry.id === id);
    if (!entry || digest(entry.source) !== expected) {
      throw new Error(
        `Evidence record ${id} in ${publicationPath} has no experiment declaration in this checkout or an exact retained CI source. Select its verified reports/source directory alongside the publication.`,
      );
    }
    declared.push({ ...entry, archived: true });
  }
}
declared.sort((a, b) => Number(a.id.slice(1)) - Number(b.id.slice(1)));

function page(sourcePath: string): string {
  return `docs/evidence/${basename(dirname(sourcePath))}/${basename(sourcePath, '.py').replaceAll('_', '-')}`;
}

const pages = new Map(declared.map((entry) => [page(entry.sourcePath), entry.id]));

function links(markdown: string, sourcePath: string, revision: string): string {
  return markdown.replace(
    /(!?\[[^\]]*\])\(([^)]+)\)/g,
    (original, label: string, target: string) => {
      if (/^(?:[a-z][a-z\d+.-]*:|#)/i.test(target)) return original;
      if (/^\/evidence\/P\d+\//.test(target) || target.startsWith('/data-types/')) return original;
      if (target.startsWith('/') && !target.startsWith('//')) {
        const route =
          new URL(target, 'https://evidence.invalid').pathname.replace(/^\/|\/$/g, '') || 'index';
        if (
          ['.mdx', '/index.mdx'].some((suffix) =>
            existsSync(join(site, 'src/content/docs', route + suffix)),
          )
        )
          return original;
      }
      const local = new URL(target, `https://evidence.invalid/${page(sourcePath)}.py`);
      const pathname = decodeURIComponent(local.pathname).replace(/^\//, '');
      const experiment = pages.get(pathname.replace(/\.(?:html|qmd|py)$/, ''));
      if (experiment) return `${label}(/evidence/${experiment}/)`;
      const datatype = /^docs\/datatypes\/([^/]+)\.qmd$/.exec(pathname)?.[1];
      const slug = datatype === 'index' ? '' : datatype;
      if (slug !== undefined) {
        const filename = slug ? `data-types/${slug}.mdx` : 'data-types/index.mdx';
        try {
          readFileSync(join(site, 'src/content/docs', filename));
          return `${label}(/data-types/${slug ? `${slug}/` : ''})`;
        } catch {
          /* Legacy-only pages retain their exact archived source link. */
        }
      }
      return `${label}(https://github.com/relflow/relflow/blob/${revision}/${pathname})`;
    },
  );
}

function presentation(
  source: string,
  sourcePath: string,
  revision: string,
): { code: string; sections: Sections; diagram: { caption: string; alt: string } } {
  const parts = cells(source);
  const code = parts
    .filter((part) => !part.markdown)
    .map((part) => part.text)
    .join('');
  let markdown = parts
    .filter((part) => part.markdown)
    .map((part) => part.text)
    .join('\n');
  markdown = markdown.replace(/^\s*---\r?\n[\s\S]*?\r?\n---(?:\r?\n|$)/, '');
  const diagram = { caption: '', alt: '' };
  markdown = markdown.replace(
    /^```\{typst\}[^\n]*\n([\s\S]*?)^```[ \t]*$/gm,
    (_block, body: string) => {
      for (const match of body.matchAll(/^\s*\/\/\|\s*fig-(cap|alt):\s*(.+)$/gm)) {
        const value: unknown = parse(match[2]);
        if (typeof value === 'string') {
          if (match[1] === 'cap' && !diagram.caption) diagram.caption = value;
          if (match[1] === 'alt' && !diagram.alt) diagram.alt = value;
        }
      }
      return '';
    },
  );
  markdown = markdown.replace(
    /<details class="evidence-details">\s*<summary>Experiment details and code<\/summary>/,
    '\n## __method__\n',
  );
  const closing = markdown.lastIndexOf('</details>');
  if (closing !== -1)
    markdown = markdown.slice(0, closing) + markdown.slice(closing + '</details>'.length);
  markdown = markdown
    .replace(/\{\{<\s*evidence\s+[^>]*>\}\}/g, '')
    .replace(/^### (?:Run this experiment|Complete experiment code)[ \t]*$/gm, '');
  const sections: Sections = { intro: '', example: '', comparison: '', result: '', method: '' };
  let active: keyof Sections = 'intro';
  for (const line of markdown.split('\n')) {
    const heading = /^## (Example|Comparison|Result|__method__)[ \t]*$/.exec(line);
    if (heading)
      active =
        heading[1] === '__method__' ? 'method' : (heading[1].toLowerCase() as keyof Sections);
    else sections[active] += `${line}\n`;
  }
  for (const key of Object.keys(sections) as (keyof Sections)[]) {
    sections[key] = links(sections[key].trim(), sourcePath, revision).replace(/\n{3,}/g, '\n\n');
  }
  return { code, sections, diagram };
}

function readouts(metadata: Header, record: EvidenceRecord | null): Readout[] {
  const values = metadata['evidence-readout'];
  if (values === undefined) return [];
  if (!Array.isArray(values)) throw new Error('Evidence readouts must be a list.');
  return values.map((selection: unknown) => {
    if (
      !object(selection) ||
      typeof selection.label !== 'string' ||
      !Array.isArray(selection.metric) ||
      !selection.metric.length ||
      selection.metric.some((part) => typeof part !== 'string' && typeof part !== 'number') ||
      typeof selection.format !== 'string' ||
      !['number', 'percent', 'error', 'auc'].includes(selection.format)
    ) {
      throw new Error('Evidence readouts require a label, metric path, and supported format.');
    }
    return {
      label: selection.label,
      metric: selection.metric as (string | number)[],
      format: selection.format,
      value: metricAt(record?.metrics, selection.metric as (string | number)[]),
    };
  });
}

function selectedSource(
  entry: Declaration,
  record: EvidenceRecord | null,
  revision: string,
): { source: string; matches: boolean; sha256: string } {
  if (!record) return { source: entry.source, matches: true, sha256: digest(entry.source) };
  const expected = record.provenance.script_sha256;
  if (typeof expected !== 'string' || !/^[a-f\d]{64}$/.test(expected))
    return { source: '', matches: false, sha256: '' };
  for (const source of [
    entry.source,
    frozen.find((candidate) => candidate.id === entry.id)?.source,
  ]) {
    if (
      typeof source === 'string' &&
      digest(source) === expected &&
      identify(source, header(source)) === entry.id
    )
      return { source, matches: true, sha256: expected };
  }
  const plotSource = join(root, 'docs/assets/evidence-plots', `${entry.id}-proof.py`);
  if (existsSync(plotSource)) {
    const source = readFileSync(plotSource, 'utf8');
    if (digest(source) === expected && identify(source, header(source)) === entry.id)
      return { source, matches: true, sha256: expected };
  }
  if (/^[a-f\d]{40}$/.test(revision) && !entry.sourcePath.includes('..')) {
    const source = git(['show', `${revision}:${entry.sourcePath}`]);
    if (
      source !== null &&
      digest(source) === expected &&
      identify(source, header(source)) === entry.id
    )
      return { source, matches: true, sha256: expected };
  }
  return { source: '', matches: false, sha256: expected };
}

export const evidenceEntries: EvidenceEntry[] = declared.map((entry) => {
  const result = publication.results[entry.id] || null;
  const sourceRevision = result
    ? result.provenance.evidence_revision || selection.revision
    : checkoutRevision;
  const retained = frozen.find(
    (candidate) =>
      candidate.id === entry.id && digest(candidate.source) === result?.provenance.script_sha256,
  );
  const sourcePath = retained?.sourcePath || entry.sourcePath;
  const selected = selectedSource(entry, result, sourceRevision);
  const content = selected.matches
    ? presentation(selected.source, sourcePath, sourceRevision)
    : {
        code: '',
        sections: { intro: '', example: '', comparison: '', result: '', method: '' },
        diagram: { caption: '', alt: '' },
      };
  const metadata = selected.matches ? header(selected.source) : entry.header;
  const requirements = entry.header['evidence-requires'] || [];
  if (!Array.isArray(requirements) || requirements.some((value) => typeof value !== 'string'))
    throw new Error(`${entry.id}: resource requirements must be strings.`);
  for (const name of ['title', 'description', 'evidence-group']) {
    if (typeof entry.header[name] !== 'string') throw new Error(`${entry.id}: missing ${name}.`);
  }
  return {
    id: entry.id,
    tags: entry.tags,
    title: entry.header.title as string,
    description: entry.header.description as string,
    family: entry.header['evidence-group'] as string,
    lane: entry.lane,
    requirements: requirements as string[],
    sourcePath,
    sourceRevision,
    sourceUrl: `https://github.com/relflow/relflow/blob/${sourceRevision}/${sourcePath}`,
    sourceSha256: selected.sha256,
    source: selected.source,
    ...content,
    readouts: selected.matches ? readouts(metadata, result) : [],
    result,
    sourceMatches: selected.matches,
    archived: entry.archived === true,
  };
});
