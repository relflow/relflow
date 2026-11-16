import { execFileSync } from 'node:child_process';
import { existsSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { apiCatalog } from '../data/api-map';

interface Location {
  file: string;
  line: number;
}

interface DocumentedType {
  type: string;
  description: string;
}

interface Parameter extends DocumentedType {
  name: string;
  default: string | null;
  kind: string;
}

export interface Signature {
  name: string;
  code: string;
  text: string;
  parameters: Parameter[];
  returns: DocumentedType[];
  raises: DocumentedType[];
  source: Location;
}

interface Field extends DocumentedType {
  name: string;
  default: string | null;
  factory: boolean;
  property: boolean;
  inherited: boolean;
  source: Location;
}

interface Definition {
  name: string;
  kind: string;
  text: string;
  definition: string;
  signatures: Signature[];
  overloads: Signature[];
  fields: Field[];
  methods: Signature[];
  external: string;
  source: Location;
  documentationSource?: Location;
  canonicalPath?: string;
  aliasOf?: string;
}

const root = resolve(process.cwd(), '../..');
const python =
  process.env.RELFLOW_DOCS_PYTHON ||
  join(root, 'docs/.venv', process.platform === 'win32' ? 'Scripts/python.exe' : 'bin/python');
if (!existsSync(python))
  throw new Error(
    'API reference extraction needs the isolated docs environment. Run make prepare.',
  );

export const apiRevision =
  process.env.RELFLOW_DOCS_REVISION ||
  execFileSync('git', ['rev-parse', 'HEAD'], { cwd: root, encoding: 'utf8' }).trim();
if (!/^[a-f0-9]{40}$/.test(apiRevision))
  throw new Error('API source revision must be a full Git SHA.');

export function sourceURL(source: Location): string {
  return `https://github.com/relflow/relflow/blob/${apiRevision}/${source.file}#L${source.line}`;
}

export function apiSlug(name: string): string {
  return name
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/-$/, '');
}

export function apiPath(name: string): string {
  return `/api/${apiSlug(name)}/`;
}

const definitions: Definition[] = JSON.parse(
  execFileSync(python, [join(root, 'docs/api.py')], {
    cwd: root,
    input: JSON.stringify(apiCatalog.flatMap(({ entries }) => entries.map(({ name }) => name))),
    encoding: 'utf8',
    maxBuffer: 16 * 1024 * 1024,
  }),
);
const byName = new Map(definitions.map((definition) => [definition.name, definition]));

export const apiSections = apiCatalog.map((section) => ({
  ...section,
  entries: section.entries.map((entry) => {
    const definition = byName.get(entry.name);
    if (!definition) throw new Error(`API extractor omitted ${entry.name}.`);
    const description =
      definition.text.split('\n\n')[0].replace(/`+/g, '').replace(/\s+/g, ' ').trim() ||
      (definition.external
        ? `Re-exported from ${definition.external}.`
        : definition.definition ||
          `${entry.name}: ${definition.fields.map((field) => field.name).join(', ')}.`);
    return {
      ...entry,
      ...definition,
      description:
        description.length > 240 ? `${description.slice(0, 237).trimEnd()}…` : description,
      sourceUrl: sourceURL(definition.source),
      href: apiPath(entry.name),
      sectionId: section.id,
      sectionTitle: section.title,
    };
  }),
}));

export const apiEntries = apiSections.flatMap(({ entries }) => entries);
export type ApiReference = (typeof apiEntries)[number];
export const apiSources = {
  schema: `https://github.com/relflow/relflow/tree/${apiRevision}/src/relflow`,
  fields: `https://github.com/relflow/relflow/tree/${apiRevision}/src/relflow/tensorfields`,
};
