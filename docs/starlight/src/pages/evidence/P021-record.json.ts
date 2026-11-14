import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const { schema, version, revision, report, results } = JSON.parse(
  readFileSync(resolve(process.cwd(), '../../evidence/results.json'), 'utf8'),
);
const raw =
  JSON.stringify({ schema, version, revision, report, results: { P021: results.P021 } }, null, 2) +
  '\n';

export function GET() {
  return new Response(raw, {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
