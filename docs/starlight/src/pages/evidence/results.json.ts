import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const raw = readFileSync(resolve(process.cwd(), '../../evidence/results.json'), 'utf8');

export function GET() {
  return new Response(raw, {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
