import type { APIContext } from 'astro';
import { evidenceEntries } from '../../../evidence/data';

export function getStaticPaths() {
  return evidenceEntries
    .filter((entry) => entry.result)
    .map((entry) => ({ params: { id: entry.id }, props: { record: entry.result } }));
}

export function GET({ props }: APIContext) {
  return new Response(JSON.stringify(props.record, null, 2) + '\n', {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
