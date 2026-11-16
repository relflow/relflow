import type { APIContext } from 'astro';
import { evidenceEntries } from '../../../evidence/data';

export function getStaticPaths() {
  return evidenceEntries
    .filter((entry) => entry.sourceMatches && entry.source)
    .map((entry) => ({ params: { id: entry.id }, props: { source: entry.source } }));
}

export function GET({ props }: APIContext) {
  return new Response(props.source, { headers: { 'Content-Type': 'text/plain; charset=utf-8' } });
}
