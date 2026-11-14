import { selection } from '../../evidence/data';

export function GET() {
  return new Response(selection.raw, {
    headers: { 'Content-Type': 'application/json; charset=utf-8' },
  });
}
