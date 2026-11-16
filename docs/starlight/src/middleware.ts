import { defineMiddleware } from 'astro:middleware';
import { brandHTML } from './site/brand.mjs';

export const onRequest = defineMiddleware(async (context, next) => {
  const response = await next();
  if (
    context.request.method !== 'GET' ||
    response.status !== 200 ||
    !response.body ||
    !response.headers.get('content-type')?.includes('text/html')
  )
    return response;

  const html = brandHTML(await response.text());
  const headers = new Headers(response.headers);
  headers.delete('content-length');
  headers.delete('etag');
  return new Response(html, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
});
