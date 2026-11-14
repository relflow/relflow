import assert from 'node:assert/strict';
import test from 'node:test';
import { scopeHTML } from '../src/site/build.mjs';
import { sitePath } from '../src/site/urls.mjs';

test('version routes preserve queries, fragments, external URLs, and already scoped paths', () => {
  const base = '/en/v0.5.0/';
  for (const [source, expected] of [
    ['/', '/en/v0.5.0/'],
    ['/concepts/models/?case=claims#example', '/en/v0.5.0/concepts/models/?case=claims#example'],
    ['/en/v0.5.0/evidence/selected.json', '/en/v0.5.0/evidence/selected.json'],
    ['//cdn.example.invalid/file', '//cdn.example.invalid/file'],
    ['https://example.invalid/', 'https://example.invalid/'],
    ['#selection', '#selection'],
    ['?case=tickets', '?case=tickets'],
  ])
    assert.equal(sitePath(source, base), expected);
});

test('hosted HTML rewrites actual attributes, including no-JS and template content, without touching code', () => {
  const source = `<!doctype html><html><head><script>const example = '<a href="/unchanged/">';</script></head><body>
<a href='/quickstart/?a=1&amp;b=2#train'>Guide</a><img src="/_astro/logo.svg">
<noscript><a href="/data-types/number/">Number</a></noscript>
<template><a href="/evidence/">Proofs</a></template>
<pre><code>&lt;a href="/literal/"&gt;</code></pre>
<a href="/en/latest/concepts/models/">Already scoped</a></body></html>`;
  const result = scopeHTML(source, '/en/latest/');
  assert.match(result, /href="\/en\/latest\/quickstart\/\?a=1&amp;b=2#train"/);
  assert.match(result, /src="\/en\/latest\/_astro\/logo.svg"/);
  assert.match(result, /<noscript><a href="\/en\/latest\/data-types\/number\/">/);
  assert.match(result, /<template><a href="\/en\/latest\/evidence\/">/);
  assert.ok(result.includes(`const example = '<a href="/unchanged/">';`));
  assert.ok(result.includes('&lt;a href="/literal/"&gt;'));
  assert.ok(result.includes('href="/en/latest/concepts/models/"'));
  assert.equal(scopeHTML(source, '/'), source);
});
