import assert from 'node:assert/strict';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { gzipSync } from 'node:zlib';
import test from 'node:test';
import { measure, validate, violations } from '../scripts/performance.ts';

function fixture(t, html) {
  const root = mkdtempSync(join(tmpdir(), 'relflow-docs-budget-'));
  t.after(() => rmSync(root, { recursive: true, force: true }));
  mkdirSync(join(root, '_astro'));
  writeFileSync(join(root, 'index.html'), html);
  return root;
}

test('page budgets deduplicate scripts/styles and follow static imports and reexports only', (t) => {
  const root = fixture(
    t,
    `<!doctype html><html><head>
    <link rel="stylesheet" href="/_astro/site.css"><link rel="stylesheet" href="/_astro/site.css">
    </head><body><script type="module" src="/_astro/main.js"></script>
    <script type="module" src="/_astro/main.js"></script>
    <script type="module">import '/_astro/exports.js';</script>
    <script type="application/json">{"example":"import '/missing-data.js'"}</script></body></html>`,
  );
  const files = {
    'main.js': `import './shared.js'; export { value } from './exports.js'; import('./lazy.js'); const text = "import './missing-string.js'";`,
    'shared.js': `import './main.js'; export const shared = 1;`,
    'exports.js': 'export const value = 7;',
    'lazy.js': 'export const later = 0;',
    'site.css': 'body { color: green; }',
  };
  for (const [name, source] of Object.entries(files))
    writeFileSync(join(root, '_astro', name), source);
  const result = measure(root, '/');
  assert.deepEqual(result.scripts.sort(), [
    '_astro/exports.js',
    '_astro/main.js',
    '_astro/shared.js',
  ]);
  assert.deepEqual(result.styles, ['_astro/site.css']);
  assert.equal(
    result.initialJsGzipBytes,
    ['main.js', 'shared.js', 'exports.js'].reduce(
      (sum, file) => sum + gzipSync(files[file], { level: 9 }).length,
      0,
    ),
  );
  assert.equal(result.cssGzipBytes, gzipSync(files['site.css'], { level: 9 }).length);
});

test('inert template contents count toward markup budgets rather than hiding their cost', (t) => {
  const root = fixture(t, '<!doctype html><html><head></head><body><p>Read me</p></body></html>');
  const before = measure(root, '/');
  writeFileSync(
    join(root, 'index.html'),
    '<!doctype html><html><head></head><body><p>Read me</p><template><section><span>Deferred</span></section></template></body></html>',
  );
  const after = measure(root, '/');
  assert.equal(after.elements, before.elements + 3);
  assert.ok(after.htmlBytes > before.htmlBytes);
});

for (const base of ['/latest/', '/v0.5.0/', '/previews/102/'])
  test(`hosted ${base} pages resolve their asset closure inside the build directory`, (t) => {
    const html = `<!doctype html><html><head>
      <script type="module" src="${base}_astro/main.js"></script>
      <link rel="stylesheet" href="${base}_astro/site.css">
      <link rel="canonical" href="https://www.relflow.ai${base}">
      </head><body>Hosted page</body></html>`;
    const root = fixture(t, html);
    const files = {
      'main.js': "import './shared.js'; import './site.css';",
      'shared.js': 'export const value = 7;',
      'site.css': 'body { color: green; }',
    };
    for (const [name, source] of Object.entries(files))
      writeFileSync(join(root, '_astro', name), source);
    mkdirSync(join(root, 'guide'));
    writeFileSync(
      join(root, 'guide/index.html'),
      html.replace(`https://www.relflow.ai${base}"`, `https://www.relflow.ai${base}guide/"`),
    );
    for (const route of ['/', '/guide/']) {
      const result = measure(root, route);
      assert.deepEqual(result.scripts.sort(), ['_astro/main.js', '_astro/shared.js']);
      assert.deepEqual(result.styles, ['_astro/site.css']);
      assert.equal(
        result.initialJsGzipBytes,
        ['main.js', 'shared.js'].reduce(
          (sum, file) => sum + gzipSync(files[file], { level: 9 }).length,
          0,
        ),
      );
      assert.equal(result.cssGzipBytes, gzipSync(files['site.css'], { level: 9 }).length);
    }
  });

test('hosted bundles reject assets outside their version and still require every referenced file', (t) => {
  for (const [asset, error] of [
    ['/_astro/main.js', /documentation version/],
    ['/other/_astro/main.js', /documentation version/],
    ['https://cdn.invalid/main.js', /require local script\/style assets/],
    ['/latest/_astro/missing.js', /missing\.js/],
  ]) {
    const root = fixture(
      t,
      `<link rel="canonical" href="https://www.relflow.ai/latest/"><script type="module" src="${asset}"></script>`,
    );
    writeFileSync(join(root, '_astro/main.js'), 'export const value = 7;');
    assert.throws(() => measure(root, '/'), error);
  }
});

test('missing referenced assets fail the budget check instead of understating the bundle', (t) => {
  const root = fixture(t, '<script type="module" src="/_astro/missing.js"></script>');
  assert.throws(() => measure(root, '/'), /missing\.js/);
});

test('remote scripts cannot bypass the local bundle budget', (t) => {
  const root = fixture(t, '<script src="https://cdn.invalid/unmeasured.js"></script>');
  assert.throws(() => measure(root, '/'), /require local script\/style assets.*cdn\.invalid/);
});

test('every enforced page dimension reports a clear failure when its budget is exceeded', () => {
  const page = {
    route: '/fixture/',
    htmlBytes: 10,
    htmlGzipBytes: 10,
    elements: 10,
    initialJsGzipBytes: 10,
    cssGzipBytes: 10,
    scripts: [],
    styles: [],
  };
  const budget = { route: page.route, htmlKiB: 0, htmlGzipKiB: 0, elements: 0 };
  const limits = { initialJsGzipKiB: 0, cssGzipKiB: 0 };
  assert.equal(violations(page, budget, limits).length, 5);
  for (const failure of violations(page, budget, limits))
    assert.match(failure, /\/fixture\/: .+ 10 exceeds budget 0\./);
  assert.deepEqual(
    violations(
      page,
      { ...budget, htmlKiB: 1, htmlGzipKiB: 1, elements: 10 },
      { initialJsGzipKiB: 1, cssGzipKiB: 1 },
    ),
    [],
  );
});

test('missing, renamed, nonnumeric, nonfinite, and empty budget policies cannot silently pass', () => {
  const policy = {
    initialJsGzipKiB: 12,
    cssGzipKiB: 32,
    javascriptAssetGzipKiB: 32,
    pages: [{ route: '/', htmlKiB: 180, htmlGzipKiB: 20, elements: 2100 }],
  };
  assert.doesNotThrow(() => validate(policy));
  for (const invalid of [
    null,
    { ...policy, initialJsGzipKiB: undefined, renamedLimit: 12 },
    { ...policy, cssGzipKiB: '32' },
    { ...policy, javascriptAssetGzipKiB: NaN },
    { ...policy, cssGzipKiB: Infinity },
    { ...policy, pages: [] },
    { ...policy, pages: [policy.pages[0], policy.pages[0]] },
    { ...policy, pages: [{ ...policy.pages[0], htmlGzipKiB: undefined }] },
    { ...policy, pages: [{ ...policy.pages[0], elements: 1.5 }] },
    { ...policy, pages: [{ ...policy.pages[0], route: '//other.invalid/' }] },
  ])
    assert.throws(() => validate(invalid), /Performance/);
});
