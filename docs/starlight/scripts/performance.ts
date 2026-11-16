import { mkdirSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';
import { gzipSync } from 'node:zlib';
import { init, parse as imports } from 'es-module-lexer';
import { parse, type DefaultTreeAdapterTypes } from 'parse5';

await init;

interface PageBudget {
  route: string;
  htmlKiB: number;
  htmlGzipKiB: number;
  elements: number;
}

interface Budgets {
  initialJsGzipKiB: number;
  cssGzipKiB: number;
  javascriptAssetGzipKiB: number;
  pages: PageBudget[];
}

function object(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

/** Invalid policy must fail closed rather than turn a comparison into actual > NaN. */
export function validate(value: unknown): asserts value is Budgets {
  if (!object(value)) throw new Error('Performance budgets must be an object.');
  for (const name of ['initialJsGzipKiB', 'cssGzipKiB', 'javascriptAssetGzipKiB']) {
    if (typeof value[name] !== 'number' || !Number.isFinite(value[name]) || value[name] < 0)
      throw new Error(`Performance budget ${name} must be a finite nonnegative number.`);
  }
  if (!Array.isArray(value.pages) || !value.pages.length)
    throw new Error('Performance budgets require at least one page.');
  const routes = new Set<string>();
  for (const page of value.pages) {
    if (
      !object(page) ||
      typeof page.route !== 'string' ||
      !page.route.startsWith('/') ||
      page.route.startsWith('//') ||
      !page.route.endsWith('/') ||
      /[?#]/.test(page.route)
    )
      throw new Error(
        'Performance page budgets require local directory routes such as /installation/.',
      );
    if (routes.has(page.route))
      throw new Error(`Performance page budget ${page.route} is duplicated.`);
    routes.add(page.route);
    for (const name of ['htmlKiB', 'htmlGzipKiB', 'elements']) {
      if (
        typeof page[name] !== 'number' ||
        !Number.isFinite(page[name]) ||
        page[name] < 0 ||
        (name === 'elements' && !Number.isInteger(page[name]))
      )
        throw new Error(
          `Performance page budget ${page.route}.${name} must be a finite nonnegative ${name === 'elements' ? 'integer' : 'number'}.`,
        );
    }
  }
}

export interface PageSize {
  route: string;
  htmlBytes: number;
  htmlGzipBytes: number;
  elements: number;
  initialJsGzipBytes: number;
  cssGzipBytes: number;
  scripts: string[];
  styles: string[];
}

/** Inspect static output without executing its scripts or downloading its resources. */
export function measure(directory: string, route: string): PageSize {
  const origin = 'https://docs.invalid';
  const root = resolve(directory);
  const address = new URL(route, origin);
  const file = resolve(
    root,
    `.${address.pathname}${address.pathname.endsWith('/') ? 'index.html' : ''}`,
  );
  if (address.origin !== origin || !file.startsWith(`${root}${sep}`))
    throw new Error(`Performance page must stay inside the build directory: ${route}`);
  const html = readFileSync(file, 'utf8');
  const document = parse(html);
  const scripts = new Set<string>();
  const styles = new Set<string>();
  let elements = 0;

  function asset(specifier: string, parent: URL): { file: string; url: URL } {
    const url = new URL(specifier, parent);
    if (url.origin !== origin)
      throw new Error(
        `Performance budgets require local script/style assets; cannot measure ${url.href}.`,
      );
    const file = resolve(root, `.${decodeURIComponent(url.pathname)}`);
    if (!file.startsWith(`${root}${sep}`))
      throw new Error(`Bundle asset escapes the build directory: ${specifier}`);
    return { file, url };
  }

  function dependencies(source: string, parent: URL) {
    for (const entry of imports(source)[0]) {
      if (entry.d !== -1 || !entry.n) continue;
      const imported = asset(entry.n, parent);
      if (scripts.has(imported.file)) continue;
      if (imported.file.endsWith('.css')) {
        styles.add(imported.file);
        continue;
      }
      scripts.add(imported.file);
      dependencies(readFileSync(imported.file, 'utf8'), imported.url);
    }
  }

  function visit(node: DefaultTreeAdapterTypes.Node) {
    if ('tagName' in node) {
      elements++;
      const attributes = Object.fromEntries(node.attrs.map(({ name, value }) => [name, value]));
      if (
        node.tagName === 'script' &&
        (!attributes.type ||
          ['module', 'text/javascript', 'application/javascript'].includes(attributes.type))
      ) {
        if (attributes.src) {
          const script = asset(attributes.src, address);
          if (!scripts.has(script.file)) {
            scripts.add(script.file);
            dependencies(readFileSync(script.file, 'utf8'), script.url);
          }
        } else {
          const source = node.childNodes
            .filter((child) => child.nodeName === '#text')
            .map((child) => (child as DefaultTreeAdapterTypes.TextNode).value)
            .join('');
          dependencies(source, address);
        }
      }
      if (node.tagName === 'link' && attributes.rel?.split(/\s+/).includes('stylesheet')) {
        const stylesheet = asset(attributes.href, address);
        styles.add(stylesheet.file);
      }
      // Count template contents too: inert fragments still consume parsing and memory.
      if (node.tagName === 'template' && 'content' in node) visit(node.content);
    }
    if ('childNodes' in node) for (const child of node.childNodes) visit(child);
  }

  visit(document);
  const bytes = (files: Set<string>) =>
    [...files].reduce((sum, file) => sum + gzipSync(readFileSync(file), { level: 9 }).length, 0);
  return {
    route,
    htmlBytes: Buffer.byteLength(html),
    htmlGzipBytes: gzipSync(html, { level: 9 }).length,
    elements,
    initialJsGzipBytes: bytes(scripts),
    cssGzipBytes: bytes(styles),
    scripts: [...scripts].map((file) => relative(root, file)),
    styles: [...styles].map((file) => relative(root, file)),
  };
}

export function violations(
  page: PageSize,
  budget: PageBudget,
  limits: Pick<Budgets, 'initialJsGzipKiB' | 'cssGzipKiB'>,
): string[] {
  const checks: [string, number, number][] = [
    ['HTML bytes', page.htmlBytes, budget.htmlKiB * 1024],
    ['gzip HTML bytes', page.htmlGzipBytes, budget.htmlGzipKiB * 1024],
    ['HTML elements', page.elements, budget.elements],
    ['initial gzip JavaScript bytes', page.initialJsGzipBytes, limits.initialJsGzipKiB * 1024],
    ['gzip stylesheet bytes', page.cssGzipBytes, limits.cssGzipKiB * 1024],
  ];
  return checks
    .filter(([, actual, limit]) => actual > limit)
    .map(([name, actual, limit]) => `${page.route}: ${name} ${actual} exceeds budget ${limit}.`);
}

function run() {
  const site = fileURLToPath(new URL('../', import.meta.url));
  const root = resolve(site, 'dist');
  const budgets: unknown = JSON.parse(readFileSync(resolve(site, 'performance.json'), 'utf8'));
  validate(budgets);
  const pages = budgets.pages.map((budget) => measure(root, budget.route));
  const failed = pages.flatMap((page, index) => violations(page, budgets.pages[index], budgets));
  for (const file of readdirSync(resolve(root, '_astro')).filter((file) => file.endsWith('.js'))) {
    const size = gzipSync(readFileSync(resolve(root, '_astro', file)), { level: 9 }).length;
    if (size > budgets.javascriptAssetGzipKiB * 1024)
      failed.push(
        `${file}: gzip JavaScript bytes ${size} exceeds asset budget ${budgets.javascriptAssetGzipKiB * 1024}.`,
      );
  }
  const report = resolve(site, 'performance-report/budgets.json');
  mkdirSync(dirname(report), { recursive: true });
  writeFileSync(
    report,
    JSON.stringify(
      {
        method:
          'Build-time gzip level 9; static import closure excludes dynamic/idle imports; inline scripts count in HTML; template elements count toward DOM.',
        pages,
        failures: failed,
      },
      null,
      2,
    ),
  );
  console.table(
    pages.map((page) => ({
      route: page.route,
      'HTML KiB': (page.htmlBytes / 1024).toFixed(1),
      'gzip HTML KiB': (page.htmlGzipBytes / 1024).toFixed(1),
      elements: page.elements,
      'initial gzip JS KiB': (page.initialJsGzipBytes / 1024).toFixed(1),
      'gzip CSS KiB': (page.cssGzipBytes / 1024).toFixed(1),
    })),
  );
  if (failed.length)
    throw new Error(
      `Documentation performance budgets failed:\n${failed.join('\n')}\nInspect ${report}; optimize the page or deliberately revise its documented budget.`,
    );
  console.log(`Performance budgets passed for ${pages.length} representative pages.`);
}

if (process.argv[1] && import.meta.url === pathToFileURL(resolve(process.argv[1])).href) run();
