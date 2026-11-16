import { test, expect, type Locator } from '@playwright/test';

declare global {
  interface Window {
    docsLayoutShift: number;
    docsTriggerListeners: { target: Element; type: string }[];
    docsDiagramObservations: Element[];
    docsObservedDiagrams: Set<Element>;
  }
}

async function geometry(root: Locator) {
  return root.evaluate((element) => {
    const picker = element.querySelector('.case-picker, .example-picker')!;
    const panel = [
      ...element.querySelectorAll<HTMLElement>('[data-case], [data-type-example]'),
    ].find((panel) => panel.getClientRects().length)!;
    const box = (node: Element) => {
      const rect = node.getBoundingClientRect();
      return { x: rect.x + scrollX, y: rect.y + scrollY, width: rect.width, height: rect.height };
    };
    return { picker: box(picker), panel: box(panel), panelId: panel.id };
  });
}

for (const item of [
  { route: '/?case=claims', root: '[data-use-cases]', panel: 'case-claims' },
  { route: '/concepts/models/?case=churn', root: '[data-use-cases]', panel: 'models-churn' },
  {
    route: '/featureless/?case=maintenance',
    root: '[data-use-cases]',
    panel: 'featureless-maintenance',
  },
]) {
  test(`initial selection and geometry survive delayed enhancement: ${item.route}`, async ({
    page,
  }) => {
    let release!: () => void;
    const gate = new Promise<void>((resolve) => {
      release = resolve;
    });
    await page.route('**/_astro/*.js', async (route) => {
      await gate;
      await route.continue();
    });
    await page.addInitScript(() => {
      window.docsLayoutShift = 0;
      new PerformanceObserver((list) => {
        for (const entry of list.getEntries()) {
          if (
            'hadRecentInput' in entry &&
            entry.hadRecentInput === false &&
            'value' in entry &&
            typeof entry.value === 'number'
          )
            window.docsLayoutShift += entry.value;
        }
      }).observe({ type: 'layout-shift', buffered: true });
    });
    try {
      await page.goto(item.route, { waitUntil: 'commit' });
      await page.waitForFunction(
        () =>
          document.readyState !== 'loading' &&
          [...document.querySelectorAll<HTMLLinkElement>('link[rel="stylesheet"]')].every(
            (link) => link.sheet,
          ),
      );
      const root = page.locator(item.root);
      await expect(root).toHaveAttribute('data-prepared', '');
      await expect(root).not.toHaveAttribute('data-ready');
      const before = await geometry(root);
      expect(before.panelId).toBe(item.panel);
      expect(before.picker.height).toBeGreaterThan(50);
      await expect(root.locator('[data-case]:visible, [data-type-example]:visible')).toHaveCount(1);
      // Give the browser a painted frame before allowing any enhancement modules.
      await page.evaluate(
        () =>
          new Promise<void>((resolve) =>
            requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
          ),
      );
      release();
      await page.waitForLoadState('networkidle');
      await expect(root).toHaveAttribute('data-ready', '');
      const after = await geometry(root);
      expect(after.panelId).toBe(item.panel);
      for (const dimension of ['x', 'y', 'width', 'height'] as const)
        expect(
          Math.abs(after.picker[dimension] - before.picker[dimension]),
          `picker.${dimension} changed during enhancement`,
        ).toBeLessThanOrEqual(1);
      for (const dimension of ['x', 'y', 'width'] as const)
        expect(
          Math.abs(after.panel[dimension] - before.panel[dimension]),
          `panel.${dimension} changed during enhancement`,
        ).toBeLessThanOrEqual(1);
      expect(await page.evaluate(() => window.docsLayoutShift)).toBeLessThan(0.01);
    } finally {
      release();
    }
  });
}

test('inactive cases have no per-trigger listeners or initialized diagrams', async ({ page }) => {
  await page.addInitScript(() => {
    window.docsTriggerListeners = [];
    window.docsDiagramObservations = [];
    window.docsObservedDiagrams = new Set();
    const Resize = window.ResizeObserver;
    window.ResizeObserver = class extends Resize {
      private diagrams = new Set<Element>();
      observe(target: Element, options?: ResizeObserverOptions) {
        if (target.matches('.erd-canvas, .erd-scroll')) {
          window.docsDiagramObservations.push(target);
          window.docsObservedDiagrams.add(target);
          this.diagrams.add(target);
        }
        super.observe(target, options);
      }
      unobserve(target: Element) {
        this.diagrams.delete(target);
        window.docsObservedDiagrams.delete(target);
        super.unobserve(target);
      }
      disconnect() {
        for (const target of this.diagrams) window.docsObservedDiagrams.delete(target);
        this.diagrams.clear();
        super.disconnect();
      }
    };
    const original = EventTarget.prototype.addEventListener;
    EventTarget.prototype.addEventListener = function (type, callback, options) {
      if (this instanceof Element && this.matches('.rf-code-note, .erd-column'))
        window.docsTriggerListeners.push({ target: this, type });
      return original.call(this, type, callback, options);
    };
  });
  await page.goto('/concepts/models/?case=churn');
  const diagram = page.locator('[data-case="churn"] .erd-data [data-erd]');
  await diagram.scrollIntoViewIfNeeded();
  await expect(diagram).toHaveAttribute('data-erd-ready', '');
  await expect(page.locator('[data-case]:not([data-case="churn"]) [data-erd-ready]')).toHaveCount(
    0,
  );
  expect(await page.evaluate(() => window.docsTriggerListeners.length)).toBe(0);
  expect(
    await page.evaluate(() =>
      window.docsDiagramObservations.every(
        (target) => target.closest('[data-case]')?.getAttribute('data-case') === 'churn',
      ),
    ),
  ).toBe(true);
  await page.locator('[data-case-select="claims"]').click();
  const next = page.locator('[data-case="claims"] .erd-data [data-erd]');
  await next.scrollIntoViewIfNeeded();
  await expect(next).toHaveAttribute('data-erd-ready', '');
  expect(await page.evaluate(() => window.docsTriggerListeners.length)).toBe(0);
  await expect
    .poll(() =>
      page.evaluate(
        () =>
          window.docsObservedDiagrams.size === 2 &&
          [...window.docsObservedDiagrams].every(
            (target) => target.closest('[data-case]')?.getAttribute('data-case') === 'claims',
          ),
      ),
    )
    .toBe(true);
});

test('compact Evidence JSON remains complete, copyable, and readable without JavaScript', async ({
  page,
  context,
  browser,
  baseURL,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const publication = await context.request.get('/evidence/selected.json');
  expect(publication.ok()).toBe(true);
  const selected = await publication.json();
  const expected = selected.results.P047;
  const response = await context.request.get('/evidence/records/P047.json');
  expect(response.status()).toBe(expected ? 200 : 404);
  await page.goto('/evidence/P047/');
  if (expected) {
    expect(await response.json()).toEqual(expected);
    await page.locator('.report-details > summary').click();
    await page.locator('.raw-record > summary').click();
    const record = page.locator('[data-json-record]');
    expect(JSON.parse((await record.locator('code').textContent())!)).toEqual(expected);
    expect(await record.locator('*').count()).toBeLessThan(12);
    await record.getByRole('button', { name: 'Copy JSON', exact: true }).click();
    expect(JSON.parse(await page.evaluate(() => navigator.clipboard.readText()))).toEqual(expected);
  } else {
    await expect(page.locator('.run-status')).toHaveText('No selected run');
    await expect(page.locator('[data-json-record]')).toHaveCount(0);
  }
  const withoutScripts = await browser.newContext({ javaScriptEnabled: false });
  try {
    const fallback = await withoutScripts.newPage();
    await fallback.goto(`${baseURL}/evidence/P047/`);
    if (!expected) {
      await expect(fallback.locator('.run-status')).toHaveText('No selected run');
      await expect(fallback.locator('[data-json-record]')).toHaveCount(0);
      return;
    }
    await fallback.locator('.report-details > summary').click();
    await fallback.locator('.raw-record > summary').click();
    await expect(fallback.locator('[data-json-record] pre')).toBeVisible();
    expect(JSON.parse((await fallback.locator('[data-json-record] code').textContent())!)).toEqual(
      expected,
    );
    await expect(fallback.locator('[data-json-copy]')).toBeHidden();
  } finally {
    await withoutScripts.close();
  }
});
