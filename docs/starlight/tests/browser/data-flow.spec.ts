import { test as base, expect, type Locator } from '@playwright/test';
import { flowPaths, type FlowMode, type FlowStageId } from '../../src/data-flow/types';
import { predictions } from '../../src/data/concepts/predictions';

declare global {
  interface Window {
    docsFlowLayoutShift: number;
  }
}

const test = base.extend({
  page: async ({ page }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await use(page);
    expect(errors, 'The data flow must not throw browser errors.').toEqual([]);
  },
});

const caseIds = predictions.examples.map((example) => example.caseId);

function stage(root: Locator, caseId: string, id: FlowStageId) {
  return root.locator(`details#flow-${caseId}-${id}[data-flow-stage]`);
}

async function phase(root: Locator, caseId: string, mode: FlowMode, id: FlowStageId) {
  await expect(root).toHaveAttribute('data-flow-case', caseId);
  await expect(root).toHaveAttribute('data-flow-mode', mode);
  await expect(root).toHaveAttribute('data-flow-stage', id);
  await expect(stage(root, caseId, id)).toHaveAttribute('data-flow-active', '');
  await expect(stage(root, caseId, id)).toBeVisible();
  await expect(root.locator('details[data-flow-stage]:visible')).toHaveCount(1);
  await expect(root.locator(`[data-case="${caseId}"] [data-flow-select="${id}"]`)).toHaveAttribute(
    'aria-current',
    'step',
  );
}

test('all six cases expose seven ordered stages per path and navigate forward, backward, and into another batch', async ({
  page,
}) => {
  test.setTimeout(90_000);
  await page.goto('/concepts/data-flow/?case=tickets&mode=predict#flow-tickets-load');
  const root = page.locator('[data-data-flow]');
  await expect(root).toHaveAttribute('data-flow-ready', '');
  expect(caseIds).toHaveLength(6);
  await expect(root.locator('[data-case-select]')).toHaveCount(6);

  for (const caseId of caseIds) {
    await root.locator(`[data-case-select="${caseId}"]`).click();
    for (const mode of ['predict', 'train'] as const) {
      const path = flowPaths[mode];
      await root.locator(`[data-flow-mode-select="${mode}"]`).click();
      await phase(root, caseId, mode, 'load');
      const steps = root.locator(`[data-case="${caseId}"] [data-flow-select]:visible`);
      await expect(steps).toHaveCount(7);
      expect(
        await steps.evaluateAll((nodes) =>
          nodes.map((node) => node.getAttribute('data-flow-select')),
        ),
      ).toEqual([...path]);
      await expect(root.locator(`[data-flow-mode-select="${mode}"]`)).toHaveAttribute(
        'aria-pressed',
        'true',
      );
      await expect(stage(root, caseId, 'load').locator('[data-flow-previous]')).toBeDisabled();

      for (let index = 1; index < path.length; index++) {
        const next = stage(root, caseId, path[index - 1]).locator('[data-flow-next]');
        await expect(next).toHaveAttribute('aria-controls', `flow-${caseId}-${path[index]}`);
        await next.click();
        await phase(root, caseId, mode, path[index]);
        await expect(stage(root, caseId, path[index])).toHaveJSProperty('open', true);
        await expect(root.locator('details[data-flow-stage][open]')).toHaveCount(1);
      }

      const last = path.at(-1)!;
      const previous = stage(root, caseId, last).locator('[data-flow-previous]');
      await expect(previous).toHaveAttribute('aria-controls', `flow-${caseId}-${path.at(-2)}`);
      await previous.click();
      await phase(root, caseId, mode, path.at(-2)!);
      await stage(root, caseId, path.at(-2)!).locator('[data-flow-next]').click();
      await phase(root, caseId, mode, last);
      const restart = stage(root, caseId, last).locator('[data-flow-next]');
      if (mode === 'train') await expect(restart).toHaveText('Next batch');
      await expect(restart).toHaveAttribute('aria-controls', `flow-${caseId}-load`);
      await restart.click();
      await phase(root, caseId, mode, 'load');
      await expect(stage(root, caseId, 'load')).toHaveJSProperty('open', true);
    }
  }
});

test('deep links, browser history, mode changes, and case changes preserve the requested flow position', async ({
  page,
}) => {
  await page.goto('/concepts/data-flow/?case=claims&mode=train#flow-claims-loss');
  const root = page.locator('[data-data-flow]');
  await expect(root).toHaveAttribute('data-flow-ready', '');
  await phase(root, 'claims', 'train', 'loss');
  await expect(stage(root, 'claims', 'loss')).toHaveJSProperty('open', true);
  await stage(root, 'claims', 'loss').locator('[data-flow-next]').click();
  await phase(root, 'claims', 'train', 'backward');
  await root.locator('[data-flow-mode-select="predict"]').click();
  await phase(root, 'claims', 'predict', 'postprocess');
  await expect(page).toHaveURL(
    (url) => url.searchParams.get('mode') === 'predict' && url.hash === '#flow-claims-postprocess',
  );

  await page.goBack();
  await phase(root, 'claims', 'train', 'backward');
  await page.goBack();
  await phase(root, 'claims', 'train', 'loss');
  await page.goForward();
  await phase(root, 'claims', 'train', 'backward');
  await page.goForward();
  await phase(root, 'claims', 'predict', 'postprocess');

  await root.locator('[data-case-select="demand"]').click();
  await phase(root, 'demand', 'predict', 'postprocess');
  await expect(page).toHaveURL(
    (url) =>
      url.searchParams.get('case') === 'demand' &&
      url.searchParams.get('mode') === 'predict' &&
      url.hash === '#flow-demand-postprocess',
  );
  await root.locator('[data-flow-mode-select="train"]').click();
  await phase(root, 'demand', 'train', 'backward');
  await root.locator('[data-case-select="claims"]').click();
  await phase(root, 'claims', 'train', 'backward');
  await expect(stage(root, 'claims', 'backward')).toHaveJSProperty('open', true);

  await page.goto('/concepts/data-flow/?case=claims&mode=predict#flow-claims-loss');
  await phase(root, 'claims', 'train', 'loss');
});

test('nested disclosures stay inside their phase and prediction controls retain the complete canonical row', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/concepts/data-flow/?case=claims&mode=predict#flow-claims-model');
  const root = page.locator('[data-data-flow]');
  await expect(root).toHaveAttribute('data-flow-ready', '');
  const model = stage(root, 'claims', 'model');
  await model.locator(':scope > summary').click();
  await expect(model).toHaveJSProperty('open', true);
  await phase(root, 'claims', 'predict', 'model');
  const branch = model.locator('[data-tree] details').first();
  await expect(branch).toHaveJSProperty('open', true);
  await branch.locator(':scope > summary').click();
  await expect(branch).toHaveJSProperty('open', false);
  await phase(root, 'claims', 'predict', 'model');
  await expect(model).toHaveJSProperty('open', true);
  await branch.locator(':scope > summary').click();
  await expect(branch).toHaveJSProperty('open', true);
  const code = model.locator('.df-code');
  await code.locator(':scope > summary').click();
  await expect(code).toHaveJSProperty('open', true);
  await phase(root, 'claims', 'predict', 'model');
  await code.locator(':scope > summary').click();
  await expect(code).toHaveJSProperty('open', false);
  await phase(root, 'claims', 'predict', 'model');

  await root.locator('[data-case="claims"] [data-flow-select="result"]').click();
  await phase(root, 'claims', 'predict', 'result');
  const tree = stage(root, 'claims', 'result').locator('[data-value-tree]');
  await tree.locator('[data-tree-collapse]').click();
  await expect(tree.locator('details[open]')).toHaveCount(0);
  await phase(root, 'claims', 'predict', 'result');
  await tree.locator('[data-tree-expand]').click();
  expect(await tree.locator('details').count()).toBeGreaterThan(0);
  await expect(tree.locator('details:not([open])')).toHaveCount(0);
  await phase(root, 'claims', 'predict', 'result');
  const result = predictions.examples
    .find((example) => example.caseId === 'claims')!
    .blocks.find((block) => block.kind === 'prediction');
  if (!result || result.kind !== 'prediction')
    throw new Error('Claims needs its canonical illustrative prediction row.');
  await tree.locator('[data-tree-copy]').click();
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(JSON.stringify(result.value, null, 2));
  await phase(root, 'claims', 'predict', 'result');
});

test('stage links support direct selection and keyboard navigation along the active path', async ({
  page,
}) => {
  await page.goto('/concepts/data-flow/?case=claims&mode=predict#flow-claims-load');
  const root = page.locator('[data-data-flow]');
  await expect(root).toHaveAttribute('data-flow-ready', '');
  const link = (id: FlowStageId) => root.locator(`[data-case="claims"] [data-flow-select="${id}"]`);

  await link('prepare').click();
  await phase(root, 'claims', 'predict', 'prepare');
  await link('prepare').focus();
  await page.keyboard.press('ArrowDown');
  await phase(root, 'claims', 'predict', 'model');
  await expect(link('model')).toBeFocused();
  await page.keyboard.press('ArrowUp');
  await phase(root, 'claims', 'predict', 'prepare');
  await expect(link('prepare')).toBeFocused();
  await page.keyboard.press('End');
  await phase(root, 'claims', 'predict', 'result');
  await expect(link('result')).toBeFocused();
  await page.keyboard.press('Home');
  await phase(root, 'claims', 'predict', 'load');
  await expect(link('load')).toBeFocused();

  await root.locator('[data-flow-mode-select="train"]').click();
  await link('model').click();
  await link('model').focus();
  await page.keyboard.press('ArrowDown');
  await phase(root, 'claims', 'train', 'loss');
  await expect(link('loss')).toBeFocused();
  await page.keyboard.press('End');
  await phase(root, 'claims', 'train', 'update');
  await expect(link('update')).toBeFocused();
});

test('without JavaScript all six cases and both paths remain readable through native disclosures', async ({
  browser,
  baseURL,
  page,
}) => {
  const context = await browser.newContext({
    javaScriptEnabled: false,
    viewport: page.viewportSize()!,
  });
  const fallback = await context.newPage();
  try {
    await fallback.goto(`${baseURL}/concepts/data-flow/`);
    const root = fallback.locator('[data-data-flow]');
    await expect(root.locator('[data-case]:visible')).toHaveCount(6);
    for (const caseId of caseIds) {
      const panel = root.locator(`[data-case="${caseId}"]`);
      await expect(panel.locator('details[data-flow-stage]:visible')).toHaveCount(10);
      await expect(panel.locator('details[data-flow-stage][open]')).toHaveCount(10);
      for (const id of flowPaths.predict.slice(0, 4))
        await expect(stage(root, caseId, id).locator('summary [data-flow-stage-mode]')).toHaveText(
          'Shared stages',
        );
      for (const mode of ['predict', 'train'] as const)
        for (const id of flowPaths[mode].slice(4))
          await expect(
            stage(root, caseId, id).locator('summary [data-flow-stage-mode]'),
          ).toHaveText(mode === 'train' ? 'Training path' : 'Prediction path');
      expect(
        await panel
          .locator('details[data-flow-stage]:visible')
          .evaluateAll((nodes) => nodes.map((node) => node.getAttribute('data-flow-stage'))),
      ).toEqual([...flowPaths.predict, ...flowPaths.train.slice(4)]);
    }
    const usableControls = await root
      .locator('[data-flow-mode-select], [data-flow-next], [data-flow-previous]')
      .evaluateAll(
        (buttons) =>
          buttons.filter(
            (button) =>
              button instanceof HTMLButtonElement &&
              !button.disabled &&
              button.getClientRects().length > 0 &&
              getComputedStyle(button).visibility !== 'hidden',
          ).length,
      );
    expect(
      usableControls,
      'Enhancement-only navigation must be disabled or hidden without JavaScript.',
    ).toBe(0);
    const loss = stage(root, 'claims', 'loss');
    await expect(loss).toHaveJSProperty('open', true);
    await loss.locator(':scope > summary').click();
    await expect(loss).toHaveJSProperty('open', false);
    await loss.locator(':scope > summary').click();
    await expect(loss.locator('.df-stage-body')).toBeVisible();
    const code = stage(root, 'claims', 'prepare').locator('.df-code');
    await code.locator(':scope > summary').click();
    await expect(code.locator('.expressive-code')).toBeVisible();
  } finally {
    await context.close();
  }
});

test('the requested mode, path, and stage keep their first-paint geometry while enhancement is deferred', async ({
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
    window.docsFlowLayoutShift = 0;
    new PerformanceObserver((list) => {
      for (const entry of list.getEntries()) {
        if (
          'hadRecentInput' in entry &&
          entry.hadRecentInput === false &&
          'value' in entry &&
          typeof entry.value === 'number'
        )
          window.docsFlowLayoutShift += entry.value;
      }
    }).observe({ type: 'layout-shift', buffered: true });
  });
  try {
    await page.goto('/concepts/data-flow/?case=claims&mode=train#flow-claims-loss', {
      waitUntil: 'commit',
    });
    await page.waitForFunction(
      () =>
        document.readyState !== 'loading' &&
        [...document.querySelectorAll<HTMLLinkElement>('link[rel="stylesheet"]')].every(
          (link) => link.sheet,
        ),
    );
    const root = page.locator('[data-data-flow]');
    await expect(root).toHaveAttribute('data-prepared', '');
    await expect(root).not.toHaveAttribute('data-flow-ready');
    await expect(root).toHaveAttribute('data-flow-mode', 'train');
    await expect(root).toHaveAttribute('data-flow-initial-stage', 'loss');
    await expect(root.locator('[data-case]:visible')).toHaveCount(1);
    await expect(root.locator('[data-case="claims"] details[data-flow-stage]:visible')).toHaveCount(
      1,
    );
    await expect(root.locator('[data-case="claims"] [data-flow-select]:visible')).toHaveCount(7);
    await expect(root.locator('[data-case="claims"] .df-stage-body:visible')).toHaveCount(1);
    const geometry = () =>
      root.evaluate((element) => {
        const box = (selector: string) => {
          const rect = element.querySelector(selector)!.getBoundingClientRect();
          return {
            x: rect.x + scrollX,
            y: rect.y + scrollY,
            width: rect.width,
            height: rect.height,
          };
        };
        return {
          cases: box('.case-picker'),
          modes: box('.df-mode-picker'),
          selected: box('[data-case="claims"] [data-flow-select="loss"]'),
          stage: box('#flow-claims-loss'),
        };
      });
    const before = await geometry();
    expect(before.cases.height).toBeGreaterThan(0);
    expect(before.modes.height).toBeGreaterThan(0);
    await page.evaluate(
      () =>
        new Promise<void>((resolve) =>
          requestAnimationFrame(() => requestAnimationFrame(() => resolve())),
        ),
    );
    release();
    await page.waitForLoadState('networkidle');
    await expect(root).toHaveAttribute('data-flow-ready', '');
    await phase(root, 'claims', 'train', 'loss');
    await expect(stage(root, 'claims', 'loss')).toHaveJSProperty('open', true);
    const after = await geometry();
    for (const control of ['cases', 'modes', 'selected', 'stage'] as const)
      for (const dimension of ['x', 'y', 'width', 'height'] as const)
        expect(
          Math.abs(after[control][dimension] - before[control][dimension]),
          `${control}.${dimension} changed during enhancement`,
        ).toBeLessThanOrEqual(1);
    expect(await page.evaluate(() => window.docsFlowLayoutShift)).toBeLessThan(0.01);
  } finally {
    release();
  }
});
