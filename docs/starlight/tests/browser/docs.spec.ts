import { test as base, expect } from '@playwright/test';
import { scenarios } from '../../src/data/concepts/scenarios';
import { predictions } from '../../src/data/concepts/predictions';

const test = base.extend({
  page: async ({ page }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await use(page);
    expect(errors, 'Browser interactions must not throw uncaught errors.').toEqual([]);
  },
});

test('a selected use case follows native links and explicit URLs take precedence', async ({
  page,
}) => {
  await page.goto('/?case=tickets');
  const choice = page.locator('[data-case-select="maintenance"]');
  await choice.click();
  await expect(choice).toHaveAttribute('aria-pressed', 'true');
  await expect(page).toHaveURL(/case=maintenance/);
  await page.locator('.cases-next a[href^="/concepts/models/"]').click();
  await expect(page).toHaveURL(/\/concepts\/models\/\?case=maintenance/);
  await expect(page.locator('[data-case="maintenance"]')).toBeVisible();

  await page.goto('/concepts/models/?case=churn');
  await expect(page.locator('[data-case-select="churn"]')).toHaveAttribute('aria-pressed', 'true');
  await expect(page.locator('[data-case="maintenance"]')).toBeHidden();
  const first = page.locator('[data-case-select]').first();
  await first.focus();
  await page.keyboard.press('End');
  await expect(page.locator('[data-case-select="claims"]')).toBeFocused();
  await expect(page.locator('[data-case="claims"]')).toBeVisible();
  await page.keyboard.press('Home');
  await expect(first).toBeFocused();
  await expect(first).toHaveAttribute('aria-pressed', 'true');
});

test('the homepage switches among six shared data, model, and schema viewers', async ({ page }) => {
  await page.goto('/?case=tickets');
  const cases = page.locator('[data-use-cases]');
  await expect(cases).toHaveAttribute('data-ready', '');
  await expect(cases.locator('[data-case-select]')).toHaveCount(6);
  await expect(cases.locator('[data-data-views]')).toHaveCount(6);
  await expect(cases.locator('[data-value-tree]')).toHaveCount(0);

  for (const id of ['tickets', 'churn', 'demand', 'credit', 'maintenance', 'claims']) {
    const choice = cases.locator(`[data-case-select="${id}"]`);
    await choice.click();
    await expect(choice).toHaveAttribute('aria-pressed', 'true');
    await expect(cases.locator('[data-case]:visible')).toHaveCount(1);
    const selected = cases.locator(`[data-case="${id}"]`);
    await expect(selected).toBeVisible();
    await expect(selected.locator('[data-data-views]')).toBeVisible();
    const diagram = selected.locator('.erd-data [data-erd]');
    await diagram.scrollIntoViewIfNeeded();
    await expect(diagram).toHaveAttribute('data-erd-ready', '');
    await expect(diagram.locator('[data-entity]').first()).toBeVisible();
    await expect(diagram.locator('.erd-column').first()).toHaveAttribute('title', /.+/);
    await expect
      .poll(() =>
        diagram
          .locator('[data-edge-from] .erd-edge-line')
          .evaluateAll(
            (edges) => edges.length > 0 && edges.every((edge) => Boolean(edge.getAttribute('d'))),
          ),
      )
      .toBe(true);
  }
});

test('use-case deep links select the matching schema and copying keeps exact source', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/concepts/runtime/?case=tickets');
  const choice = page.locator('[data-case-select="churn"]');
  await choice.click();
  const example = page.locator('[data-case="churn"]');
  await expect(example).toBeVisible();
  await example.getByRole('radio', { name: 'Schema', exact: true }).check();
  await example.locator('.erd-schema .copy button').click();
  await expect
    .poll(() => page.evaluate(() => navigator.clipboard.readText()))
    .toBe(scenarios.find((scenario) => scenario.id === 'churn')!.schema);
  await example.locator('[data-case-permalink]').click();
  await page.reload();
  await expect(example).toBeVisible();
  await expect(choice).toHaveAttribute('aria-pressed', 'true');
});

test('tree controls include returned metadata, preserve other instances, and copy the full row', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/concepts/runtime/?case=tickets');
  const other = page.locator('[data-case="churn"] [data-value-tree]');
  const initial = await other
    .locator('details')
    .evaluateAll((nodes) => nodes.map((node) => (node as HTMLDetailsElement).open));
  const tree = page.locator('[data-case="tickets"] [data-value-tree]');
  await tree.getByRole('button', { name: 'Collapse all', exact: true }).click();
  await expect(tree.locator('details[open]')).toHaveCount(0);
  const expand = tree.getByRole('button', { name: 'Expand all', exact: true });
  await expand.focus();
  await page.keyboard.press('Enter');
  await expect(tree.locator('details:not([open])')).toHaveCount(0);
  await expect(tree.locator('.vt-returned').first()).toBeVisible();
  expect(
    await other
      .locator('details')
      .evaluateAll((nodes) => nodes.map((node) => (node as HTMLDetailsElement).open)),
  ).toEqual(initial);
  await tree.getByRole('button', { name: 'Copy JSON', exact: true }).click();
  const copied = JSON.parse(await page.evaluate(() => navigator.clipboard.readText()));
  const block = predictions.examples
    .find((example) => example.caseId === 'tickets')!
    .blocks.find((block) => block.kind === 'prediction')!;
  expect(copied).toEqual(block.value);
  expect(copied).not.toHaveProperty('inputs');
});

test('opening a column note dismisses a sidebar note even while its link keeps focus', async ({
  page,
  isMobile,
}) => {
  test.skip(isMobile, 'The persistent sidebar is a desktop interaction.');
  await page.goto('/');
  const example = page.locator('[data-case][aria-hidden="false"]');
  const column = example.locator('.erd-column').first();
  await column.scrollIntoViewIfNeeded();
  await expect(example.locator('.erd-data [data-erd]')).toHaveAttribute('data-erd-ready', '');
  const staticDescription = await column.getAttribute('aria-describedby');
  const link = page.locator('.sidebar-content a').first();
  await link.focus();
  await expect(page.locator('#rf-sidebar-tooltip')).toBeVisible();
  await column.hover();
  const datasetNote = example.locator('.erd-data [data-erd-tooltip]');
  await expect(datasetNote).toBeVisible();
  await expect(page.locator('#rf-sidebar-tooltip')).toBeHidden();
  await expect(link).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(datasetNote).toBeHidden();
  await expect(column).toHaveAttribute('aria-describedby', staticDescription!);
  await expect(link).toBeFocused();
});

test('column and code notes coordinate hover, focus, tap, and Escape', async ({
  page,
  isMobile,
}) => {
  await page.goto('/concepts/runtime/?case=tickets');
  const example = page.locator('[data-case="tickets"]');
  const column = example.locator('.erd-column').first();
  const datasetNote = example.locator('.erd-data [data-erd-tooltip]');
  const staticDescription = await column.getAttribute('aria-describedby');
  await column.scrollIntoViewIfNeeded();
  await expect(example.locator('.erd-data [data-erd]')).toHaveAttribute('data-erd-ready', '');
  if (isMobile) await column.tap();
  else await column.focus();
  await expect(datasetNote).toBeVisible();
  await page.keyboard.press('Escape');
  await expect(datasetNote).toBeHidden();
  await expect(column).toHaveAttribute('aria-describedby', staticDescription!);

  await example.getByRole('radio', { name: 'Schema', exact: true }).check();
  const note = example.locator('.erd-schema .rf-code-note').first();
  await note.scrollIntoViewIfNeeded();
  if (isMobile) {
    await note.tap();
    await expect(page.locator('#rf-code-tooltip')).toBeVisible();
    await note.tap();
    await expect(page.locator('#rf-code-tooltip')).toBeHidden();
  } else {
    await note.focus();
    const popup = page.locator('#rf-code-tooltip');
    await expect(popup).toBeVisible();
    await popup.hover();
    await expect(popup).toBeVisible();
    await page.keyboard.press('Escape');
    // Let delayed hover events run when removing the popup exposes a term below it.
    await page.waitForTimeout(250);
    await expect(popup).toBeHidden();
    await expect(note).toBeFocused();
    await example.getByRole('radio', { name: 'Data', exact: true }).check();
    await column.hover();
    await expect(datasetNote).toBeVisible();
    await expect(popup).toBeHidden();
  }
  await expect(page.locator('[role="tooltip"]:visible')).toHaveCount(isMobile ? 0 : 1);
});

test('code explanations distinguish fit settings and JSON decoding from model construction', async ({
  page,
}) => {
  await page.goto('/advanced/training/?case=churn');
  const example = page.locator('[data-case="churn"]');
  const fitLine = example.locator('.ec-line').filter({ hasText: /fit\(data, preset=/ });
  await expect(fitLine.locator('.rf-code-note').filter({ hasText: 'preset' })).toHaveAttribute(
    'data-note-text',
    /training/i,
  );
  await expect(fitLine.locator('.rf-code-note').filter({ hasText: 'preset' })).not.toHaveAttribute(
    'data-note-text',
    /Construct a new/,
  );
  await page.goto('/advanced/inference-and-deployment/');
  const jsonLine = page.locator('.ec-line').filter({ hasText: 'json.load(response)' });
  await expect(jsonLine).toHaveCount(1);
  await expect(jsonLine.locator('.rf-code-note')).toHaveCount(0);
});

test('diagram overflow hints update after the canvas stops shrinking', async ({
  page,
  isMobile,
}) => {
  test.skip(isMobile, 'Exercise the desktop-to-mobile resize path in desktop contexts.');
  await page.goto('/featureless/?case=churn');
  const diagram = page.locator('[data-case="churn"] .erd-data [data-erd]');
  await diagram.scrollIntoViewIfNeeded();
  await expect(diagram).toHaveAttribute('data-erd-ready', '');
  for (const width of [1440, 900, 800, 700, 650, 635, 620, 601, 600, 599, 390]) {
    await page.setViewportSize({ width, height: 1000 });
    await expect
      .poll(async () =>
        diagram.evaluate((root) => {
          const canvas = root.querySelector<HTMLElement>('.erd-canvas')!;
          const viewport = root.querySelector<HTMLElement>('.erd-scroll')!;
          const hint = root.querySelector<HTMLElement>('[data-erd-scroll-hint]')!;
          return hint.hidden === canvas.getBoundingClientRect().width <= viewport.clientWidth + 2;
        }),
      )
      .toBe(true);
  }
});

test('all cases, native trees, and descriptions stay readable without JavaScript', async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  try {
    await page.goto(`${baseURL}/`);
    const cases = page.locator('[data-use-cases]');
    await expect(cases.locator('[data-case]:visible')).toHaveCount(6);
    await expect(cases.locator('[data-erd]:visible')).toHaveCount(6);
    await expect(cases.locator('[data-value-tree]')).toHaveCount(0);
    for (const example of await cases.locator('[data-case]').all()) {
      const column = example.locator('.erd-column').first();
      const description = await column.getAttribute('title');
      expect(description).toBeTruthy();
      const descriptionId = await column.getAttribute('aria-describedby');
      expect(descriptionId).toBeTruthy();
      await expect(page.locator(`[id="${descriptionId}"]`)).toHaveText(description!);
    }
    await page.goto(`${baseURL}/data-types/boolean/`);
    await expect(page.locator('[data-type-showcase], [data-data-explorer]')).toHaveCount(0);
    const tree = page.locator('[data-value-tree]').first();
    await expect(tree).toBeVisible();
    await tree.locator('details').first().locator(':scope > summary').click();
    await expect(tree.locator('details').first()).not.toHaveAttribute('open');
    await expect(tree.locator('[data-tree-tools]')).toBeHidden();
  } finally {
    await context.close();
  }
});
