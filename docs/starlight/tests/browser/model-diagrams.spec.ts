import { test, expect } from '@playwright/test';
import cases from '../../src/data/use-cases.json' with { type: 'json' };
import { datasets } from '../../src/data/datasets';
import { flatten, parseSchema } from '../../src/model-explainer/schema';

test('all six cases switch among data ERDs, model ERDs, and copyable schemas', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/concepts/models/?case=tickets');
  const main = page.locator('main');
  await expect(
    main.getByRole('heading', { name: 'Explore a use case', exact: true }),
  ).toBeVisible();
  await expect(main.locator('[data-model-explainer], [data-model-step]')).toHaveCount(0);
  await expect(
    main.getByRole('heading', { name: 'How it fits together', exact: true }),
  ).toHaveCount(0);
  await expect(main.locator('[data-case-select]')).toHaveCount(6);

  for (const item of cases) {
    await page.locator(`[data-case-select="${item.id}"]`).click();
    const example = main.locator(`[data-case="${item.id}"]`);
    await expect(example).toBeVisible();
    await expect(main.locator('[data-case]:visible')).toHaveCount(1);
    const data = example.getByRole('region', { name: 'Data', exact: true });
    const model = example.getByRole('region', { name: 'Model', exact: true });
    const schema = example.getByRole('region', { name: 'Schema', exact: true });
    await expect(example.getByRole('radio')).toHaveCount(3);
    await expect(data).toBeVisible();
    await expect(model).toBeHidden();
    await expect(schema).toBeHidden();
    const dataset = datasets.find((dataset) => dataset.id === item.id)!;
    await expect(data.locator('.erd-column')).toHaveCount(
      dataset.entities.flatMap((entity) => entity.columns).length,
    );

    await example.getByRole('radio', { name: 'Data', exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(example.getByRole('radio', { name: 'Model', exact: true })).toBeChecked();
    await expect(model).toBeVisible();
    await expect(data).toBeHidden();
    const expected = flatten(parseSchema(item.code).nodes)
      .map((node) => node.name)
      .sort();
    expect((await model.locator('.erd-column-name').allTextContents()).sort()).toEqual(expected);
    const column = model.locator('.erd-column').first();
    await column.scrollIntoViewIfNeeded();
    await expect(model.locator('[data-erd]')).toHaveAttribute('data-erd-ready', '');
    await column.focus();
    await expect(model.locator('[data-erd-tooltip]')).toBeVisible();
    await expect(model.locator('[data-erd-tooltip]')).toContainText('Configuration:');
    await page.keyboard.press('Escape');
    await expect(model.locator('[data-erd-tooltip]')).toBeHidden();

    await example.getByRole('radio', { name: 'Model', exact: true }).focus();
    await page.keyboard.press('ArrowRight');
    await expect(example.getByRole('radio', { name: 'Schema', exact: true })).toBeChecked();
    await expect(schema).toBeVisible();
    await expect(model).toBeHidden();
    await expect(data).toBeHidden();
    await schema.locator('.copy button').click();
    await expect.poll(() => page.evaluate(() => navigator.clipboard.readText())).toBe(item.code);
    const note = schema.locator('.rf-code-note').first();
    await note.focus();
    await expect(page.locator('#rf-code-tooltip')).toBeVisible();
    await page.keyboard.press('Escape');
    await expect(page.locator('#rf-code-tooltip')).toBeHidden();
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
      await page.evaluate(() => innerWidth),
    );
  }
  expect(errors).toEqual([]);
});

test('all three views remain selectable without JavaScript', async ({ browser, baseURL }) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  try {
    await page.goto(`${baseURL}/concepts/models/`);
    await expect(page.locator('[data-case]:visible')).toHaveCount(6);
    await expect(page.locator('.erd-data:visible')).toHaveCount(6);
    await expect(page.locator('[data-model-explainer]')).toHaveCount(0);
    for (const item of cases) {
      const example = page.locator(`[data-case="${item.id}"]`);
      await example.getByRole('radio', { name: 'Model', exact: true }).check();
      await expect(example.locator('.erd-model')).toBeVisible();
      await expect(example.locator('.erd-data')).toBeHidden();
      await expect(example.locator('.erd-model .erd-column').first()).toHaveAttribute(
        'title',
        /Configuration:/,
      );
      await example.getByRole('radio', { name: 'Schema', exact: true }).check();
      await expect(example.locator('.erd-schema .expressive-code')).toBeVisible();
      await expect(example.locator('.erd-model')).toBeHidden();
      await expect(example.locator('.erd-data')).toBeHidden();
    }
    await expect(page.locator('.erd-schema:visible')).toHaveCount(6);
  } finally {
    await context.close();
  }
});

test('the shared viewer works across the homepage, concepts, pipeline, and advanced guides', async ({
  page,
}) => {
  for (const route of [
    '/?case=tickets',
    '/featureless/?case=tickets',
    '/concepts/branches/?case=tickets',
    '/concepts/runtime/?case=tickets',
    '/concepts/data-flow/?case=tickets',
    '/advanced/training/?case=tickets',
    '/advanced/custom-data-modules/',
    '/advanced/inference-and-deployment/',
    '/advanced/online-vocabulary/',
    '/advanced/masking-and-pretraining/',
    '/advanced/custom-data-types/',
  ]) {
    const response = await page.goto(route);
    expect(response?.status(), route).toBe(200);
    const viewer = page.locator('[data-data-views]:visible').first();
    await expect(viewer.getByRole('radio')).toHaveCount(3);
    await viewer.getByRole('radio', { name: 'Model', exact: true }).check();
    await expect(viewer.getByRole('region', { name: 'Model', exact: true })).toBeVisible();
    await viewer.getByRole('radio', { name: 'Schema', exact: true }).check();
    await expect(viewer.locator('.erd-schema .expressive-code')).toBeVisible();
    await viewer.getByRole('radio', { name: 'Data', exact: true }).check();
    await expect(viewer.getByRole('region', { name: 'Data', exact: true })).toBeVisible();
    expect(await page.evaluate(() => document.documentElement.scrollWidth), route).toBe(
      await page.evaluate(() => innerWidth),
    );
  }
});
