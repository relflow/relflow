import { test, expect } from '@playwright/test';

test('API map opens source-derived fields, inherited documentation, and request methods', async ({
  page,
}) => {
  await page.goto('/api-map/?q=rf.Number');
  const map = page.locator('[data-api-map]');
  await expect(map.locator('[data-api-entry]:visible')).toHaveCount(1);
  await map.getByRole('link', { name: 'rf.Number', exact: true }).click();
  await expect(page).toHaveURL(/\/api\/rf-number\/$/);
  await expect(page.getByRole('heading', { level: 1 })).toHaveText('rf.Number');
  const reference = page.locator('.api-reference');
  await expect(reference).toContainText('Continuous numeric scalar with raw and signed-log lanes');
  await expect(reference.getByRole('link', { name: 'the base definition ↗' })).toHaveAttribute(
    'href',
    /\/src\/relflow\/tensorfields\/extensions\/number\.py#L\d+$/,
  );
  const fields = reference.locator('.api-fields');
  await expect(
    fields.getByRole('row').filter({ has: page.getByRole('rowheader', { name: /^active/ }) }),
  ).toContainText('Inherited');
  await expect(
    fields.getByRole('row').filter({ has: page.getByRole('rowheader', { name: /^n_bands/ }) }),
  ).toContainText('8');
  const normalization = reference.locator('#method-normalization');
  await normalization.locator('summary').click();
  await expect(normalization).toContainText('dict[str, float | int | None]');
  await expect(normalization).toContainText('source-unit normalizer');
  await expect(reference.locator('nav a').last()).toHaveAttribute(
    'href',
    /\/blob\/[a-f0-9]{40}\/src\/relflow\/tensorfields\/extensions\/number.py#L\d+$/,
  );
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    await page.evaluate(() => innerWidth),
  );
});

test('model methods show source parameter documentation and typed returns', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/api/model-update/');
  const reference = page.locator('.api-reference');
  await expect(
    reference
      .locator('table')
      .filter({ has: page.locator('caption').filter({ hasText: 'Parameters' }) }),
  ).toContainText('Predicates used to select nodes.');
  await expect(reference).toContainText('Raise when a selected node cannot accept');
  await expect(
    reference
      .locator('table')
      .filter({ has: page.locator('caption').filter({ hasText: 'Returns' }) }),
  ).toContainText('None');
  const code = reference.locator('.expressive-code').first();
  await code.locator('.copy button').click();
  expect(await page.evaluate(() => navigator.clipboard.readText())).toContain('**values: object');
  await page.goto('/api/rf-model/');
  await page.locator('.api-overloads summary').click();
  await expect(page.locator('.api-overloads .expressive-code')).toHaveCount(3);
  await page.locator('.api-method-link a').filter({ hasText: 'rf.Model.fit' }).click();
  await expect(page).toHaveURL(/\/api\/model-fit\/$/);
  await expect(page.locator('.api-reference')).toContainText('ema_decay');
});

test('aliases and external reexports identify their source without invented interfaces', async ({
  page,
}) => {
  await page.goto('/api/rf-requestbase/');
  await page.locator('.api-reference').getByRole('link', { name: 'rf.Leaf', exact: true }).click();
  await expect(page).toHaveURL(/\/api\/rf-leaf\/$/);
  await page.goto('/api/rf-earlystopping/');
  const reference = page.locator('.api-reference');
  await expect(reference).toContainText('lightning.pytorch.callbacks.EarlyStopping');
  await expect(
    reference.getByRole('link', { name: 'Read the upstream reference ↗' }),
  ).toHaveAttribute('href', /^https:\/\/lightning.ai\//);
  await expect(reference.getByRole('heading', { name: 'Signature' })).toHaveCount(0);
});

test('API navigation and signatures remain readable without JavaScript', async ({ browser }) => {
  const context = await browser.newContext({
    javaScriptEnabled: false,
    viewport: { width: 390, height: 844 },
  });
  const page = await context.newPage();
  try {
    await page.goto('/api-map/');
    await page
      .locator('[data-api-entry]')
      .getByRole('link', { name: 'rf.Model', exact: true })
      .click();
    await expect(page.getByRole('heading', { level: 1 })).toHaveText('rf.Model');
    await page.locator('.api-overloads summary').click();
    await expect(page.locator('.api-overloads')).toContainText('preset: str');
    expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
      await page.evaluate(() => innerWidth),
    );
  } finally {
    await context.close();
  }
});
