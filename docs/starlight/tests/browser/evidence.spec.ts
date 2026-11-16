import { test, expect } from '@playwright/test';

test('Evidence leads with the modeled question and a CI-driven comparison while diagnostics stay opt-in', async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const response = await request.get('/evidence/records/P001.json');
  const record = await response.json();
  await page.goto('/evidence/P001/');
  await expect(page.locator('h1')).toHaveText('Can relflow add a list of numbers?');
  const report = page.locator('[data-evidence-report]');
  await expect(report.locator('h2')).toHaveText([
    'How is the question modeled?',
    'Primary comparison',
    'What did the checks establish?',
  ]);
  await expect(report.locator('.illustrative-record')).not.toHaveAttribute('open');
  await expect(report.locator('.check-disclosure')).not.toHaveAttribute('open');
  await expect(report.locator('.report-details')).not.toHaveAttribute('open');
  const chart = report.locator('[data-primary-comparison]');
  await expect(chart).toHaveAttribute('data-metric', 'RMSE');
  const points = chart.locator('[data-metric-path]');
  await expect(points).toHaveCount(2);
  await expect(points.nth(0)).toHaveAttribute(
    'data-value',
    String(record.metrics.measured.baseline_rmse),
  );
  await expect(points.nth(1)).toHaveAttribute('data-value', String(record.metrics.measured.rmse));
  await expect(chart).not.toContainText('drift');
  await report.locator('.check-disclosure > summary').click();
  await expect(report.locator('tbody tr[data-check-name]')).toHaveCount(
    Object.keys(record.checks).length,
  );
  await report.locator('.report-details > summary').click();
  await expect(report.locator('.report-details')).toContainText('permutation_drift');
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    await page.evaluate(() => innerWidth),
  );
  expect(errors).toEqual([]);
});

test('the catalog verdict agrees with the report, including unmet and pending entries when present', async ({
  page,
}) => {
  await page.goto('/evidence/');
  const unmet = page.locator('[data-evidence-entry][data-status="not_met"]').first();
  const pending = page.locator('[data-evidence-entry][data-status="missing"]').first();
  const unmetId = (await unmet.count()) ? await unmet.getAttribute('id') : null;
  const pendingId = (await pending.count()) ? await pending.getAttribute('id') : null;
  if (unmetId) {
    await page.goto(`/evidence/${unmetId}/`);
    await expect(page.locator('[data-evidence-report]')).toHaveAttribute(
      'data-evidence-outcome',
      'not_met',
    );
    await expect(page.locator('.run-status')).toHaveText('Checks unmet');
    await expect(page.locator('.check-disclosure')).not.toHaveAttribute('open');
    await page.locator('.check-disclosure > summary').click();
    expect(await page.locator('[data-check-outcome="not_met"]').count()).toBeGreaterThan(0);
  }
  if (pendingId) {
    await page.goto(`/evidence/${pendingId}/`);
    await expect(page.locator('.run-status')).toHaveText('No selected run');
    await expect(page.locator('[data-primary-comparison] [data-value]')).toHaveCount(0);
    await expect(page.locator('[data-primary-comparison]')).toContainText('No measured comparison');
  }
});

test('leaf tags filter the catalog, preserve shareable URLs, and reveal linked proofs', async ({
  page,
}) => {
  await page.goto('/evidence/?type=Hash');
  const library = page.locator('[data-evidence-library]');
  await expect(library).toHaveAttribute('data-ready', '');
  const total = await library.locator('[data-evidence-entry]').count();
  const choice = library.locator('[data-library-type]');
  await expect(choice).toHaveValue('Hash');
  const rows = library.locator('[data-evidence-entry]:visible');
  expect(await rows.count()).toBeGreaterThan(1);
  for (const row of await rows.all())
    expect((await row.getAttribute('data-types'))!.split(' ')).toContain('Hash');
  await expect(library.locator('#P021')).toBeVisible();
  await library.locator('#P021 h3 a').click();
  await expect(page.locator('h1')).toContainText('matching IDs');
  await page.locator('.evidence-report .evidence-types a').filter({ hasText: 'Category' }).click();
  await expect(page).toHaveURL(/type=Category/);
  await expect(choice).toHaveValue('Category');
  await page.goto('/evidence/?type=Text#P001');
  await expect(library.locator('#P001')).toBeVisible();
  await expect(choice).toHaveValue('');
  await choice.selectOption('Set');
  await library.locator('[data-library-search]').fill('no such evidence phrase');
  await expect(library.locator('[data-library-empty]')).toBeVisible();
  await library.locator('[data-library-reset]').click();
  await expect(choice).toHaveValue('');
  await expect(library.locator('[data-evidence-entry]:visible')).toHaveCount(total);
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
    await page.evaluate(() => innerWidth),
  );
});

test('the primary chart and nested investigation disclosures work without JavaScript', async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  try {
    const page = await context.newPage();
    await page.goto(`${baseURL}/evidence/P021/`);
    await expect(page.locator('[data-primary-comparison] [data-value]')).toHaveCount(2);
    await expect(page.locator('.report-details')).not.toHaveAttribute('open');
    await page.locator('.report-details > summary').click();
    await page.locator('.raw-record > summary').click();
    const raw = JSON.parse((await page.locator('[data-json-record] code').textContent())!);
    const values = await page
      .locator('[data-primary-comparison] [data-value]')
      .evaluateAll((elements) =>
        elements.map((element) => Number(element.getAttribute('data-value'))),
      );
    expect(values).toEqual([raw.metrics.category_oov_auc, raw.metrics.hash_auc]);
    await page.goto(`${baseURL}/evidence/`);
    expect(await page.locator('[data-evidence-entry]').count()).toBeGreaterThan(0);
    await expect(page.locator('.library-controls')).toBeHidden();
  } finally {
    await context.close();
  }
});
