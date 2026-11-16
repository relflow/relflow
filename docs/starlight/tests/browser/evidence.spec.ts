import { test, expect } from '@playwright/test';

test('Evidence leads with the modeled question and a CI-driven comparison while diagnostics stay opt-in', async ({
  page,
  request,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  const response = await request.get('/evidence/selected.json');
  const publication = await response.json();
  const record = publication.results.P001;
  await page.goto('/evidence/P001/');
  await expect(page.locator('h1')).toHaveText('Can relflow add a list of numbers?');
  const report = page.locator('[data-evidence-report]');
  if (!record) {
    await expect(report.locator('.run-status')).toHaveText('No selected run');
    await expect(report.locator('[data-primary-comparison]')).toContainText(
      'No measured comparison',
    );
    await expect(report.locator('[data-primary-comparison] [data-value]')).toHaveCount(0);
    expect(errors).toEqual([]);
    return;
  }
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
  await expect(page.locator('[data-evidence-library] .readouts')).toHaveCount(0);
  await expect(page.locator('#P001')).not.toContainText('Constant-prediction baseline');
  await expect(page.locator('#P001')).not.toContainText('RMSE');
  const unmetId = (await unmet.count()) ? await unmet.getAttribute('id') : null;
  const pendingId = (await pending.count()) ? await pending.getAttribute('id') : null;
  if (unmetId) {
    await expect(unmet).toHaveCSS('box-shadow', /inset/);
    await expect(unmet.locator('.result-status')).toHaveText('Checks unmet');
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
    await page.goto('/evidence/');
    await expect(page.locator(`#${pendingId}`)).toHaveCSS('box-shadow', 'none');
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
  request,
}) => {
  const response = await request.get('/evidence/selected.json');
  expect(response.ok()).toBe(true);
  const publication = await response.json();
  const context = await browser.newContext({ javaScriptEnabled: false });
  try {
    const page = await context.newPage();
    await page.goto(`${baseURL}/evidence/P021/`);
    if (publication.results.P021) {
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
    } else {
      await expect(page.locator('.run-status')).toHaveText('No selected run');
      await expect(page.locator('[data-primary-comparison]')).toContainText(
        'No measured comparison',
      );
      await expect(page.locator('[data-primary-comparison] [data-value]')).toHaveCount(0);
      await expect(page.locator('[data-json-record]')).toHaveCount(0);
    }
    await page.goto(`${baseURL}/evidence/`);
    expect(await page.locator('[data-evidence-entry]').count()).toBeGreaterThan(0);
    await expect(page.locator('.library-controls')).toBeHidden();
  } finally {
    await context.close();
  }
});

test('every Evidence page renders its own reviewed question, setup, and scope', async ({
  page,
}, info) => {
  test.skip(
    info.project.name !== 'desktop-light',
    'Full catalog text audit once; shared mobile and theme behavior is checked above.',
  );
  test.setTimeout(120_000);
  const { reviews } = await import('../../src/evidence/reviews');
  await page.goto('/evidence/');
  const ids = await page
    .locator('[data-evidence-entry]')
    .evaluateAll((entries) => entries.map((entry) => entry.id));
  for (const id of ids) {
    await test.step(id, async () => {
      await page.goto(`/evidence/${id}/`);
      const content = reviews[id];
      await expect(page.locator('h1')).toHaveText(content.question);
      const setup = page.locator('#modeling-title').locator('..');
      await expect(setup).toContainText(content.modeling);
      const comparison = page.locator('[data-primary-comparison]');
      await expect(comparison).toHaveAttribute('data-metric', content.comparison.metric);
      if (content.note)
        await expect(comparison.locator('[data-evidence-scope]')).toHaveText(content.note);
      await expect(page.locator('.report-details')).not.toHaveAttribute('open');
      await expect(page.locator('.check-disclosure[open]')).toHaveCount(0);
    });
  }
});

test('capabilities and evaluation conditions combine, reload, and reset without changing the publication', async ({
  page,
  request,
}) => {
  const publication = await (await request.get('/evidence/selected.json')).text();
  await page.goto('/evidence/?group=structure&type=Hash&condition=unseen-identities');
  const library = page.locator('[data-evidence-library]');
  await expect(library).toHaveAttribute('data-ready', '');
  const group = library.locator('[data-library-group]');
  for (const capability of [
    'computation',
    'structure',
    'representation',
    'training',
    'reliability',
  ])
    await expect(group.locator(`option[value="${capability}"]`)).toHaveCount(1);
  await expect(group).toHaveValue('structure');
  await expect(library.locator('#P025')).toBeVisible();
  const ids = await library
    .locator('[data-evidence-entry]:visible')
    .evaluateAll((nodes) => nodes.map((node) => node.id));
  for (const row of await library.locator('[data-evidence-entry]:visible').all()) {
    await expect(row).toHaveAttribute('data-group', 'structure');
    await expect(row).toHaveAttribute('data-types', /Hash/);
    await expect(row).toHaveAttribute('data-conditions', /unseen-identities/);
  }
  await page.reload();
  await expect(library).toHaveAttribute('data-ready', '');
  expect(
    await library
      .locator('[data-evidence-entry]:visible')
      .evaluateAll((nodes) => nodes.map((node) => node.id)),
  ).toEqual(ids);
  await group.selectOption('training');
  await expect(library.locator('[data-library-empty]')).toBeVisible();
  await library.locator('[data-library-reset]').click();
  await expect(library.locator('[data-evidence-entry]:visible')).toHaveCount(
    await library.locator('[data-evidence-entry]').count(),
  );
  await group.selectOption('computation');
  await expect(group).toHaveValue('computation');
  await expect(page).toHaveURL(/group=computation/);
  expect(await (await request.get('/evidence/selected.json')).text()).toBe(publication);
});
