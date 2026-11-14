import { test as base, expect, type Locator } from '@playwright/test';
import { explainerCases, explainerByCase } from '../../src/data/model-explainer';
import { flatten } from '../../src/model-explainer/schema';

const test = base.extend({
  page: async ({ page }, use) => {
    const errors: string[] = [];
    page.on('pageerror', (error) => errors.push(error.message));
    await use(page);
    expect(errors, 'The model explainer must not throw browser errors.').toEqual([]);
  },
});

function explainer(page: Locator, id: string) {
  return page.locator(`[data-case="${id}"] [data-model-explainer]`);
}

async function renderedSource(root: Locator) {
  return root
    .locator('.me-source .ec-line:visible > .code')
    .evaluateAll((lines) =>
      lines.map((line) => (line.textContent === '\n' ? '' : line.textContent || '')).join('\n'),
    );
}

test('all six cases grow through three coherent schemas with one source block and exact copying', async ({
  page,
  context,
}) => {
  await context.grantPermissions(['clipboard-read', 'clipboard-write']);
  await page.goto('/concepts/models/?case=tickets');
  for (const item of explainerCases) {
    await page.locator(`[data-case-select="${item.id}"]`).click();
    const root = explainer(page.locator('main'), item.id);
    await expect(root).toBeVisible();
    await expect(root).toHaveAttribute('data-model-ready', '');
    await expect(root.locator('.me-source .expressive-code')).toHaveCount(1);
    await root.locator('.me-source > summary').click();
    for (const stage of item.stages) {
      await root.locator(`[data-model-step="${stage.id}"]`).click();
      await expect(root).toHaveAttribute('data-model-stage', stage.id);
      await expect(root.locator(`[data-model-step="${stage.id}"]`)).toHaveAttribute(
        'aria-pressed',
        'true',
      );
      const fields = flatten(stage.nodes);
      const visible = await root
        .locator('[data-model-select]:visible')
        .evaluateAll((nodes) => nodes.map((node) => node.getAttribute('data-model-select')));
      expect(visible).toEqual(['/', ...fields.map((node) => node.path)]);
      expect(await renderedSource(root)).toBe(stage.schema);
      await root.locator('.me-source .copy button').click();
      await expect
        .poll(() => page.evaluate(() => navigator.clipboard.readText()))
        .toBe(stage.schema);
      const info = root.locator(`[data-model-stage-info="${stage.id}"]`);
      await expect(info).toBeVisible();
      await expect(info.locator('.me-flow-target strong')).toHaveText(
        String(fields.filter((node) => node.target).length),
      );
    }
    await expect(root.locator('.me-label-requirement:visible')).toContainText('Training labels');
    const additional = flatten(item.stages.at(-1)!.nodes).find((node) => node.added);
    if (additional) {
      const choice = root.locator(`[data-model-select="${additional.path}"]`);
      await expect(choice.locator('.me-node-added')).toHaveText('New label');
      await choice.click();
      await expect(root.locator('[data-model-requirement]')).toHaveText(
        additional.labelRequirement!,
      );
      await root.locator('[data-model-step="start"]').click();
      await expect(choice).toBeHidden();
      await expect(root.locator('[data-model-path]')).toHaveText('/');
    }
    await expect(root.locator('.me-footnote')).toContainText(
      'no executed predictions or performance measurements',
    );
    expect(
      await page.evaluate(
        () => document.documentElement.scrollWidth - document.documentElement.clientWidth,
      ),
    ).toBeLessThanOrEqual(1);
  }
});

test('field, branch, and target inspection explain representation, context, objectives, and nested outputs', async ({
  page,
}) => {
  await page.goto('/concepts/models/?case=tickets');
  const main = page.locator('main');
  const cases = [
    {
      id: 'tickets',
      input: '/product',
      type: 'Category',
      pattern: /learned vocabulary/,
      branch: '/messages',
      target: '/escalated',
      address: 'predictions.escalated.content.probability',
    },
    {
      id: 'churn',
      input: '/monthly_usage/month',
      type: 'Period',
      pattern: /calendar parts.*cyclical/,
      branch: '/monthly_usage',
      target: '/churn_90d',
      address: 'predictions.churn_90d.content.probability',
    },
    {
      id: 'demand',
      input: '/history/units_sold',
      type: 'Number',
      pattern: /raw and signed-log/,
      branch: '/history',
      target: '/horizon/units_sold',
      address: 'predictions.horizon[i].units_sold.content',
    },
    {
      id: 'credit',
      input: '/product',
      type: 'Category',
      pattern: /learned vocabulary/,
      branch: '/accounts',
      target: '/default_12m',
      address: 'predictions.default_12m.content.probability',
    },
    {
      id: 'maintenance',
      input: '/work_orders/parts_replaced',
      type: 'Set',
      pattern: /without imposing an order/,
      branch: '/sensor_readings',
      target: '/failure_30d',
      address: 'predictions.failure_30d.content.probability',
    },
    {
      id: 'claims',
      input: '/claims/invoices/provider',
      type: 'Hash',
      pattern: /without a persistent learned ID vocabulary/,
      branch: '/claims/invoices/line_items',
      target: '/claims/paid_amount',
      address: 'predictions.claims[i].paid_amount.content',
    },
  ];
  for (const item of cases) {
    await page.locator(`[data-case-select="${item.id}"]`).click();
    const root = explainer(main, item.id);
    await root.locator('[data-model-step="targets"]').click();
    await root.locator(`[data-model-select="${item.input}"]`).click();
    await expect(root.locator('[data-model-kind]')).toHaveText(`${item.type} input`);
    await expect(root.locator('[data-model-representation]')).toContainText(item.pattern);
    await expect(root.locator('[data-model-learning]')).toContainText(
      'no reconstruction objective',
    );
    await expect(root.locator('[data-model-address-wrap]')).toBeHidden();
    await root.locator(`[data-model-select="${item.branch}"]`).click();
    await expect(root.locator('[data-model-kind]')).toHaveText('Record collection');
    await expect(root.locator('[data-model-context]')).toContainText(item.branch);
    await expect(root.locator('[data-model-learning]')).toContainText(
      'does not create a learning objective',
    );
    await root.locator(`[data-model-select="${item.target}"]`).click();
    await expect(root.locator('[data-model-representation]')).toContainText(
      'out of encoder context',
    );
    await expect(root.locator('[data-model-learning]')).toContainText(
      /classification objective|regression objective/,
    );
    await expect(root.locator('[data-model-address]')).toHaveText(item.address);
    await expect(root.locator('[data-model-output]')).toContainText('state and inferred');
    await expect(root.locator('[data-model-select][aria-pressed="true"]')).toHaveCount(1);
  }
  const claims = explainer(main, 'claims');
  await claims.locator('[data-model-select="/claims/is_fraudulent"]').click();
  await expect(claims.locator('[data-model-address]')).toHaveText(
    'predictions.claims[i].is_fraudulent.content.probability',
  );
});

test('keyboard selection, source highlighting, common code notes, and case state work together', async ({
  page,
  isMobile,
}) => {
  await page.goto('/concepts/models/?case=tickets');
  const root = explainer(page.locator('main'), 'tickets');
  await root.locator('[data-model-step="start"]').focus();
  await page.keyboard.press('End');
  await expect(root.locator('[data-model-step="targets"]')).toBeFocused();
  await expect(root).toHaveAttribute('data-model-stage', 'targets');
  await page.keyboard.press('ArrowLeft');
  await expect(root.locator('[data-model-step="context"]')).toBeFocused();
  await expect(root).toHaveAttribute('data-model-stage', 'context');
  const body = root.locator('[data-model-select="/messages/body"]');
  await body.focus();
  await page.keyboard.press('Enter');
  if (isMobile) {
    await expect(root.locator('[data-model-name]')).toBeFocused();
    await expect(root.locator('[data-model-name]')).toBeInViewport();
    await root.locator('[data-model-back]').click();
    await expect(body).toBeFocused();
    await expect(body).toBeInViewport();
  } else await expect(root.locator('[data-model-back]')).toBeHidden();
  await expect(root.locator('[data-model-kind]')).toHaveText('Text input');
  await expect(root.locator('[data-model-representation]')).toContainText(
    'frozen pretrained text encoder',
  );
  await root.locator('.me-source > summary').click();
  const line = root.locator('.ec-line[data-model-paths~="/messages/body"]');
  await expect(line).toHaveAttribute('data-model-active', '');
  const note = line.locator('.rf-code-note').filter({ hasText: 'rf.Text' }).first();
  if (isMobile) await note.tap();
  else await note.hover();
  await expect(page.locator('#rf-code-tooltip')).toBeVisible();
  await expect(page.locator('#rf-code-tooltip')).toContainText('frozen pretrained text encoder');
  await page.keyboard.press('Escape');
  await expect(page.locator('#rf-code-tooltip')).toBeHidden();
  await page.locator('[data-case-select="claims"]').click();
  const claims = explainer(page.locator('main'), 'claims');
  await expect(claims).toHaveAttribute('data-model-stage', 'start');
  await expect(claims.locator('[data-model-path]')).toHaveText('/');
  await page.locator('[data-case-select="tickets"]').click();
  await expect(root).toHaveAttribute('data-model-stage', 'context');
  await expect(root.locator('[data-model-path]')).toHaveText('/messages/body');
});

test('without JavaScript every complete schema and field explanation remains readable', async ({
  browser,
  baseURL,
}) => {
  const context = await browser.newContext({ javaScriptEnabled: false });
  const page = await context.newPage();
  try {
    await page.goto(`${baseURL}/concepts/models/`);
    await expect(page.locator('[data-model-explainer]:visible')).toHaveCount(6);
    for (const item of explainerCases) {
      const root = explainer(page.locator('main'), item.id);
      await expect(root.locator('.me-steps')).toBeHidden();
      await expect(root.locator('[data-model-select="/"]')).toBeDisabled();
      await root.locator('.me-source > summary').click();
      expect(await renderedSource(root)).toBe(explainerByCase[item.id].stages.at(-1)!.schema);
      await root.locator('.me-field-reference > summary').click();
      await expect(root.locator('.me-field-reference dl')).toBeVisible();
      const targets = flatten(item.stages.at(-1)!.nodes).filter((node) => node.target);
      for (const target of targets)
        await expect(
          root.locator('.me-field-reference dt').filter({ hasText: `${target.path} ·` }),
        ).toHaveCount(1);
    }
  } finally {
    await context.close();
  }
});
