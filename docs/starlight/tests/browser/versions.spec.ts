import { test, expect } from '@playwright/test';

const data = {
  versions: {
    current: { slug: 'latest' },
    active: [
      { slug: 'latest', urls: { documentation: 'https://www.relflow.ai/latest/' } },
      {
        slug: 'stable',
        verbose_name: '0.5.0',
        urls: { documentation: 'https://www.relflow.ai/stable/' },
      },
      { slug: 'v0.4.7', urls: { documentation: 'https://www.relflow.ai/v0.4.7/' } },
    ],
  },
};

test('the version dialog consumes host data, filters releases, and supports keyboard dismissal', async ({
  page,
}) => {
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));
  await page.goto('/concepts/models/');
  const root = page.locator('[data-version-switcher]');
  await expect(root).toBeHidden();
  await expect(page.locator('meta[name="readthedocs-addons-api-version"]')).toHaveAttribute(
    'content',
    '1',
  );
  await page.evaluate((payload) => {
    const native = document.createElement('readthedocs-flyout');
    native.textContent = 'Native version picker';
    document.body.append(native);
    document.dispatchEvent(
      new CustomEvent('readthedocs-addons-data-ready', { detail: { data: () => payload } }),
    );
  }, data);
  await expect(root).toBeVisible();
  await expect(page.locator('readthedocs-flyout')).toBeHidden();
  const trigger = root.getByRole('button', { name: 'Documentation version: latest' });
  await trigger.click();
  const dialog = root.getByRole('dialog', { name: 'Switch documentation' });
  await expect(dialog).toBeVisible();
  const search = dialog.getByRole('searchbox', { name: 'Find a version' });
  await expect(search).toBeFocused();
  await search.fill('0.5');
  await expect(dialog.getByRole('link')).toHaveCount(1);
  await expect(dialog.getByRole('link')).toHaveAttribute('href', 'https://www.relflow.ai/stable/');
  await search.fill('missing');
  await expect(dialog.locator('[data-version-empty]')).toBeVisible();
  await search.fill('');
  await page.keyboard.press('ArrowDown');
  await expect(dialog.getByRole('link').first()).toBeFocused();
  await page.keyboard.press('End');
  await expect(dialog.getByRole('link').last()).toBeFocused();
  await page.keyboard.press('Escape');
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();
  await trigger.click();
  await expect(search).toHaveValue('');
  const bounds = await dialog.boundingBox();
  expect(bounds!.x).toBeGreaterThanOrEqual(0);
  expect(bounds!.x + bounds!.width).toBeLessThanOrEqual(page.viewportSize()!.width);
  await dialog.getByRole('button', { name: 'Close version picker' }).click();
  await expect(dialog).toBeHidden();
  await expect(trigger).toBeFocused();
  expect(errors).toEqual([]);
});

test('cached host data and later Addons updates keep one current version list', async ({
  page,
}) => {
  await page.addInitScript((payload) => {
    window.ReadTheDocsEventData = { data: () => payload };
  }, data);
  await page.goto('/');
  const root = page.locator('[data-version-switcher]');
  await expect(root).toBeVisible();
  await page.evaluate(
    (payload) =>
      document.dispatchEvent(
        new CustomEvent('readthedocs-addons-data-ready', {
          detail: {
            data: () => ({
              ...payload,
              versions: { ...payload.versions, current: { slug: 'stable' } },
            }),
          },
        }),
      ),
    data,
  );
  await root.getByRole('button', { name: 'Documentation version: 0.5.0' }).click();
  await expect(root.getByRole('link')).toHaveCount(3);
  await expect(root.locator('a[aria-current]')).toHaveText(/0.5.0/);
});

test('unavailable or malformed host data leaves the native flyout available', async ({ page }) => {
  await page.goto('/');
  await page.evaluate(() => {
    const native = document.createElement('readthedocs-flyout');
    native.textContent = 'Native version picker';
    document.body.append(native);
    for (const payload of [null, { versions: {} }])
      document.dispatchEvent(
        new CustomEvent('readthedocs-addons-data-ready', { detail: { data: () => payload } }),
      );
  });
  await expect(page.locator('[data-version-switcher]')).toBeHidden();
  await expect(page.locator('readthedocs-flyout')).toBeVisible();
});
