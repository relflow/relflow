import { test, expect } from '@playwright/test';
import { outputs } from '../../src/datatypes/predictions';

const leaves = [
  'number',
  'quantile',
  'vector',
  'coordinates',
  'ordinal',
  'category',
  'boolean',
  'enum',
  'hash',
  'set',
  'group',
  'period',
  'duration',
  'text',
  'image',
  'audio',
];
const headings = [
  'Input values',
  'Standardization and processor pipelines',
  'How is this leaf trained?',
  'Leaf-specific parameters',
  'Errors raised',
  'Relevant evidence',
];

interface Publication {
  version: string;
  results: Record<string, { mode: string; outcome: string }>;
}

test('all leaf references follow Number, show standalone output trees, and report selected publication outcomes', async ({
  page,
  request,
}) => {
  test.setTimeout(90_000);
  const response = await request.get('/evidence/selected.json');
  expect(response.ok()).toBeTruthy();
  const publication: Publication = await response.json();
  const { evidenceEntries } = await import('../../src/evidence/data');
  const errors: string[] = [];
  page.on('pageerror', (error) => errors.push(error.message));

  for (const leaf of leaves) {
    await test.step(leaf, async () => {
      const response = await page.goto(`/data-types/${leaf}/`);
      expect(response?.status()).toBe(200);
      const main = page.locator('main[data-pagefind-body]');
      await expect(main.locator('h2')).toHaveText(headings);
      for (const title of ['Predictions', 'Request methods', 'Footguns', 'Related types'])
        await expect(main.getByRole('heading', { level: 3, name: title, exact: true })).toHaveCount(
          1,
        );

      await expect(
        main.locator('[data-type-showcase], [data-example-select], [data-data-explorer]'),
      ).toHaveCount(0);
      const explanation = main
        .locator('#predictions')
        .locator('..')
        .locator('xpath=following-sibling::*[1]');
      await expect(explanation).toHaveJSProperty('tagName', 'P');
      await expect(explanation).toHaveText(/\S/);
      const tree = main.locator('[data-value-tree]');
      await expect(tree).toHaveCount(1);
      await expect(tree).toContainText('Illustrative prediction output');
      expect(JSON.parse((await tree.locator('[data-tree-source]').textContent())!)).toEqual(
        outputs[leaf as keyof typeof outputs].value,
      );
      await tree.getByRole('button', { name: 'Collapse all', exact: true }).click();
      await expect(tree.locator('details[open]')).toHaveCount(0);
      await tree.getByRole('button', { name: 'Expand all', exact: true }).click();
      await expect(tree.locator('details:not([open])')).toHaveCount(0);
      if (['hash', 'text', 'image', 'audio', 'period'].includes(leaf)) {
        await expect(tree.locator('[data-value-name="representations"]')).toHaveCount(1);
        await expect(tree.locator('[data-value-name="predictions"]')).toHaveCount(0);
        await expect(tree).toContainText('With embed=True');
      } else {
        await expect(tree.locator('[data-value-name="predictions"]')).toHaveCount(1);
        await expect(tree).toContainText('not answer confidence');
      }

      const evidence = main.locator('.leaf-evidence');
      await expect(
        evidence.getByRole('link', { name: `Publication ${publication.version}` }),
      ).toHaveAttribute('href', '/evidence/selected.json');
      await expect(evidence.locator('thead th')).toHaveText([
        'Proof ID',
        'Hypothesis being tested',
        'Status per release version',
      ]);
      const rows = evidence.locator('tr[data-proof]');
      const expectedIds = evidenceEntries
        .filter((entry) => entry.tags?.types.some((type) => type.toLowerCase() === leaf))
        .map((entry) => entry.id);
      expect(await rows.count()).toBe(expectedIds.length);
      expect(
        await rows.evaluateAll((elements) => elements.map((row) => row.getAttribute('data-proof'))),
      ).toEqual(expectedIds);
      if (!expectedIds.length)
        await expect(evidence).toContainText(
          'No dedicated learning experiment is declared for this leaf type.',
        );
      for (const row of await rows.all()) {
        const id = (await row.getAttribute('data-proof'))!;
        const record = publication.results[id];
        const expected = !record
          ? 'No selected run'
          : record.outcome === 'error'
            ? 'Execution error'
            : record.mode !== 'full'
              ? 'Limited run'
              : record.outcome === 'met'
                ? 'Checks met'
                : 'Checks unmet';
        await expect(row.locator('.result-status')).toHaveText(expected);
        await expect(row.getByRole('link', { name: id, exact: true })).toHaveAttribute(
          'href',
          `/evidence/${id}/`,
        );
      }
      expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBe(
        await page.evaluate(() => innerWidth),
      );
    });
  }
  expect(errors).toEqual([]);
});
