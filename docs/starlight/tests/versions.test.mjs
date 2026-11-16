import assert from 'node:assert/strict';
import test from 'node:test';
import { catalog } from '../src/versions/catalog.ts';

const currentURL = 'https://docs.example.org/latest/concepts/models/';

test('Read the Docs owns version labels, ordering, and URLs, including custom domains', () => {
  const menu = catalog(
    {
      versions: {
        current: { slug: 'latest' },
        active: [
          {
            slug: 'stable',
            verbose_name: '0.5.0',
            urls: { documentation: 'https://www.relflow.ai/stable/' },
          },
          { slug: 'latest', urls: { documentation: 'https://www.relflow.ai/latest/' } },
          {
            slug: '102',
            type: 'external',
            urls: { documentation: 'https://relflow--102.org.readthedocs.build/102/' },
          },
        ],
      },
    },
    currentURL,
  );
  assert.equal(menu.current, 'latest');
  assert.deepEqual(
    menu.versions.map(({ id, badge }) => ({ id, badge })),
    [
      { id: 'stable', badge: 'Stable' },
      { id: 'latest', badge: 'Development' },
      { id: '102', badge: 'Preview' },
    ],
  );
  assert.equal(menu.versions[0].label, '0.5.0');
  assert.equal(menu.versions[1].url, 'https://www.relflow.ai/latest/');
  assert.equal(menu.versions[2].url, 'https://relflow--102.org.readthedocs.build/102/');
});

test('a current preview missing from active versions remains selectable at its actual URL', () => {
  const menu = catalog(
    {
      versions: {
        current: { slug: '102', type: 'external' },
        active: [{ slug: 'stable', urls: { documentation: 'https://www.relflow.ai/stable/' } }],
      },
    },
    currentURL,
  );
  assert.equal(menu.versions[1].id, '102');
  assert.equal(menu.versions[1].url, currentURL);
});

test('malformed Addons responses and unsafe URLs do not produce a usable replacement', () => {
  for (const value of [
    null,
    {},
    { versions: {} },
    { versions: { current: { slug: 42 }, active: [] } },
    {
      versions: {
        current: { slug: 'latest', urls: { documentation: 'javascript:alert(1)' } },
        active: [],
      },
    },
    {
      versions: {
        current: { slug: 'latest', urls: { documentation: 'https://user:password@example.org/' } },
        active: [],
      },
    },
  ])
    assert.equal(catalog(value, currentURL), undefined);
  const menu = catalog(
    {
      versions: {
        current: { slug: 'latest' },
        active: [null, { slug: 'invalid', urls: { documentation: 'data:text/html,bad' } }],
      },
    },
    currentURL,
  );
  assert.deepEqual(
    menu.versions.map((version) => version.id),
    ['latest'],
  );
});
