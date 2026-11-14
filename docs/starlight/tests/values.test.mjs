import assert from 'node:assert/strict';
import test from 'node:test';
import { formatValue, magnitudes, percent, predictionView, tree } from '../src/values/tree.ts';

const types = [
  'boolean',
  'category',
  'duration',
  'enum',
  'group',
  'hash',
  'number',
  'ordinal',
  'period',
  'quantile',
  'set',
  'text',
  'vector',
];
const examples = (
  await Promise.all(
    types.map(async (type) => {
      const module = await import(`../src/data/examples/${type}.ts`);
      return module[`${type}Examples`].map((example) => ({ ...example, type }));
    }),
  )
).flat();

function nodes(node) {
  return [node, ...node.children.flatMap(nodes), ...(node.returned || []).flatMap(nodes)];
}

/** Rebuild from display nodes, rather than relying on the retained original reference. */
function restore(node) {
  if (node.kind === 'prediction')
    return Object.fromEntries(node.returned.map((child) => [child.name, restore(child)]));
  if (node.kind === 'record')
    return Object.fromEntries(node.children.map((child) => [child.name, restore(child)]));
  if (node.kind === 'list') return node.children.map(restore);
  return node.value;
}

function rich(content, extra = {}) {
  return {
    state: { valued: 0.96, null: 0.01, padded: 0.01, masked: 0.01, other: 0.01 },
    content,
    inferred: true,
    ...extra,
  };
}

function prediction(content, extra = {}) {
  return tree(
    { predictions: { answer: rich(content, extra) } },
    { id: 'fixture', predictions: true },
  ).children[0].children[0];
}

test('all 39 authored output rows retain all 59 rich leaves and every returned field', () => {
  assert.equal(examples.length, 39);
  const kinds = {};
  for (const example of examples) {
    const before = JSON.stringify(example.output);
    const inputsBefore = JSON.stringify(example.inputs);
    const root = tree(example.output, {
      id: `${example.type}-${example.id}`,
      predictions: true,
      inputs: example.inputs,
    });
    assert.deepEqual(
      restore(root),
      example.output,
      `${example.type}/${example.id}: display must retain the complete output row`,
    );
    const flattened = nodes(root);
    assert.equal(
      new Set(flattened.map((node) => node.id)).size,
      flattened.length,
      'Display nodes and returned-value inspectors need distinct IDs.',
    );
    for (const node of flattened.filter((node) => node.kind === 'prediction')) {
      assert.deepEqual(
        node.returned.map((child) => child.name),
        Object.keys(node.value),
      );
      kinds[node.view.kind] = (kinds[node.view.kind] || 0) + 1;
    }
    assert.equal(
      JSON.stringify(example.output),
      before,
      'Visualization must not mutate recorded example outputs.',
    );
    assert.equal(
      JSON.stringify(example.inputs),
      inputsBefore,
      'Input-context annotations must not mutate the supplied record.',
    );
  }
  assert.deepEqual(kinds, {
    probability: 21,
    category: 8,
    scalar: 23,
    memberships: 3,
    empty: 1,
    vector: 3,
  });
});

test('numeric and string ordinal labels remain scalar values without invented confidence', () => {
  assert.deepEqual(predictionView(20), { kind: 'scalar', value: 20 });
  assert.deepEqual(predictionView('high'), { kind: 'scalar', value: 'high' });
  assert.deepEqual(predictionView('17'), { kind: 'scalar', value: '17' });
  assert.deepEqual(predictionView(-1), { kind: 'scalar', value: -1 });
  assert.equal(
    prediction(1190).view.value,
    1190,
    'Quantile outputs remain source-unit scalars rather than percentiles.',
  );
});

test('Boolean event probability stays separate from the field-state distribution', () => {
  const node = prediction({ probability: 0.12 });
  assert.deepEqual(node.view, { kind: 'probability', probability: 0.12 });
  assert.equal(node.value.state.valued, 0.96);
  assert.equal(
    node.returned
      .find((child) => child.name === 'state')
      .children.find((child) => child.name === 'valued').value,
    0.96,
  );
  assert.equal(node.returned.find((child) => child.name === 'inferred').value, true);
});

test('Set membership scores and partial top-k probabilities are never renormalized', () => {
  const members = [
    { value: 'books', probability: 0.61 },
    { value: 'home', probability: 0.87 },
  ];
  const memberships = prediction(members);
  assert.equal(memberships.view.kind, 'memberships');
  assert.deepEqual(memberships.view.candidates, members);
  assert.equal(
    memberships.view.candidates.reduce((sum, candidate) => sum + candidate.probability, 0),
    1.48,
  );
  const partial = {
    value: '17',
    probability: 0.45,
    topk: [
      { value: '17', probability: 0.45 },
      { value: '42', probability: 0.2 },
    ],
  };
  const classification = prediction(partial);
  assert.equal(classification.view.kind, 'category');
  assert.equal(classification.view.value, '17');
  assert.equal(typeof classification.view.value, 'string');
  assert.deepEqual(classification.view.candidates, partial.topk);
  assert.equal(classification.view.probability, 0.45);
});

test('empty Set content is distinct from a vector, an empty branch, and a null record', () => {
  assert.deepEqual(predictionView([]), { kind: 'empty' });
  assert.deepEqual(predictionView([11.8, 19.2, 5.1]), {
    kind: 'vector',
    values: [11.8, 19.2, 5.1],
  });
  assert.deepEqual(predictionView(null), { kind: 'scalar', value: null });
  const row = {
    predictions: { claims: [null, { checks: rich([]) }], empty_history: [], no_history: null },
  };
  const root = tree(row, { id: 'nulls', predictions: true });
  assert.deepEqual(restore(root), row);
  const fields = root.children[0].children;
  assert.equal(fields.find((node) => node.name === 'empty_history').kind, 'list');
  assert.equal(fields.find((node) => node.name === 'empty_history').children.length, 0);
  assert.equal(fields.find((node) => node.name === 'no_history').value, null);
  assert.equal(fields.find((node) => node.name === 'claims').children[0].value, null);
});

test('rich-looking ordinary records are special only inside an opted-in predictions plane', () => {
  const value = rich({ probability: 0.8 });
  const row = {
    inputs: { sample: value },
    representations: [value],
    predictions: { sample: value },
  };
  const generic = tree(row, { id: 'generic' });
  assert.equal(nodes(generic).filter((node) => node.kind === 'prediction').length, 0);
  assert.deepEqual(restore(generic), row);
  const decorated = tree(row, { id: 'decorated', predictions: true });
  assert.equal(nodes(decorated).filter((node) => node.kind === 'prediction').length, 1);
  assert.deepEqual(restore(decorated), row);
  assert.equal(tree(value, { id: 'root-value', predictions: true }).kind, 'record');
});

test('unrecognized extension content and extra returned fields remain lossless', () => {
  const content = {
    interval: [1.25, 4.75],
    method: 'extension-owned',
    detail: { empty: '', missing: null },
  };
  const node = prediction(content, {
    inferred: false,
    provenance: { version: 7 },
    diagnostics: [null, false, 0],
  });
  assert.equal(node.view.kind, 'structured');
  assert.equal(node.value.inferred, false);
  assert.deepEqual(restore(node), node.value);
  assert.deepEqual(node.returned.find((child) => child.name === 'content').value, content);
  const additional = prediction({ value: 'label', probability: 0.75, extension_note: 'retain me' });
  assert.deepEqual(restore(additional).content, {
    value: 'label',
    probability: 0.75,
    extension_note: 'retain me',
  });
});

test('input context follows nested record positions without becoming returned predictions', () => {
  const inputs = {
    groups: [
      {
        site: 'north',
        items: [
          { sku: 'notebook', on_hand: 20 },
          { sku: 'pencil', on_hand: 80 },
        ],
      },
    ],
  };
  const row = {
    predictions: {
      groups: [{ items: [{ demand: rich(8.6) }, { demand: rich(4.2) }, { demand: rich(1) }] }],
    },
  };
  const before = JSON.stringify({ inputs, row });
  const root = tree(row, { id: 'alignment', predictions: true, inputs });
  const group = root.children[0].children[0].children[0];
  assert.ok(group.context.includes('site: "north"'));
  const items = group.children[0].children;
  assert.ok(items[0].context.includes('sku: "notebook"'));
  assert.ok(items[1].context.includes('sku: "pencil"'));
  assert.ok(!items[0].context.includes('pencil'));
  assert.equal(
    items[2].context,
    undefined,
    'An unmatched position must not inherit another record’s context.',
  );
  assert.deepEqual(
    restore(root),
    row,
    'Display context does not add retained inputs to the actual prediction row.',
  );
  assert.equal(JSON.stringify({ inputs, row }), before);
});

test('value formatting preserves native types and percentage precision', () => {
  assert.equal(formatValue('17'), '"17"');
  assert.equal(formatValue(17), '17');
  assert.equal(formatValue(''), '""');
  assert.equal(formatValue(null), 'null');
  assert.equal(formatValue(false), 'false');
  assert.equal(formatValue('line\nnext'), '"line\\nnext"');
  assert.equal(percent(0.123456789), '12.3457%');
  assert.equal(percent(0), '0%');
  assert.equal(percent(1), '100%');
  assert.deepEqual(
    magnitudes([0, 0, 0]),
    [0, 0, 0],
    'A zero vector has no relative magnitude rather than NaN.',
  );
  assert.deepEqual(magnitudes([0, -10, 5]), [0, 100, 50]);
});

test('large vector bars require linear input work and preserve the source values', () => {
  const values = Object.freeze(
    Array.from({ length: 16_384 }, (_, index) => (index % 2 ? -index : index)),
  );
  let reads = 0;
  const observed = new Proxy(values, {
    get(target, property, receiver) {
      if (typeof property === 'string' && /^\d+$/.test(property)) reads++;
      return Reflect.get(target, property, receiver);
    },
  });
  const bars = magnitudes(observed);
  assert.equal(bars.length, values.length);
  assert.equal(bars[0], 0);
  assert.equal(bars.at(-1), 100);
  assert.ok(
    reads <= values.length * 3,
    `Vector rendering must stay linear; read ${reads} coordinates.`,
  );
  assert.equal(values.at(-1), -16_383);
});
