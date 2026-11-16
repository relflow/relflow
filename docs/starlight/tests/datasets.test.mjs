import assert from 'node:assert/strict';
import test from 'node:test';
import { layout } from '../src/datasets/types.ts';

const names = [
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
    names.map(async (name) => {
      const module = await import(`../src/data/examples/${name}.ts`);
      return module[`${name}Examples`].map((example) => ({ ...example, type: name }));
    }),
  )
).flat();

function sourceFields(record, fields = new Map(), prefix = '') {
  for (const [name, value] of Object.entries(record)) {
    const path = `${prefix}/${name}`;
    const items = Array.isArray(value) ? value : [value];
    const records = items.filter(
      (item) => item && typeof item === 'object' && !Array.isArray(item),
    );
    fields.set(path, records.length ? (Array.isArray(value) ? 'collection' : 'record') : 'value');
    for (const child of records) sourceFields(child, fields, path);
  }
  return fields;
}

/** Follow authored record membership, independently of rendered positions and example values. */
function describedFields(dataset) {
  const fields = new Map();
  const visited = new Set();
  function visit(entity, prefix = '') {
    assert.ok(
      !visited.has(entity.id),
      `${dataset.id}: observation contracts must form a record tree`,
    );
    visited.add(entity.id);
    for (const column of entity.columns) fields.set(`${prefix}/${column.name}`, column);
    for (const relation of dataset.relations.filter((item) => item.from === entity.id)) {
      const column = entity.columns.find((item) => item.name === relation.fromColumn);
      assert.ok(
        column && ['collection', 'record'].includes(column.role),
        `${dataset.id}: nested records need an enclosing source column`,
      );
      assert.equal(relation.cardinality ?? 'many', column.role === 'record' ? 'one' : 'many');
      visit(
        dataset.entities.find((item) => item.id === relation.to),
        `${prefix}/${column.name}`,
      );
    }
  }
  visit(dataset.entities[0]);
  assert.equal(
    visited.size,
    dataset.entities.length,
    `${dataset.id}: every described entity belongs to the observation`,
  );
  return fields;
}

test('all 39 type examples describe the exact source fields and nested record boundaries', () => {
  assert.equal(examples.length, 39);
  for (const example of examples) {
    const { dataset } = example;
    const label = `${example.type}/${example.id}`;
    assert.ok(
      dataset.label && dataset.description && dataset.observation,
      `${label}: describe the dataset and observation`,
    );
    layout(dataset);
    const inputs = sourceFields(example.inputs);
    const answers = sourceFields(example.answers);
    const expected = new Map([...inputs, ...answers]);
    const actual = describedFields(dataset);
    assert.deepEqual(
      [...actual.keys()].sort(),
      [...expected.keys()].sort(),
      `${label}: source columns must match inputs plus training-answer sources`,
    );
    for (const [path, kind] of expected) {
      const column = actual.get(path);
      assert.ok(
        column.type && column.description,
        `${label}${path}: provide a source type and column description`,
      );
      if (kind !== 'value')
        assert.equal(column.role, kind, `${label}${path}: preserve record nesting`);
      else
        assert.ok(
          !['collection', 'record'].includes(column.role),
          `${label}${path}: scalar arrays remain one value column`,
        );
      if (answers.has(path) && kind === 'value') {
        assert.ok(
          ['target', 'training'].includes(column.role),
          `${label}${path}: mark historical answer sources`,
        );
        assert.ok(
          !inputs.has(path),
          `${label}${path}: training answers must remain outside prediction inputs`,
        );
      }
      if (['target', 'training'].includes(column.role)) {
        assert.equal(
          answers.get(path),
          'value',
          `${label}${path}: do not invent or relabel a target outside its answer source`,
        );
        assert.ok(
          !inputs.has(path),
          `${label}${path}: do not mark a known input as a hidden answer`,
        );
      }
    }
  }
});

test('empty Set values and fixed-width Vector values stay columns rather than record collections', () => {
  const scalarArrays = examples.flatMap((example) => {
    const fields = describedFields(example.dataset);
    return [...fields]
      .filter(([, column]) => /\[\]$/.test(column.type) && column.role !== 'collection')
      .map(([path, column]) => ({ example, path, column }));
  });
  assert.ok(scalarArrays.some(({ example }) => example.type === 'set'));
  assert.ok(scalarArrays.some(({ example }) => example.type === 'vector'));
  assert.ok(
    examples.some((example) => JSON.stringify(example.answers).includes('[]')),
    'Keep an empty primitive-list fixture in the authored examples.',
  );
  for (const { example, path, column } of scalarArrays) {
    assert.equal(
      new Map([...sourceFields(example.inputs), ...sourceFields(example.answers)]).get(path),
      'value',
    );
    assert.ok(!['collection', 'record'].includes(column.role));
  }
});

test('Duration endpoints describe training-only sources rather than fabricated numeric source columns', () => {
  const example = examples.find((item) => item.type === 'duration' && item.id === 'repair-target');
  assert.ok(example);
  const fields = describedFields(example.dataset);
  assert.equal(fields.get('/opened_at').role, 'training');
  assert.equal(fields.get('/resolved_at').role, 'training');
  assert.ok(
    !fields.has('/resolution_hours'),
    'The raw source supplies two timestamps, while the model derives its duration target.',
  );
  assert.ok(
    'resolution_hours' in example.output.predictions,
    'The prediction still returns the derived numeric target.',
  );
  const elapsed = examples.find((item) => item.type === 'duration' && item.id === 'account-age');
  const elapsedFields = describedFields(elapsed.dataset);
  assert.ok(!elapsedFields.get('/created_at').role);
  assert.ok(!elapsedFields.get('/reviewed_at').role);
  assert.ok(!elapsedFields.has('/account_age_days'));
});

function graph(ids, relations = []) {
  return {
    id: 'layout-fixture',
    label: 'Relationships',
    observation: 'One graph.',
    description: 'A static layout fixture.',
    entities: ids.map((id) => ({
      id,
      name: id,
      description: id,
      columns: [{ name: 'id', type: 'string', description: 'Source key.' }],
    })),
    relations: relations.map(([from, to]) => ({
      from,
      to,
      description: `${from} relates to ${to}.`,
    })),
  };
}

test('a shared child, cycle, and disconnected table are each rendered exactly once', () => {
  const dataset = graph(
    ['root', 'left', 'right', 'shared', 'cycle-a', 'cycle-b', 'standalone'],
    [
      ['root', 'left'],
      ['root', 'right'],
      ['left', 'shared'],
      ['right', 'shared'],
      ['cycle-a', 'cycle-b'],
      ['cycle-b', 'cycle-a'],
    ],
  );
  const actual = layout(dataset)
    .flat()
    .map((entity) => entity.id);
  assert.equal(actual.length, dataset.entities.length);
  assert.equal(new Set(actual).size, dataset.entities.length);
  assert.deepEqual([...actual].sort(), dataset.entities.map((entity) => entity.id).sort());
});

test('bad relation endpoints and duplicate entity IDs fail before a diagram can omit data', () => {
  assert.throws(() => layout(graph(['root'], [['root', 'missing']])), /unknown entity/);
  assert.throws(() => layout(graph(['root'], [['missing', 'root']])), /unknown entity/);
  assert.throws(() => layout(graph(['root', 'root'])), /unique entity IDs/);
  assert.throws(() => layout(graph([])), /at least one entity/);
});

test('relation column anchors must exist on the specified entity', () => {
  for (const anchor of ['fromColumn', 'toColumn']) {
    const dataset = graph(['root', 'child'], [['root', 'child']]);
    dataset.relations[0][anchor] = 'missing';
    assert.throws(() => layout(dataset), /unknown column/);
    dataset.relations[0][anchor] = 'id';
    assert.equal(layout(dataset).flat().length, 2);
  }
});
