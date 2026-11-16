import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import { explainerCases, explainerByCase } from '../src/data/model-explainer.ts';
import { scenarios } from '../src/data/concepts/scenarios.ts';
import { flatten, parseSchema, projectSchema } from '../src/model-explainer/schema.ts';
import { layout } from '../src/datasets/types.ts';
import { bearingSchema, repeatPurchaseSchema } from '../src/data/schema-examples.ts';

const originals = JSON.parse(
  readFileSync(new URL('../src/data/use-cases.json', import.meta.url), 'utf8'),
);
const extraTargets = {
  churn: '/retained_revenue_90d',
  credit: '/loss_12m',
  maintenance: '/repair_cost_30d',
  demand: '/horizon/stockout',
};

function contract(nodes) {
  return flatten(nodes).map(({ path, kind, type, owner, declaration, target }) => ({
    path,
    kind,
    type,
    owner,
    declaration,
    target,
  }));
}

function sourceColumns(dataset) {
  const result = new Map();
  const visit = (entityId, prefix) => {
    const entity = dataset.entities.find((entity) => entity.id === entityId);
    for (const column of entity.columns) result.set(`${prefix}/${column.name}`, column);
    for (const relation of dataset.relations.filter((relation) => relation.from === entityId))
      visit(relation.to, `${prefix}/${relation.fromColumn}`);
  };
  visit(dataset.entities[0].id, '');
  return result;
}

test('every original use case retains its canonical field declarations and record boundaries', () => {
  assert.deepEqual(
    explainerCases.map(({ id }) => id),
    originals.map(({ id }) => id),
  );
  for (const original of originals) {
    const parsed = parseSchema(original.code);
    assert.equal(
      projectSchema(parsed, parsed.nodes).schema,
      original.code,
      `${original.id}: round-trip canonical class declarations exactly`,
    );
    const example = explainerByCase[original.id];
    const full = example.stages.at(-1);
    assert.equal(example.modelClass, parsed.modelClass);
    const canonicalPaths = new Set(flatten(parsed.nodes).map(({ path }) => path));
    assert.deepEqual(
      contract(full.nodes).filter(({ path }) => canonicalPaths.has(path)),
      contract(parsed.nodes),
    );
    assert.ok(flatten(full.nodes).every(({ description }) => description.length > 0));
    layout(example.dataset);
  }
});

test('all three stages are monotonic Python-schema projections with correct source-line bindings', () => {
  for (const example of explainerCases) {
    assert.deepEqual(
      example.stages.map(({ id }) => id),
      ['start', 'context', 'targets'],
    );
    const fullLines = example.stages.at(-1).schema.split('\n');
    let previous = new Set();
    for (const stage of example.stages) {
      const nodes = flatten(stage.nodes);
      const current = new Set(nodes.map(({ path }) => path));
      for (const path of previous)
        assert.ok(current.has(path), `${example.id}/${stage.id}: preserve earlier node ${path}`);
      previous = current;
      const parsed = parseSchema(stage.schema);
      assert.deepEqual(contract(parsed.nodes), contract(stage.nodes));
      const lines = stage.schema.split('\n');
      assert.equal(lines[stage.lines['/'] - 1], `class ${example.modelClass}(rf.Model):`);
      for (const node of nodes)
        assert.equal(
          lines[stage.lines[node.path] - 1],
          `    ${node.name} = ${node.declaration}`,
          `${example.id}/${stage.id}${node.path}: bind its own class declaration`,
        );
      let cursor = 0;
      for (const line of lines) {
        while (cursor < fullLines.length && fullLines[cursor] !== line) cursor++;
        assert.ok(
          cursor < fullLines.length,
          `${example.id}/${stage.id}: the complete stage must remain a line subsequence of the final schema`,
        );
        cursor++;
      }
    }
  }
});

test('early stages preserve target ancestor branches while adding context never moves the answer', () => {
  for (const example of explainerCases) {
    const scenario = scenarios.find(({ id }) => id === example.id);
    for (const stage of example.stages.slice(0, 2))
      assert.deepEqual(
        flatten(stage.nodes)
          .filter(({ target }) => target)
          .map(({ path }) => path),
        [scenario.primaryTarget],
      );
    const startBranches = flatten(example.stages[0].nodes)
      .filter(({ kind }) => kind === 'branch')
      .map(({ path }) => path);
    const expected = scenario.primaryTarget
      .split('/')
      .slice(1, -1)
      .map((_, index, parts) => `/${parts.slice(0, index + 1).join('/')}`);
    assert.deepEqual(
      startBranches,
      expected,
      `${example.id}: only the primary target's enclosing records remain before adding context`,
    );
    const contexts = flatten(example.stages[1].nodes).filter(({ kind }) => kind === 'branch');
    assert.ok(
      contexts.length > startBranches.length,
      `${example.id}: the second stage adds record context`,
    );
    assert.equal(flatten(example.stages[2].nodes).filter(({ target }) => target).length, 2);
  }
});

test('additional example targets have explicit new-label requirements and matching dataset columns', () => {
  for (const example of explainerCases) {
    const full = example.stages.at(-1);
    const added = flatten(full.nodes).filter(({ added }) => added);
    const expected = extraTargets[example.id];
    assert.deepEqual(
      added.map(({ path }) => path),
      expected ? [expected] : [],
    );
    const columns = sourceColumns(example.dataset);
    assert.deepEqual(
      [...columns.keys()].sort(),
      flatten(full.nodes)
        .map(({ path }) => path)
        .sort(),
    );
    assert.ok(full.labelRequirement?.length > 40);
    for (const node of added) {
      assert.equal(columns.get(node.path).role, 'target');
      assert.equal(columns.get(node.path).description, node.description);
      assert.match(node.description, /Additional walkthrough label/);
      assert.ok(node.labelRequirement.includes(node.name) || node.path === '/horizon/stockout');
      assert.ok(
        example.stages
          .slice(0, 2)
          .every((stage) => !flatten(stage.nodes).some(({ path }) => path === node.path)),
      );
    }
    const original = sourceColumns(scenarios.find(({ id }) => id === example.id).dataset);
    if (expected)
      assert.ok(
        !original.has(expected),
        `${example.id}: illustrative new outcomes must not mutate the shared canonical dataset`,
      );
  }
});

test('literal string contents cannot create a target and branch placement options round-trip', () => {
  const source = `import relflow as rf


class Item(rf.Branch):
    """One supplied item."""

    label = rf.Enum("mask=True", "ordinary")


class Example(rf.Model):
    """An observation with item context."""

    items = Item(length=8, overflow="tail")
    result = rf.Boolean(mask=True)


if __name__ == "__main__":
    model = Example(preset="small")`;
  const parsed = parseSchema(source);
  assert.deepEqual(
    flatten(parsed.nodes)
      .filter(({ target }) => target)
      .map(({ path }) => path),
    ['/result'],
  );
  assert.equal(projectSchema(parsed, parsed.nodes).schema, source);
});

test('unsupported class bodies, recursive branches, and unknown references fail instead of omitting nodes', () => {
  const source = originals[0].code;
  for (const malformed of [
    source.replace('    product = rf.Category', '    product: str'),
    source.replace('    product = rf.Category', '    product = make_field()'),
    source.replace('messages = TicketMessage(', 'messages = UnknownBranch('),
    source.replace('body = rf.Text', 'body = rf.Text(option=custom())'),
    source.replace('body = rf.Text', 'body = TicketMessage'),
    source.replace(
      '    product = rf.Category',
      '    product = rf.Category\n    product = rf.Number',
    ),
  ])
    assert.throws(() => parseSchema(malformed), /Schema|schema|branch/);
});

test('declaration-only schemas render without constructing a model', () => {
  const parsed = parseSchema(repeatPurchaseSchema);
  assert.equal(parsed.modelClass, 'Customer');
  assert.deepEqual(
    flatten(parsed.nodes)
      .filter((node) => node.target)
      .map((node) => node.path),
    ['/returns_in_30d'],
  );
});

test('custom leaf names require explicit declarations and remain distinct from branches', () => {
  assert.throws(() => parseSchema(bearingSchema), /schema line/);
  const parsed = parseSchema(bearingSchema, { leaves: ['Bearing'] });
  assert.equal(parsed.nodes[0].type, 'Bearing');
  assert.equal(parsed.nodes[0].kind, 'field');
  assert.equal(parsed.nodes[0].target, false);
  assert.equal(parsed.nodes.at(-1).target, true);
});

test('all retained leaf example schemas can supply the shared model view', async () => {
  const types = [
    'number',
    'quantile',
    'vector',
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
  ];
  let count = 0;
  for (const type of types) {
    const module = await import(`../src/data/examples/${type}.ts`);
    for (const example of module[`${type}Examples`]) {
      assert.ok(parseSchema(example.schema).nodes.length, `${type}/${example.id}`);
      count++;
    }
  }
  assert.equal(count, 39);
});
