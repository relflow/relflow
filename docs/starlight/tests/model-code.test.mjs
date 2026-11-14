import assert from 'node:assert/strict';
import test from 'node:test';
import { ExpressiveCodeEngine } from '@expressive-code/core';
import { configNotes } from '../src/code/config-notes.mjs';
import { codePlan, modelCode } from '../src/model-explainer/code.mjs';
import { explainerCases } from '../src/data/model-explainer.ts';

const stages = [
  {
    id: 'start',
    schema:
      'import relflow as rf\n\nclass Customer(rf.Model):\n    amount = rf.Number\n    returned = rf.Boolean(mask=True)',
    lines: { '/amount': 4, '/returned': 5 },
  },
  {
    id: 'context',
    schema:
      'import relflow as rf\n\nclass Order(rf.Branch):\n    amount = rf.Number\n\nclass Customer(rf.Model):\n    amount = rf.Number\n    orders = Order(length=8)\n    returned = rf.Boolean(mask=True)',
    lines: { '/orders/amount': 4, '/amount': 7, '/orders': 8, '/returned': 9 },
  },
  {
    id: 'targets',
    schema:
      'import relflow as rf\n\nclass Order(rf.Branch):\n    amount = rf.Number\n\nclass Customer(rf.Model):\n    amount = rf.Number\n    orders = Order(length=8)\n    returned = rf.Boolean(mask=True)\n    refund = rf.Number(mask=True)',
    lines: { '/orders/amount': 4, '/amount': 7, '/orders': 8, '/returned': 9, '/refund': 10 },
  },
];

function text(node) {
  return node.type === 'text' ? node.value : (node.children || []).map(text).join('');
}

function descendants(node) {
  return [node, ...(node.children || []).flatMap(descendants)];
}

test('staged source plans retain repeated blank lines and repeated declarations at their own addresses', () => {
  const plan = codePlan(stages);
  const full = stages.at(-1).schema.split('\n');
  for (const stage of stages) {
    assert.equal(
      full.filter((_, index) => plan[index].stages.includes(stage.id)).join('\n'),
      stage.schema,
    );
  }
  assert.deepEqual(plan[3].paths, ['/orders/amount']);
  assert.deepEqual(plan[6].paths, ['/amount']);
  assert.deepEqual(plan[9], { stages: ['targets'], paths: ['/refund'] });
});

test('multiple placements may share one physical class declaration without losing schema addresses', () => {
  const stage = {
    id: 'full',
    schema: 'class Item(rf.Branch):\n    amount = rf.Number',
    lines: { '/history/amount': 2, '/candidates/amount': 2 },
  };
  assert.deepEqual(codePlan([stage])[1].paths, ['/history/amount', '/candidates/amount']);
});

test('one rendered code block exposes line stage/path tags while retaining code notes and exact source', async () => {
  const engine = new ExpressiveCodeEngine({ plugins: [configNotes(), modelCode()] });
  const code = stages.at(-1).schema;
  const plan = codePlan(stages);
  const { renderedGroupAst, renderedGroupContents } = await engine.render({
    code,
    language: 'python',
    meta: `contract="schema" model-lines='${JSON.stringify(plan)}'`,
  });
  assert.equal(renderedGroupContents[0].codeBlock.code, code);
  const nodes = descendants(renderedGroupAst);
  const lines = nodes.filter((node) => node.properties?.className?.includes('ec-line'));
  assert.equal(lines.length, plan.length);
  for (let index = 0; index < lines.length; index++) {
    assert.equal(lines[index].properties['data-model-stages'], plan[index].stages.join(' '));
    assert.equal(
      lines[index].properties['data-model-paths'],
      plan[index].paths.length ? plan[index].paths.join(' ') : undefined,
    );
  }
  assert.deepEqual(
    lines.map((line) => {
      const value = text(line);
      return value === '\n' ? '' : value;
    }),
    code.split('\n'),
  );
  assert.ok(
    nodes.some((node) => node.properties?.className?.includes('rf-code-note')),
    'The model map must keep the common contextual notes.',
  );
});

test('ordinary code blocks do not receive model-map attributes', async () => {
  const engine = new ExpressiveCodeEngine({ plugins: [modelCode()] });
  const { renderedGroupAst } = await engine.render({ code: 'answer = 7', language: 'python' });
  assert.ok(descendants(renderedGroupAst).every((node) => !node.properties?.['data-model-stages']));
});

test('all six use cases render every stage from the same highlighted complete schema', async () => {
  const engine = new ExpressiveCodeEngine({ plugins: [configNotes(), modelCode()] });
  assert.equal(explainerCases.length, 6);
  for (const item of explainerCases) {
    const complete = item.stages.at(-1);
    const plan = codePlan(item.stages);
    const { renderedGroupAst } = await engine.render({
      code: complete.schema,
      language: 'python',
      meta: `contract="schema" model-lines='${JSON.stringify(plan)}'`,
    });
    const lines = descendants(renderedGroupAst).filter((node) =>
      node.properties?.className?.includes('ec-line'),
    );
    for (const stage of item.stages) {
      const visible = lines.filter((line) =>
        line.properties['data-model-stages'].split(' ').includes(stage.id),
      );
      assert.equal(
        visible
          .map((line) => {
            const value = text(line);
            return value === '\n' ? '' : value;
          })
          .join('\n'),
        stage.schema,
        `${item.id}/${stage.id}: rendered stage must preserve its complete source.`,
      );
    }
  }
});

test('invalid stage source or line mappings fail before partial line plans can hide declarations', () => {
  assert.throws(() => codePlan([]), /at least one/);
  assert.throws(() => codePlan([stages[0], { ...stages[0] }]), /unique/);
  assert.throws(
    () => codePlan([{ ...stages[0], schema: 'not present' }, stages.at(-1)]),
    /ordered subset/,
  );
  assert.throws(
    () => codePlan([{ ...stages[0], lines: { '/missing': 4 } }, stages.at(-1)]),
    /does not retain/,
  );
  assert.throws(
    () => codePlan([{ ...stages[0], lines: { '/amount': 0 } }, stages.at(-1)]),
    /source line/,
  );
  assert.throws(
    () => codePlan([{ ...stages[0], lines: { '/amount': 5 } }, stages.at(-1)]),
    /does not retain/,
  );
});

test('invalid rendered metadata fails instead of attaching an incomplete or ambiguous plan', async (t) => {
  t.mock.method(console, 'error', () => {});
  const engine = new ExpressiveCodeEngine({ plugins: [modelCode()] });
  for (const metadata of [
    'no-json',
    '[]',
    JSON.stringify([{ stages: [], paths: [] }]),
    JSON.stringify([{ stages: ['start'], paths: ['relative'] }]),
  ]) {
    await assert.rejects(
      engine.render({
        code: 'value = rf.Number',
        language: 'python',
        meta: `model-lines='${metadata}'`,
      }),
      /Model source/,
    );
  }
});
