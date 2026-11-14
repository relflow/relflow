import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import test from 'node:test';
import { flowPaths } from '../src/data-flow/types.ts';
import { scenarios } from '../src/data/concepts/scenarios.ts';
import { predictions } from '../src/data/concepts/predictions.ts';
import { flatten, parseSchema } from '../src/model-explainer/schema.ts';

const sourceRecords = structuredClone(
  scenarios.map(({ id, inputs, answers }) => ({ id, inputs, answers })),
);
const sourcePredictions = structuredClone(predictions.examples);
const { flowCases } = await import('../src/data/concepts/data-flow.ts');

const paths = {
  predict: ['load', 'preprocess', 'prepare', 'model', 'write', 'postprocess', 'result'],
  train: ['load', 'preprocess', 'prepare', 'model', 'loss', 'backward', 'update'],
};

function stage(example, id) {
  const selected = example.stages.find((item) => item.id === id);
  assert.ok(selected, `${example.id}: missing ${id} stage`);
  return selected;
}

function project(value, shape, path = '/') {
  if (Array.isArray(shape)) {
    assert.ok(Array.isArray(value), `${path}: retain the supplied collection`);
    assert.equal(value.length, shape.length, `${path}: retain the supplied record count`);
    return shape.map((item, index) => project(value[index], item, `${path}[${index}]/`));
  }
  if (shape !== null && typeof shape === 'object') {
    assert.ok(value !== null && typeof value === 'object' && !Array.isArray(value), path);
    return Object.fromEntries(
      Object.entries(shape).map(([key, child]) => {
        assert.ok(Object.hasOwn(value, key), `${path}${key}: missing supplied field`);
        return [key, project(value[key], child, `${path}${key}/`)];
      }),
    );
  }
  return value;
}

function valuePaths(value, path = '') {
  if (Array.isArray(value) && value.length)
    return value.flatMap((child, index) => valuePaths(child, `${path}/${index}`));
  if (value !== null && typeof value === 'object' && Object.keys(value).length)
    return Object.entries(value).flatMap(([key, child]) => valuePaths(child, `${path}/${key}`));
  return [path];
}

function treePaths(nodes, prefix = '') {
  return nodes.flatMap((node) => {
    const path = `${prefix}/${node.label}`;
    return [{ path, node }, ...treePaths(node.children || [], path)];
  });
}

function readoutPaths(value, path = '/predictions', lists = new Set()) {
  if (Array.isArray(value)) {
    lists.add(path);
    return value.flatMap((child) => readoutPaths(child, path, lists));
  }
  assert.ok(value !== null && typeof value === 'object', `${path}: expected rich output`);
  if (['state', 'content', 'inferred'].every((key) => Object.hasOwn(value, key))) return [path];
  return Object.entries(value).flatMap(([key, child]) =>
    readoutPaths(child, `${path}/${key}`, lists),
  );
}

test('all six cases expose complete, separate prediction and training paths', () => {
  assert.deepEqual(flowPaths, paths);
  assert.deepEqual(
    flowCases.map(({ id }) => id),
    scenarios.map(({ id }) => id),
  );
  assert.equal(flowCases.length, 6);
  for (const example of flowCases) {
    assert.equal(example.stages.length, 10, example.id);
    assert.equal(new Set(example.stages.map(({ id }) => id)).size, 10, example.id);
    for (const [mode, expected] of Object.entries(paths)) {
      const selected = example.stages.filter((item) => !item.mode || item.mode === mode);
      assert.deepEqual(
        selected.map(({ id }) => id),
        expected,
        `${example.id}/${mode}`,
      );
      for (const item of selected)
        assert.ok(
          item.views.some((view) => !view.mode || view.mode === mode),
          item.id,
        );
    }
    for (const item of example.stages) {
      for (const content of [...item.views, ...(item.code || [])])
        if (content.mode) {
          assert.ok(Object.hasOwn(paths, content.mode), item.id);
          if (item.mode) assert.equal(content.mode, item.mode, `${example.id}/${item.id}`);
        }
    }
    assert.equal(stage(example, 'preprocess').optional, true);
    assert.equal(stage(example, 'postprocess').optional, true);
  }
});

test('compiling the flow leaves shared observations, answers, and prediction fixtures untouched', () => {
  assert.deepEqual(
    scenarios.map(({ id, inputs, answers }) => ({ id, inputs, answers })),
    sourceRecords,
  );
  assert.deepEqual(predictions.examples, sourcePredictions);
});

test('training answers attach to their own records without leaking into prediction inputs', () => {
  for (const example of flowCases) {
    const scenario = sourceRecords.find(({ id }) => id === example.id);
    const records = stage(example, 'load').views.filter((view) => view.kind === 'records');
    assert.equal(records.length, 2, example.id);
    const known = records.find((view) => view.mode === 'predict');
    const training = records.find((view) => view.mode === 'train');
    assert.ok(known && training, example.id);
    assert.deepEqual(
      known.value,
      scenario.inputs,
      `${example.id}: prediction has only known inputs`,
    );
    assert.deepEqual(project(training.value, scenario.inputs), scenario.inputs, example.id);
    assert.deepEqual(project(training.value, scenario.answers), scenario.answers, example.id);
    assert.deepEqual(
      new Set(valuePaths(training.value)),
      new Set([...valuePaths(scenario.inputs), ...valuePaths(scenario.answers)]),
      `${example.id}: add only the aligned historical answer fields`,
    );
    const answers = stage(example, 'loss').views.filter((view) => view.kind === 'records');
    assert.equal(answers.length, 1, example.id);
    assert.deepEqual(
      answers[0].value,
      scenario.answers,
      `${example.id}: loss uses historical answers`,
    );
  }
});

test('prepared and forward trees preserve canonical owners, field types, and target roles', () => {
  for (const example of flowCases) {
    const scenario = scenarios.find(({ id }) => id === example.id);
    const parsed = parseSchema(scenario.schema);
    const canonical = flatten(parsed.nodes);
    const expected = canonical.map(({ path }) => path);
    assert.ok(
      stage(example, 'prepare').code.some(({ source }) => source === scenario.schema),
      `${example.id}: reuse the complete canonical class schema`,
    );
    for (const mode of Object.keys(paths)) {
      const view = stage(example, 'prepare').views.find(
        (view) => view.kind === 'tree' && view.mode === mode,
      );
      assert.ok(view, `${example.id}/${mode}`);
      const fields = treePaths(view.nodes);
      assert.deepEqual(
        fields.map(({ path }) => path),
        expected,
        `${example.id}/${mode}`,
      );
      for (const field of canonical.filter(({ kind }) => kind === 'field')) {
        const detail = fields.find(({ path }) => path === field.path).node.detail;
        assert.ok(detail.includes(field.type), `${example.id}${field.path}: retain the type`);
        assert.match(detail, field.target ? /answer|target|withheld|hidden/i : /input/i);
        if (field.target) assert.doesNotMatch(detail, /typed input/i);
      }
    }
    const forward = stage(example, 'model').views.find((view) => view.kind === 'tree');
    const root = treePaths(forward.nodes).find(({ node }) => node.label === parsed.modelClass).node;
    const fields = treePaths(root.children);
    assert.deepEqual(
      fields.map(({ path }) => path),
      expected,
      example.id,
    );
    for (const target of canonical.filter(({ target }) => target)) {
      const detail = fields.find(({ path }) => path === target.path).node.detail;
      assert.match(detail, /masked|hidden|withheld/i, `${example.id}${target.path}`);
      assert.doesNotMatch(
        detail,
        /omitted/i,
        'A shared forward tree must remain accurate during training.',
      );
    }
  }
});

test('output writers show the actual nested rich-field contract from the existing result fixtures', () => {
  for (const example of flowCases) {
    const fixture = predictions.examples
      .find(({ caseId }) => caseId === example.id)
      .blocks.find((block) => block.kind === 'prediction');
    const view = stage(example, 'write').views.find(
      (view) => view.kind === 'tree' && view.nodes.some(({ label }) => label === 'predictions'),
    );
    assert.ok(view, example.id);
    const nodes = treePaths(view.nodes);
    const actual = new Map(nodes.map(({ path, node }) => [path, node]));
    assert.equal(actual.size, nodes.length, `${example.id}: output paths must be unique`);
    const lists = new Set();
    const expected = new Set(['/predictions']);
    for (const target of new Set(readoutPaths(fixture.value.predictions, '/predictions', lists))) {
      const parts = target.split('/').slice(1);
      for (let length = 1; length <= parts.length; length++)
        expected.add(`/${parts.slice(0, length).join('/')}`);
      for (const field of ['state', 'content', 'inferred']) expected.add(`${target}/${field}`);
    }
    assert.deepEqual(new Set(actual.keys()), expected, `${example.id}: public names and nesting`);
    for (const path of lists)
      assert.match(
        actual.get(path).detail,
        /list/i,
        `${example.id}${path}: preserve repeated records`,
      );
  }
});

test('returned predictions reuse the canonical illustrative rows without invented outputs', () => {
  for (const example of flowCases) {
    const scenario = scenarios.find(({ id }) => id === example.id);
    const fixture = predictions.examples
      .find(({ caseId }) => caseId === example.id)
      .blocks.find((block) => block.kind === 'prediction');
    const views = example.stages
      .flatMap((stage) => stage.views)
      .filter((view) => view.kind === 'prediction');
    assert.equal(views.length, 1, example.id);
    assert.deepEqual(views[0].value, fixture.value, example.id);
    assert.deepEqual(views[0].inputs, scenario.inputs, example.id);
    assert.deepEqual(
      Object.keys(views[0].value),
      ['predictions'],
      'Metadata is not an invented output column.',
    );
    const dataset = stage(example, 'load').views.find((view) => view.kind === 'dataset');
    assert.deepEqual(dataset.dataset, scenario.dataset, example.id);
  }
});

test('intermediate tensors, losses, gradients, and updates remain schematic', () => {
  for (const example of flowCases) {
    for (const id of ['prepare', 'model', 'backward', 'update'])
      for (const view of stage(example, id).views)
        assert.ok(
          ['tree', 'text'].includes(view.kind),
          `${example.id}/${id}: no fabricated numeric payload`,
        );
    for (const view of stage(example, 'loss').views)
      assert.ok(['records', 'tree', 'text'].includes(view.kind), example.id);
    for (const item of example.stages)
      for (const view of item.views.filter(({ kind }) => kind === 'tree'))
        for (const { node } of treePaths(view.nodes)) {
          assert.deepEqual(
            Object.keys(node).filter((key) => !['label', 'detail', 'children'].includes(key)),
            [],
            `${example.id}/${item.id}: schematic nodes contain labels, not computed values`,
          );
          assert.equal(typeof node.label, 'string');
          if (node.detail !== undefined) assert.equal(typeof node.detail, 'string');
        }
  }
});

test('Python examples parse without execution and collected prediction calls select one device', () => {
  const snippets = flowCases.flatMap((example) =>
    example.stages.flatMap((stage) =>
      (stage.code || []).map(({ source }) => ({ name: `${example.id}/${stage.id}`, source })),
    ),
  );
  const audit = JSON.parse(
    execFileSync(
      'python3',
      [
        '-c',
        `
import ast, json, sys
reports = []
for snippet in json.load(sys.stdin):
    tree = ast.parse(snippet['source'], filename=snippet['name'])
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'predict':
            keywords = {item.arg: item.value for item in node.keywords}
            calls.append({
                'datamodule': 'datamodule' in keywords,
                'devices': ast.literal_eval(keywords['devices']) if 'devices' in keywords else None,
            })
    reports.append({'name': snippet['name'], 'calls': calls})
json.dump(reports, sys.stdout)
`,
      ],
      { input: JSON.stringify(snippets), encoding: 'utf8' },
    ),
  );
  assert.equal(audit.length, snippets.length);
  const calls = audit.flatMap(({ name, calls }) => calls.map((call) => ({ name, ...call })));
  assert.equal(calls.length, flowCases.length * 2);
  for (const call of calls) {
    assert.equal(call.datamodule, true, call.name);
    assert.equal(
      call.devices,
      1,
      `${call.name}: collected prediction must not default to distributed devices`,
    );
  }
});
