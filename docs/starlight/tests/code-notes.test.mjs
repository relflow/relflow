import assert from 'node:assert/strict';
import test from 'node:test';
import { ExpressiveCodeEngine } from '@expressive-code/core';
import { configNotes } from '../src/code/config-notes.mjs';

const engine = new ExpressiveCodeEngine({ plugins: [configNotes()] });

function content(node) {
  return node.type === 'text' ? node.value : (node.children || []).map(content).join('');
}

async function annotations(code, language = 'python', meta = '') {
  const { renderedGroupAst, renderedGroupContents } = await engine.render({ code, language, meta });
  assert.equal(
    renderedGroupContents[0].codeBlock.code,
    code,
    'Adding notes must never change the copied source.',
  );
  const notes = [];
  const renderedLines = [];
  function visit(node) {
    if (node.properties?.className?.includes('ec-line')) renderedLines.push(content(node));
    if (node.properties?.className?.includes('rf-code-note'))
      notes.push({
        value: content(node),
        title: node.properties['data-note-title'],
        text: node.properties['data-note-text'],
        field: node.properties['data-contract-field'],
      });
    for (const child of node.children || []) visit(child);
  }
  visit(renderedGroupAst);
  // Expressive Code uses a newline text node to give an empty line height.
  assert.deepEqual(
    renderedLines.map((line) => (line === '\n' ? '' : line)),
    code.split('\n'),
    'Rendered note wrappers must retain every original character and line.',
  );
  return notes;
}

test('constructor and fit presets receive their own explanations, including multiline fit calls', async () => {
  const notes = await annotations(`class Customer(rf.Model):
    amount = rf.Number

model = Customer(preset="small")
model.fit(
    data,
    preset="small",
)`);
  const presets = notes.filter((note) => note.value.startsWith('preset='));
  assert.deepEqual(
    presets.map((note) => note.title),
    ['Architecture defaults', 'Training defaults'],
  );
  assert.match(presets[0].text, /new, untrained model/);
  assert.match(presets[1].text, /does not change the model architecture/);
  const medium = await annotations('model = rf.Model(preset="medium", amount=rf.Number)');
  assert.ok(medium.some((note) => note.title === 'Architecture defaults'));
});

test('unrelated loading and estimator calls never acquire relflow workflow or preset notes', async () => {
  const notes = await annotations(`row = json.load(response)
weights = torch.load(path)
estimator.fit(data, preset="small")
estimator.predict(data)
document.save(path)
factory(preset="small")`);
  assert.deepEqual(notes, []);
});

test('Text cache and writer format notes stay attached to their owning calls', async () => {
  const notes = await annotations(`body = rf.Text(
    cache=True, memory=64, resident=None,
    tokenizer_batch_size=4096, store="text-cache",
)
writer = rf.Writer("predictions", format="ipc")`);
  assert.match(notes.find((note) => note.value === 'cache=True').text, /bounded RAM/);
  assert.match(notes.find((note) => note.value === 'memory=64').text, /shared store owner/);
  assert.match(notes.find((note) => note.value === 'resident=None').text, /512 MiB/);
  assert.match(
    notes.find((note) => note.value === 'tokenizer_batch_size=4096').text,
    /tokenization/,
  );
  assert.match(notes.find((note) => note.value === 'format="ipc"').text, /rank-N\.arrow/);
  assert.deepEqual(
    await annotations(
      'other(cache=True, memory=64, resident=None, store="elsewhere", tokenizer_batch_size=4096, format="ipc")',
    ),
    [],
  );
  const mediaNotes = await annotations(
    'recording = rf.Audio(base="s3://calls/", immutable=True, max_windows=6, store="audio-cache")',
  );
  assert.match(mediaNotes.find((note) => note.value === 'rf.Audio').text, /frozen audio encoder/);
  assert.match(mediaNotes.find((note) => note.value.startsWith('base=')).text, /URI prefix/);
  assert.match(mediaNotes.find((note) => note.value === 'immutable=True').text, /zero source/);
  assert.match(mediaNotes.find((note) => note.value === 'max_windows=6').text, /6 overlapping/);
  assert.match(mediaNotes.find((note) => note.value.startsWith('store=')).text, /cache directory/);
});

test('schema visibility and reconstruction options have distinct notes', async () => {
  const notes = await annotations(`amount = rf.Number(hide=0.2, task=rf.reconstruct(rate=0.1))
history = rf.Branch(ablate=0.1, amount=amount)
label = rf.Boolean(predict=True)`);
  for (const [value, title] of [
    ['hide', 'Hide training inputs'],
    ['task', 'Learning task'],
    ['rf.reconstruct', 'Reconstruction selection'],
    ['ablate', 'Training ablation'],
    ['predict=True', 'Prediction target'],
  ]) {
    assert.ok(notes.some((note) => note.value === value && note.title === title));
  }
  assert.deepEqual(await annotations('hide = values.is_null()'), []);
});

test('Duration delta notes stay attached to Duration calls', async () => {
  const notes = await annotations(
    'gap = rf.Duration(query="booked_at", delta=True, granularity="hours")',
  );
  assert.match(notes.find((note) => note.value === 'delta=True').text, /preceding timestamp/);
  assert.deepEqual(await annotations('frame.with_columns(other(delta=True))'), []);
});

test('known model classes, loaded instances, and typed instances retain workflow notes', async () => {
  const notes = await annotations(`class Customer(rf.Model):
    amount = rf.Number

customer = Customer(preset="small")
customer.fit(data)
customer.save(path)
loaded = rf.Model.load(path)
loaded.predict(data)
def continue_training(network: rf.Model):
    network.fit(data, preset="medium")`);
  assert.deepEqual(
    notes.filter((note) => note.title === 'Model workflow').map((note) => note.value),
    ['.fit', '.save', '.load', '.predict', '.fit'],
  );
  assert.ok(
    notes.some((note) => note.value === 'preset="medium"' && note.title === 'Training defaults'),
  );
});

test('comments, strings, escaped quotes, and multiline docstrings cannot inject Python notes', async () => {
  const code = String.raw`# model.fit(data, preset="small")
description = '🙂 rf.Model(predict=True) and model.load(path)'
escaped = "a \" rf.Number and model.load(path)"
"""model.fit(data)
rf.Category(predict=True)
"""
message = "# not a comment: rf.Number"
target = rf.Boolean(predict=True)  # rf.Number and model.save(path)`;
  const notes = await annotations(code);
  assert.deepEqual(
    notes.map((note) => note.value),
    ['rf.Boolean', 'predict=True'],
  );
});

test('query string captures and indented schema fields retain linked-field annotations', async () => {
  const notes = await annotations(
    `class Delivery(rf.Model):
    """amount = rf.Number(predict=True)"""
    amount = rf.Number
    elapsed = rf.Duration(query=("opened_at", "resolved_at"))`,
    'python',
    'contract="schema"',
  );
  assert.deepEqual(
    notes.filter((note) => note.title === 'Schema field').map((note) => note.field),
    ['amount', 'elapsed'],
  );
  assert.deepEqual(
    notes.filter((note) => note.title === 'Queried source field').map((note) => note.field),
    ['opened_at', 'resolved_at'],
  );
});

test('quoted YAML keys and JSON contract notes retain exact names and source text', async () => {
  const yaml = await annotations(
    `"customer.name": Ada
'customer''s id': 7
items:
  - "sku-code": boots`,
    'yaml',
    'contract="inputs"',
  );
  assert.deepEqual(
    yaml.map((note) => note.field),
    ['customer.name', "customer's id", 'items', 'sku-code'],
  );
  const json = await annotations(
    '{"inputs": {"customer.name": "Ada"}}',
    'json',
    'contract="inputs"',
  );
  assert.deepEqual(
    json.map((note) => note.field),
    ['inputs', 'customer.name'],
  );
});
