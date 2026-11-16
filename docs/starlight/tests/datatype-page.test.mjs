import assert from 'node:assert/strict';
import test from 'node:test';
import { checkSections } from '../src/datatypes/page.ts';

const contract = [
  ['input-values', 'Input values'],
  ['standardization-and-processor-pipelines', 'Standardization and processor pipelines'],
  ['how-is-this-leaf-trained', 'How is this leaf trained?'],
  ['leaf-specific-parameters', 'Leaf-specific parameters'],
  ['errors-raised', 'Errors raised'],
  ['relevant-evidence', 'Relevant evidence'],
];

function section(id, title) {
  return `<div class="sl-heading-wrapper level-h2"><h2 id="${id}">${title}</h2><a href="#${id}">Section link</a></div><p>Authored prose.</p>`;
}

test('rendered datatype prose accepts the complete heading contract and ordinary subsection content', () => {
  const html = contract.map(([id, title]) => section(id, title)).join('\n');
  const body = html.replace(
    '<p>Authored prose.</p>',
    '<p>Authored <code>rf.Number</code> prose.</p><h3 id="source-units">Source units</h3><table><tr><td>Value</td></tr></table>',
  );
  assert.doesNotThrow(() => checkSections(body, 'number-examples'));
});

test('literal code headings and inert template content cannot masquerade as datatype sections', () => {
  const body = contract.map(([id, title]) => section(id, title)).join('\n');
  const examples =
    '<pre><code>&lt;h2 id="unexpected"&gt;Example HTML&lt;/h2&gt;</code></pre>' +
    '<script>const sample = "<h2>Not a section</h2>";</script>' +
    '<template><h2 id="deferred">Not visible content</h2></template>';
  assert.doesNotThrow(() => checkSections(body + examples, 'text-examples'));
});

test('missing or reordered sections fail with page identity and the observed contract', () => {
  const missing = contract
    .slice(1)
    .map(([id, title]) => section(id, title))
    .join('');
  assert.throws(
    () => checkSections(missing, 'duration-examples'),
    /duration-examples.*Input values.*received Standardization/,
  );
  const swapped = [...contract];
  [swapped[0], swapped[1]] = [swapped[1], swapped[0]];
  assert.throws(
    () =>
      checkSections(swapped.map(([id, title]) => section(id, title)).join(''), 'duration-examples'),
    /in that order/,
  );
});

test('duplicate sections, extra sections, changed titles, and broken deep-link IDs are rejected', () => {
  const body = contract.map(([id, title]) => section(id, title)).join('');
  for (const changed of [
    body + section('relevant-evidence', 'Relevant evidence'),
    body + section('extra', 'Additional section'),
    body.replace('>Input values</h2>', '>Inputs</h2>'),
    body.replace('id="errors-raised"', 'id="errors"'),
    body.replace(' id="input-values"', ''),
  ])
    assert.throws(() => checkSections(changed, 'category-examples'), /must provide/);
});

test('section titles retain inline Markdown text while requiring actual level-two headings', () => {
  const body = contract.map(([id, title]) => section(id, title)).join('');
  assert.doesNotThrow(() =>
    checkSections(
      body.replace('>Input values</h2>', '>Input <em>values</em></h2>'),
      'set-examples',
    ),
  );
  assert.throws(
    () => checkSections(body.replaceAll('<h2', '<h3').replaceAll('</h2>', '</h3>'), 'set-examples'),
    /received no sections/,
  );
});
