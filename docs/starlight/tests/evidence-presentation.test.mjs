import assert from 'node:assert/strict';
import test from 'node:test';
import { fileURLToPath } from 'node:url';
import { comparison } from '../src/evidence/comparison.ts';
import { metricAt } from '../src/evidence/metrics.ts';
import { experimentsForType, leafTypes, question, typesFor } from '../src/evidence/catalog.ts';
import { status } from '../src/evidence/presentation.ts';

// These unit fixtures exercise known measurements independently of a host's
// selected (possibly pending) publication. Adapter tests cover that selection.
const selectedPublication = process.env.RELFLOW_EVIDENCE_PUBLICATION;
process.env.RELFLOW_EVIDENCE_PUBLICATION = fileURLToPath(
  new URL('../../../evidence/results.json', import.meta.url),
);
const { evidenceEntries } = await import('../src/evidence/data.ts');
if (selectedPublication === undefined) delete process.env.RELFLOW_EVIDENCE_PUBLICATION;
else process.env.RELFLOW_EVIDENCE_PUBLICATION = selectedPublication;

const entry = (id) => {
  const result = evidenceEntries.find((candidate) => candidate.id === id);
  assert.ok(result, `${id} must exist in the experiment catalog`);
  return structuredClone(result);
};

test('metric lookup preserves literal keys and Python array indexes without coercion', () => {
  const metrics = { 0.2: { 'intact/A/nrmse': [2, 4, 8] }, empty: [], nil: null, invalid: Infinity };
  assert.equal(metricAt(metrics, ['0.2', 'intact/A/nrmse', -1]), 8);
  assert.equal(metricAt(metrics, ['0.2', 'intact/A/nrmse', 0]), 2);
  for (const path of [
    ['empty', -1],
    ['nil', 'x'],
    ['invalid'],
    ['missing'],
    ['__proto__'],
    ['0.2', 'intact/A/nrmse', '1'],
  ]) {
    assert.equal(metricAt(metrics, path), null);
  }
});

test('the opening sum comparison uses recorded source-unit RMSE, leaving drift secondary', () => {
  const selected = entry('P001');
  const chart = comparison(selected);
  assert.equal(chart.metric, 'RMSE');
  assert.equal(chart.kind, 'Baseline reference');
  assert.deepEqual(
    chart.points.map((point) => point.path),
    [
      ['measured', 'baseline_rmse'],
      ['measured', 'rmse'],
    ],
  );
  assert.deepEqual(
    chart.points.map((point) => point.value),
    [selected.result.metrics.measured.baseline_rmse, selected.result.metrics.measured.rmse],
  );
  assert.equal(chart.domain[0], 0);
  assert.equal(typeof selected.result.metrics.permutation_drift, 'number');
});

test('primary values and verdicts follow the selected record independently', () => {
  const selected = entry('P001');
  selected.result.metrics.measured.baseline_rmse = 7;
  selected.result.metrics.measured.rmse = 9;
  selected.result.outcome = 'not_met';
  selected.result.checks = { 'Synthetic unmet criterion': false };
  const chart = comparison(selected);
  assert.deepEqual(
    chart.points.map((point) => point.value),
    [7, 9],
  );
  assert.equal(chart.available, true);
  assert.equal(status(selected).id, 'not_met');
  assert.equal(chart.domain[1], 9);
});

test('no chart substitutes authored claims for missing, erroneous, or unmatched execution data', () => {
  for (const change of [
    (selected) => {
      selected.result = null;
    },
    (selected) => {
      selected.result.outcome = 'error';
    },
    (selected) => {
      selected.sourceMatches = false;
    },
    (selected) => {
      selected.result.mode = 'smoke';
    },
  ]) {
    const selected = entry('P021');
    change(selected);
    const chart = comparison(selected);
    assert.equal(chart.available, false);
    assert.ok(chart.points.every((point) => point.value === null));
  }
  const selected = entry('P021');
  delete selected.result.metrics.hash_auc;
  assert.equal(comparison(selected).available, false);
});

test('paired comparisons retain their metric, target, and time-point identity', () => {
  const hash = comparison(entry('P021'));
  assert.equal(hash.metric, 'AUC');
  assert.deepEqual(
    hash.points.map((point) => point.path),
    [['category_oov_auc'], ['hash_auc']],
  );
  for (const [id, arm, target] of [
    ['P047', 'extended', 'w'],
    ['P048', 'selective_reset', 'u'],
  ]) {
    const chart = comparison(entry(id));
    assert.equal(chart.kind, 'Before and after');
    assert.equal(chart.metric, 'RMSE');
    assert.deepEqual(
      chart.points.map((point) => point.path),
      ['before', 'after'].map((stage) => ['arms', arm, stage, target, 'rmse']),
    );
  }
  const calibrated = comparison(entry('P060'));
  assert.equal(calibrated.metric, 'Probability');
  assert.equal(calibrated.kind, 'Recorded conditions');
  assert.equal(calibrated.points.length, 3);
  assert.deepEqual(calibrated.domain, [0, 1]);
});

test('all plotted values are traceable to exact selected metric paths, including archived counters', () => {
  for (const selected of evidenceEntries) {
    const chart = comparison(selected);
    if (!chart.available) continue;
    for (const point of chart.points) {
      assert.equal(
        point.value,
        metricAt(selected.result.metrics, point.path),
        `${selected.id} ${JSON.stringify(point.path)}`,
      );
      assert.ok(point.value >= chart.domain[0] && point.value <= chart.domain[1]);
    }
  }
  const archived = entry('P018');
  assert.equal(archived.readouts[0].value, archived.result.metrics.final_committed.at(-1));
  assert.equal(status(archived).id, 'not_met');
});

test('primary type tags are the same source used by the leaf reference tables', () => {
  for (const type of leafTypes) {
    const subjects = experimentsForType(evidenceEntries, type);
    assert.equal(new Set(subjects.map((item) => item.id)).size, subjects.length);
  }
  assert.deepEqual(typesFor('P021'), ['Category', 'Hash']);
  assert.deepEqual(
    typesFor('P060'),
    ['Category'],
    'Yes/no outcomes in this experiment use Category, not Boolean.',
  );
  assert.deepEqual(typesFor('P069'), ['Enum']);
  assert.deepEqual(typesFor('P070'), ['Enum']);
  assert.deepEqual(typesFor('P100'), ['Number', 'Ordinal', 'Category']);
  assert.deepEqual(
    experimentsForType(evidenceEntries, 'Group').map((item) => item.id),
    ['P108', 'P109', 'P110', 'P111', 'P112'],
  );
  assert.ok(
    !typesFor('P018').includes('Group'),
    'Retired Cluster results must not become current Group evidence.',
  );
  assert.ok(
    !typesFor('P093').includes('Period'),
    'Elapsed Number inputs are not calendar-part evidence.',
  );
  assert.equal(question(entry('P001')), 'Can relflow add a list of numbers?');
  assert.ok(evidenceEntries.every((item) => question(item).endsWith('?')));
});

test('every experiment has a complete individual explanation and explicit primary comparison', async () => {
  const { reviews } = await import('../src/evidence/reviews.ts');
  const { modeling, review } = await import('../src/evidence/catalog.ts');
  assert.deepEqual(Object.keys(reviews).sort(), evidenceEntries.map((entry) => entry.id).sort());
  const texts = new Set();
  for (const entry of evidenceEntries) {
    const content = review(entry);
    assert.ok(content.question.endsWith('?'), entry.id);
    assert.ok(content.modeling.split(/\s+/).length <= 100, `${entry.id} should stay concise`);
    assert.ok(content.modeling.split(/\s+/).length >= 25, `${entry.id} must explain a real setup`);
    assert.equal(
      modeling(entry),
      content.modeling,
      `${entry.id} cannot silently fall back to extracted prose`,
    );
    assert.ok(content.comparison.conditions.length, entry.id);
    assert.ok(
      content.comparison.conditions.every((condition) => condition.label && condition.path.length),
    );
    texts.add(content.modeling);
  }
  assert.equal(texts.size, evidenceEntries.length, 'Every setup is individually authored.');
  assert.throws(() => review({ id: 'P999' }), /individually reviewed/);
});

test('limitation proofs and confounded comparisons retain their scientific scope', async () => {
  const { reviews } = await import('../src/evidence/reviews.ts');
  assert.match(reviews.P028.note, /limitation.*not.*succeeded/);
  assert.match(reviews.P024.note, /shortcut/);
  assert.match(reviews.P068.note, /does not mean.*extrapolated/);
  assert.equal(reviews.P043.comparison.kind, 'Recorded conditions');
  assert.match(reviews.P043.note, /different training budgets/);
  const operations = comparison(entry('P027'));
  assert.equal(operations.points.length, 12);
  assert.equal(operations.metric, 'Normalized RMSE');
  const sensors = comparison(entry('P074'));
  assert.equal(sensors.metric, 'MSE');
  assert.deepEqual(
    sensors.points.map((point) => point.path),
    ['100', '010', '001', '111'].map((pattern) => ['patterns', pattern, 'mse']),
  );
  assert.equal(reviews.P080.comparison.metric, 'Adjusted Rand index');
  assert.equal(reviews.P096.comparison.metric, 'False positive rate');
});

test('human-readable criterion labels retain separate criterion identities and original records', async () => {
  const { criterionLabel } = await import('../src/evidence/criteria.ts');
  for (const entry of evidenceEntries) {
    const originals = Object.keys(entry.result?.checks || {});
    const copy = JSON.stringify(entry.result);
    const labels = originals.map((name) => criterionLabel(entry.id, name));
    assert.equal(
      new Set(labels).size,
      originals.length,
      `${entry.id} must not merge distinct criteria`,
    );
    assert.ok(labels.every((label) => label.length && !/\bnRMSE\b|\bSD\b/.test(label)));
    assert.equal(JSON.stringify(entry.result), copy);
  }
  assert.equal(
    criterionLabel('P074', '101: tracks conditional mean'),
    'Sensors 1 and 3 visible: tracks conditional mean',
  );
  assert.equal(
    criterionLabel('P061', '0.2: evaluation preserves mapping and exposures'),
    '20% input unavailability: evaluation preserves mapping and exposures',
  );
  assert.equal(
    criterionLabel('P001', 'Seen-length nRMSE below 0.35'),
    'Seen-length normalized RMSE below 0.35',
  );
});
