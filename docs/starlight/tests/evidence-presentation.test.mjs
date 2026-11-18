import assert from 'node:assert/strict';
import test from 'node:test';
import { comparison } from '../src/evidence/comparison.ts';
import { metricAt } from '../src/evidence/metrics.ts';
import {
  experimentsForType,
  leafTypes,
  question,
  tags,
  typesFor,
} from '../src/evidence/catalog.ts';
import { status } from '../src/evidence/presentation.ts';
import { record } from './fixtures/evidence.mjs';
import { readReview } from '../src/evidence/metadata.ts';

test('an independent experiment owns new metric semantics and labels without a shared entry', () => {
  const review = readReview(
    {
      question: 'Can this independently added experiment define a new metric?',
      modeling: 'Its own synthetic setup.',
      note: 'Its own scope.',
      comparison: {
        kind: 'Recorded measurement',
        metric: 'Custom distance',
        definition: 'Distance in the units owned by this experiment.',
        direction: 'Closer to zero is better',
        conditions: [{ label: 'New condition', path: ['distance'], format: 'number' }],
      },
      criteria: { prefixes: { own: 'Experiment-owned condition' } },
    },
    'independent.py',
  );
  const item = {
    id: 'P999',
    review,
    sourceMatches: true,
    result: { ...record(), metrics: { distance: 7 } },
  };
  const chart = comparison(item);
  assert.equal(chart.metric, 'Custom distance');
  assert.equal(chart.definition, review.comparison.definition);
  assert.equal(chart.direction, 'Closer to zero is better');
  assert.equal(chart.points[0].value, 7);
  assert.throws(() => readReview(undefined, 'missing.py'), /evidence-review/);
  assert.throws(
    () =>
      readReview(
        {
          ...review,
          comparison: {
            ...review.comparison,
            conditions: [{ label: 'Bad', path: [1.5], format: 'number' }],
          },
        },
        'invalid.py',
      ),
    /paths/,
  );
});

// Presentation tests use invented values, independent of downloaded CI measurements.
const selectedPublication = process.env.RELFLOW_EVIDENCE_PUBLICATION;
delete process.env.RELFLOW_EVIDENCE_PUBLICATION;
const { evidenceEntries } = await import('../src/evidence/data.ts');
if (selectedPublication !== undefined)
  process.env.RELFLOW_EVIDENCE_PUBLICATION = selectedPublication;
const metrics = {
  P001: { measured: { baseline_rmse: 1, rmse: 0.3, nrmse: 0.2 }, permutation_drift: 0.01 },
  P021: { category_oov_auc: 0.5, hash_auc: 0.8 },
  P047: { arms: { extended: { before: { w: { rmse: 2 } }, after: { w: { rmse: 1 } } } } },
  P048: { arms: { selective_reset: { before: { u: { rmse: 2 } }, after: { u: { rmse: 1 } } } } },
  P060: { full: 0.8, partial: 0.5, none: 0.1 },
};
const entry = (id) => {
  const selected = evidenceEntries.find((candidate) => candidate.id === id);
  assert.ok(selected, `${id} must exist in the experiment catalog`);
  return { ...structuredClone(selected), result: record(metrics[id] || {}) };
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

test('all plotted values are traceable to exact selected metric paths', () => {
  for (const id of Object.keys(metrics)) {
    const selected = entry(id);
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
});

test('primary type tags are the same source used by the leaf reference tables', () => {
  for (const type of leafTypes) {
    const subjects = experimentsForType(evidenceEntries, type);
    assert.equal(new Set(subjects.map((item) => item.id)).size, subjects.length);
  }
  assert.deepEqual(typesFor(entry('P021')), ['Category', 'Hash']);
  assert.deepEqual(
    typesFor(entry('P060')),
    ['Category'],
    'Yes/no outcomes in this experiment use Category, not Boolean.',
  );
  assert.deepEqual(typesFor(entry('P069')), ['Enum']);
  assert.deepEqual(typesFor(entry('P070')), ['Enum']);
  assert.deepEqual(typesFor(entry('P100')), ['Number', 'Ordinal', 'Category']);
  assert.deepEqual(
    experimentsForType(evidenceEntries, 'Group').map((item) => item.id),
    ['P108', 'P109', 'P110', 'P111', 'P112'],
  );
  assert.ok(
    !typesFor({ tags: null }).includes('Group'),
    'Retired Cluster results must not become current Group evidence.',
  );
  assert.ok(
    !typesFor(entry('P093')).includes('Period'),
    'Elapsed Number inputs are not calendar-part evidence.',
  );
  assert.equal(question(entry('P001')), 'Can relflow add a list of numbers?');
  assert.ok(evidenceEntries.every((item) => question(item).endsWith('?')));
});

test('every experiment has a complete individual explanation and explicit primary comparison', async () => {
  const { modeling, review } = await import('../src/evidence/catalog.ts');
  assert.ok(evidenceEntries.every((entry) => entry.review));
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
});

test('limitation proofs and confounded comparisons retain their scientific scope', async () => {
  assert.match(entry('P028').review.note, /limitation.*not.*succeeded/);
  assert.match(entry('P024').review.note, /shortcut/);
  assert.match(entry('P068').review.note, /does not mean.*extrapolated/);
  assert.equal(entry('P043').review.comparison.kind, 'Recorded conditions');
  assert.match(entry('P043').review.note, /different training budgets/);
  const operations = comparison(entry('P027'));
  assert.equal(operations.points.length, 12);
  assert.equal(operations.metric, 'Normalized RMSE');
  const sensors = comparison(entry('P074'));
  assert.equal(sensors.metric, 'MSE');
  assert.deepEqual(
    sensors.points.map((point) => point.path),
    ['100', '010', '001', '111'].map((pattern) => ['patterns', pattern, 'mse']),
  );
  assert.equal(entry('P080').review.comparison.metric, 'Adjusted Rand index');
  assert.equal(entry('P096').review.comparison.metric, 'False positive rate');
});

test('human-readable criterion labels retain separate criterion identities and original records', async () => {
  const { criterionLabel } = await import('../src/evidence/criteria.ts');
  for (const entry of evidenceEntries) {
    const originals = Object.keys(entry.result?.checks || {});
    const copy = JSON.stringify(entry.result);
    const labels = originals.map((name) => criterionLabel(entry, name));
    assert.equal(
      new Set(labels).size,
      originals.length,
      `${entry.id} must not merge distinct criteria`,
    );
    assert.ok(labels.every((label) => label.length && !/\bnRMSE\b|\bSD\b/.test(label)));
    assert.equal(JSON.stringify(entry.result), copy);
  }
  assert.equal(
    criterionLabel(entry('P074'), '101: tracks conditional mean'),
    'Sensors 1 and 3 visible: tracks conditional mean',
  );
  assert.equal(
    criterionLabel(entry('P061'), '0.2: evaluation preserves mapping and exposures'),
    '20% input unavailability: evaluation preserves mapping and exposures',
  );
  assert.equal(
    criterionLabel(entry('P001'), 'Seen-length nRMSE below 0.35'),
    'Seen-length normalized RMSE below 0.35',
  );
});

test('script-owned tags cover every current proof without central ID membership lists', async () => {
  const { capabilities, capabilityFor, conditions, conditionsFor } =
    await import('../src/evidence/catalog.ts');
  assert.deepEqual(
    capabilities.map((group) => group.id),
    ['computation', 'structure', 'representation', 'training', 'reliability'],
  );
  for (const group of capabilities) assert.equal('experiments' in group, false);
  assert.ok(
    evidenceEntries.every(
      (entry) => entry.tags && capabilities.some((group) => group.id === entry.tags.capability),
    ),
  );
  assert.equal(capabilityFor(entry('P025')).id, 'structure');
  assert.equal(capabilityFor(entry('P099')).id, 'representation');
  assert.throws(
    () => capabilityFor({ id: 'P999', tags: null, archived: false }),
    /script metadata/,
  );
  for (const condition of conditions) assert.equal('experiments' in condition, false);
  assert.ok(conditionsFor(entry('P025')).some((item) => item.id === 'unseen-identities'));
  assert.ok(conditionsFor(entry('P002')).some((item) => item.id === 'longer-collections'));
  for (const entry of evidenceEntries)
    assert.ok(conditionsFor(entry).length, `${entry.id}: review its evaluation conditions`);
  assert.ok(conditionsFor(entry('P028')).some((item) => item.id === 'unseen-identities'));
  assert.ok(conditionsFor(entry('P074')).some((item) => item.id === 'new-combinations'));
  assert.ok(conditionsFor(entry('P082')).some((item) => item.id === 'longer-collections'));
  assert.ok(
    !conditionsFor(entry('P075')).some(
      (item) => item.id === 'longer-collections' || item.id === 'extrapolation',
    ),
    'P075 tests capacity at lengths already seen during fitting',
  );
  assert.ok(
    !conditionsFor(entry('P111')).some((item) => item.id === 'unseen-identities'),
    'P111 tests new observations of known identities',
  );
  assert.ok(typesFor(entry('P089')).includes('Boolean'));
  assert.ok(typesFor(entry('P090')).includes('Number'));
});

test('tag assignments follow script metadata without registering experiment IDs', async () => {
  const { capabilityFor, conditionsFor } = await import('../src/evidence/catalog.ts');
  const source =
    '# /// script\n# dependencies = []\n# [tool.relflow.evidence]\n' +
    '# capability = "structure"\n# types = ["Hash"]\n# conditions = ["unseen-identities"]\n# ///\n' +
    '@experiment("P987654", seed=0)\nraise RuntimeError("must never import")\n';
  const entry = { id: 'P987654', tags: tags(source) };
  assert.equal(capabilityFor(entry).id, 'structure');
  assert.deepEqual(typesFor(entry), ['Hash']);
  assert.deepEqual(
    conditionsFor(entry).map((item) => item.id),
    ['unseen-identities'],
  );
  entry.tags = tags(source.replace('"structure"', '"training"').replace('"Hash"', '"Vector"'));
  assert.equal(capabilityFor(entry).id, 'training');
  assert.deepEqual(typesFor(entry), ['Vector']);
  assert.equal(tags('# Historical script without inline metadata\n'), null);
  assert.throws(() => capabilityFor({ id: 'P123', tags: null }), /needs a capability/);
  for (const changed of [
    source.replace('"structure"', '"made-up"'),
    source.replace('"Hash"', '"made-up"'),
    source.replace('"unseen-identities"', '"made-up"'),
    source.replace('types = ["Hash"]', 'types = ["Hash", "Hash"]'),
    source.replace('conditions = ["unseen-identities"]', 'conditions = []'),
    source.replace('types = ["Hash"]', 'typos = ["Hash"]'),
    source.replace('capability = "structure"', 'capability = ["structure"]'),
    source + source,
    source.replace('# ///\n', ''),
  ])
    assert.throws(() => tags(changed));
});
