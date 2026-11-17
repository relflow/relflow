import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import test, { after } from 'node:test';
import { record } from './fixtures/evidence.mjs';
const site = fileURLToPath(new URL('../', import.meta.url));
const adapter = new URL('../src/evidence/data.ts', import.meta.url).href;
const source = readFileSync(
  new URL('../../../evidence/experiments/aggregation/attention_sum_in_range.py', import.meta.url),
  'utf8',
);
const snapshot = {
  schema: 1,
  version: 'unit-fixture',
  revision: 'a'.repeat(40),
  report: {},
  results: {
    P001: {
      ...record(),
      provenance: { script_sha256: createHash('sha256').update(source).digest('hex') },
    },
  },
};
const inputDirectory = mkdtempSync(join(tmpdir(), 'relflow-evidence-unit-'));
after(() => rmSync(inputDirectory, { recursive: true, force: true }));
const baselinePath = join(inputDirectory, 'publication.json');
writeFileSync(baselinePath, JSON.stringify(snapshot));

// Each process imports the adapter once, with its own publication selection.
const inspect = `
  const { selection, evidenceEntries } = await import(${JSON.stringify(adapter)});
  process.stdout.write(JSON.stringify({
    selection,
    entries: evidenceEntries.map(({ id, tags, requirements, diagram, result, readouts, sourceRevision, sourceUrl, sourcePath, sourceMatches, sourceSha256, source, code, sections, archived }) => ({
      id, tags, requirements, diagram, result, readouts, sourceRevision, sourceUrl, sourcePath, sourceMatches, sourceSha256, source, code, sections, archived,
    })),
  }));
`;

function readAdapter(publicationPath, directory = site) {
  const env = { ...process.env };
  delete env.RELFLOW_EVIDENCE_PUBLICATION;
  if (publicationPath !== undefined) env.RELFLOW_EVIDENCE_PUBLICATION = publicationPath;
  return spawnSync(
    process.execPath,
    ['--experimental-strip-types', '--input-type=module', '--eval', inspect],
    {
      cwd: directory,
      env,
      encoding: 'utf8',
      maxBuffer: 32 * 1024 * 1024,
      timeout: 30_000,
    },
  );
}

function loaded(publicationPath, directory = site) {
  const child = readAdapter(publicationPath, directory);
  assert.ifError(child.error);
  assert.equal(child.status, 0, child.stderr);
  return JSON.parse(child.stdout);
}

function fixture(t, document, suffix = 'publication.json') {
  const directory = mkdtempSync(join(tmpdir(), 'rock-docs-evidence-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const path = join(directory, suffix);
  const raw = typeof document === 'string' ? document : `${JSON.stringify(document, null, 3)}\n\n`;
  writeFileSync(path, raw);
  return { path, raw, directory };
}

let baseline;
function currentSnapshot() {
  return (baseline ??= loaded(baselinePath));
}

function measuredExample() {
  const entry = currentSnapshot().entries.find(
    (candidate) =>
      candidate.result?.outcome === 'met' &&
      Object.values(candidate.result.checks).some((value) => value === true) &&
      candidate.readouts.some(
        (readout) => typeof readout.value === 'number' && Number.isFinite(readout.value),
      ),
  );
  assert.ok(
    entry,
    'The input snapshot must supply a measured experiment with checks and an authored numeric readout.',
  );
  const readout = entry.readouts.find(
    (candidate) => typeof candidate.value === 'number' && Number.isFinite(candidate.value),
  );
  assert.ok(readout.metric.length, 'The readout must identify its source metric path.');
  return { entry, readout };
}

function replaceMetric(metrics, path, value) {
  let owner = metrics;
  for (const part of path.slice(0, -1)) owner = owner[part];
  owner[path.at(-1)] = value;
}

test('a selected publication supplies every retained record without a separate result catalog', () => {
  const actual = currentSnapshot();
  assert.equal(actual.selection.mode, 'selected');
  assert.deepEqual(JSON.parse(actual.selection.raw), snapshot);
  const retained = actual.entries.filter((entry) => entry.result !== null);
  assert.deepEqual(retained.map((entry) => entry.id).sort(), Object.keys(snapshot.results).sort());
  for (const entry of retained) {
    assert.deepEqual(entry.result, snapshot.results[entry.id]);
    assert.equal(entry.sourceMatches, true, `${entry.id}: retain exact executed source`);
    assert.equal(
      createHash('sha256').update(entry.source).digest('hex'),
      entry.result.provenance.script_sha256,
    );
    assert.equal(entry.sourceSha256, entry.result.provenance.script_sha256);
    assert.ok(entry.code, `${entry.id}: show computational code only from matching source`);
  }
});

test('without a publication every current declaration remains pending', () => {
  const actual = loaded();
  assert.equal(actual.selection.mode, 'pending');
  assert.ok(actual.entries.length);
  assert.ok(actual.entries.every((entry) => entry.result === null && !entry.archived));
  assert.deepEqual(JSON.parse(actual.selection.raw).results, {});
});

test('every current source belongs to one catalog and preserves plain Markdown presentation', () => {
  const actual = loaded();
  for (const entry of actual.entries) {
    assert.match(entry.sourcePath, /^evidence\/experiments\/[^/]+\/[^/]+\.py$/);
    assert.equal('lane' in entry, false);
    assert.ok(entry.diagram.caption.trim(), `${entry.id}: keep the figure caption`);
    assert.ok(entry.diagram.alt.trim(), `${entry.id}: keep alternative text`);
    assert.ok(entry.sections.method.trim(), `${entry.id}: retain the scientific method`);
    assert.doesNotMatch(entry.sections.result, /## Method|\{\{</);
  }
});

test('an explicit publication changes its check, outcome, and authored readout together', (t) => {
  const { entry, readout } = measuredExample();
  const document = structuredClone(snapshot);
  const record = document.results[entry.id];
  const check = Object.keys(record.checks).find((name) => record.checks[name] === true);
  const value = readout.value === 0 ? 1 : readout.value / 2;
  record.checks[check] = false;
  record.outcome = 'not_met';
  replaceMetric(record.metrics, readout.metric, value);
  const input = fixture(t, document);

  const actual = loaded(input.path);
  const changed = actual.entries.find((candidate) => candidate.id === entry.id);
  assert.equal(actual.selection.mode, 'selected');
  assert.equal(changed.result.checks[check], false);
  assert.equal(changed.result.outcome, 'not_met');
  assert.deepEqual(changed.result, record);
  const updatedReadout = changed.readouts.find(
    (candidate) =>
      candidate.label === readout.label &&
      JSON.stringify(candidate.metric) === JSON.stringify(readout.metric),
  );
  assert.ok(updatedReadout);
  assert.equal(updatedReadout.value, value);
  assert.notEqual(updatedReadout.value, readout.value);
});

test('a pending selection keeps every result absent instead of falling back to historical runs', async (t) => {
  const document = {
    schema: snapshot.schema,
    version: 'ci-pending',
    revision: snapshot.revision,
    report: { status: 'The selected CI attempt has not published its measurements.' },
    results: {},
  };
  const input = fixture(t, document);
  const actual = loaded(input.path);
  assert.equal(actual.selection.mode, 'selected');
  assert.equal(actual.selection.version, 'ci-pending');
  assert.equal(actual.selection.report.status, document.report.status);
  assert.ok(
    actual.entries.length,
    'Declarations remain browsable while their selected measurements are pending.',
  );
  assert.ok(actual.entries.every((entry) => entry.result === null));
  assert.ok(
    actual.entries.every((entry) => entry.readouts.every((readout) => readout.value === null)),
  );
  assert.deepEqual(JSON.parse(actual.selection.raw).results, {});
  assert.equal(actual.selection.raw, input.raw);
});

test('a CI preview with an unmet aggregate gate retains its complete candidate records', (t) => {
  const document = structuredClone(snapshot);
  document.version = 'ci-preview';
  document.report.passed = false;
  const actual = loaded(fixture(t, document).path);
  assert.equal(actual.selection.version, 'ci-preview');
  assert.equal(actual.selection.report.passed, false);
  assert.deepEqual(
    actual.entries
      .filter((entry) => entry.result !== null)
      .map((entry) => entry.id)
      .sort(),
    Object.keys(document.results).sort(),
  );
  for (const [id, record] of Object.entries(document.results)) {
    assert.deepEqual(actual.entries.find((entry) => entry.id === id).result, record);
  }
});

test('individual check outcomes survive independently of an aggregate passed flag', (t) => {
  const entry = currentSnapshot().entries.find(
    (candidate) => candidate.result && Object.keys(candidate.result.checks).length > 1,
  );
  assert.ok(entry, 'The input snapshot must contain multiple criteria for one experiment.');
  const document = structuredClone(snapshot);
  const record = document.results[entry.id];
  const names = Object.keys(record.checks);
  record.checks = Object.fromEntries(names.map((name, index) => [name, index !== 0]));
  record.outcome = 'not_met';
  document.report.passed = true;
  const actual = loaded(fixture(t, document).path);
  const checks = actual.entries.find((candidate) => candidate.id === entry.id).result.checks;
  assert.equal(actual.selection.report.passed, true);
  assert.deepEqual(checks, record.checks);
  assert.deepEqual(Object.keys(checks), names);
  assert.equal(checks[names[0]], false);
  assert.ok(names.slice(1).every((name) => checks[name] === true));
});

test('selected raw bytes and every nested metric remain available for result downloads', (t) => {
  const { entry } = measuredExample();
  const document = structuredClone(snapshot);
  document.results[entry.id].metrics.adapter_test_payload = {
    zero: 0,
    negative: -3.5,
    enabled: false,
    missing: null,
    empty_list: [],
    empty_record: {},
    sequence: [0, null, { residual: -0.125 }],
  };
  const input = fixture(t, document);
  const actual = loaded(input.path);
  assert.equal(
    actual.selection.raw,
    input.raw,
    'Download data preserves the selected publication bytes.',
  );
  assert.equal(actual.selection.sha256, createHash('sha256').update(input.raw).digest('hex'));
  const downloadable = JSON.parse(actual.selection.raw);
  assert.deepEqual(downloadable, document);
  for (const [id, record] of Object.entries(document.results)) {
    const result = actual.entries.find((candidate) => candidate.id === id).result;
    assert.deepEqual(
      result,
      record,
      `${id}: preserve all recorded fields, not only headline readouts`,
    );
    assert.deepEqual(result.metrics, downloadable.results[id].metrics);
  }
});

test('an explicit missing publication path fails instead of selecting the snapshot', (t) => {
  const input = fixture(t, snapshot);
  const missing = join(input.directory, 'missing-publication.json');
  const child = readAdapter(missing);
  assert.ifError(child.error);
  assert.notEqual(child.status, 0);
  assert.ok(child.stderr.includes('missing-publication.json'), child.stderr);
  assert.equal(child.stdout, '');
});

for (const [name, document] of [
  ['malformed JSON', '{ this is not JSON'],
  ['an invalid publication envelope', { results: {} }],
]) {
  test(`explicit ${name} fails instead of selecting the snapshot`, (t) => {
    const child = readAdapter(fixture(t, document).path);
    assert.ifError(child.error);
    assert.notEqual(child.status, 0);
    assert.ok(
      child.stderr.trim(),
      'The adapter must explain why the explicit selection is invalid.',
    );
    assert.equal(child.stdout, '');
  });
}

for (const [field, value] of [
  ['outcome', 'gobbledygook'],
  ['mode', 'smoke'],
  ['checks', { criterion: 'true' }],
  ['metrics', []],
  ['provenance', []],
  ['seed', 0.5],
  ['started_at', 'not a timestamp'],
  ['started_at', '2026-11-13T12:00:00'],
  ['duration_seconds', -1],
  ['accelerator', 'missing-device'],
  ['threads', 0],
  ['steps_override', 1],
  ['error', { message: 'wrong shape' }],
]) {
  test(`a selected record rejects invalid ${field}=${JSON.stringify(value)}`, (t) => {
    const document = structuredClone(snapshot);
    const id = Object.keys(document.results)[0];
    document.results[id][field] = value;
    const input = fixture(t, document);
    const child = readAdapter(input.path);
    assert.ifError(child.error);
    assert.notEqual(child.status, 0);
    assert.ok(child.stderr.includes(`Evidence record ${id}`), child.stderr);
    assert.ok(child.stderr.includes(field), child.stderr);
    assert.ok(child.stderr.includes(input.path), child.stderr);
    assert.equal(child.stdout, '');
  });
}

test('required rendering fields cannot be omitted from a selected record', (t) => {
  for (const field of [
    'seed',
    'started_at',
    'duration_seconds',
    'accelerator',
    'threads',
    'steps_override',
  ]) {
    const document = structuredClone(snapshot);
    const id = Object.keys(document.results)[0];
    delete document.results[id][field];
    const child = readAdapter(fixture(t, document, `${field}.json`).path);
    assert.ifError(child.error);
    assert.notEqual(child.status, 0, `${field}: malformed selected data must fail at the boundary`);
    assert.ok(
      child.stderr.includes(`Evidence record ${id}`) && child.stderr.includes(field),
      child.stderr,
    );
  }
});

test('nested provenance types and nonfinite measurements fail before rendering', (t) => {
  for (const [field, value] of [
    ['evidence_revision', 42],
    ['relflow_revision', []],
    ['script_sha256', 42],
    ['packages', { relflow: 42 }],
  ]) {
    const document = structuredClone(snapshot);
    const id = Object.keys(document.results)[0];
    document.results[id].provenance[field] = value;
    const child = readAdapter(fixture(t, document, `${field}.json`).path);
    assert.notEqual(child.status, 0);
    assert.ok(child.stderr.includes(`provenance.${field}`), child.stderr);
  }
  const document = structuredClone(snapshot);
  const id = Object.keys(document.results)[0];
  document.results[id].metrics.nonfinite = 'overflow-fixture';
  const raw = JSON.stringify(document).replace('"overflow-fixture"', '1e400');
  const child = readAdapter(fixture(t, raw, 'overflow.json').path);
  assert.notEqual(child.status, 0);
  assert.ok(child.stderr.includes('metrics'), child.stderr);
});

test('publication report fields used by links and status retain their declared types', (t) => {
  for (const [field, value] of [
    ['release', {}],
    ['run_id', 42],
    ['passed', 'yes'],
    ['github_run', []],
    ['github_run', { html_url: {} }],
  ]) {
    const document = structuredClone(snapshot);
    document.report[field] = value;
    const input = fixture(t, document, `${field}.json`);
    const child = readAdapter(input.path);
    assert.ifError(child.error);
    assert.notEqual(child.status, 0);
    assert.ok(child.stderr.includes(input.path) && child.stderr.includes('report'), child.stderr);
    assert.equal(child.stdout, '');
  }
});

test('an execution error remains a complete selected record with its original details', (t) => {
  const document = structuredClone(snapshot);
  const id = Object.keys(document.results)[0];
  document.results[id] = {
    ...document.results[id],
    outcome: 'error',
    error: 'A recorded execution failure.',
    checks: {},
    metrics: {},
  };
  document.report.passed = false;
  const actual = loaded(fixture(t, document).path);
  assert.deepEqual(actual.entries.find((entry) => entry.id === id).result, document.results[id]);
});

test('results outside the active catalog remain in the raw publication without creating pages', (t) => {
  const document = structuredClone(snapshot);
  document.results.P999 = { ...record(), outcome: 'error', error: 'A retained execution failure.' };
  const input = fixture(t, document);
  const actual = loaded(input.path);
  assert.ok(actual.entries.every((entry) => entry.id !== 'P999'));
  assert.deepEqual(JSON.parse(actual.selection.raw).results.P999, document.results.P999);
});

function sourceFixture(t) {
  const directory = mkdtempSync(join(tmpdir(), 'rock-docs-evidence-source-'));
  t.after(() => rmSync(directory, { recursive: true, force: true }));
  const workspace = join(directory, 'docs/starlight');
  const sourcePath = 'evidence/experiments/fixtures/source.py';
  mkdirSync(join(workspace, 'src/data'), { recursive: true });
  mkdirSync(join(directory, 'evidence/experiments/fixtures'), { recursive: true });
  mkdirSync(join(directory, 'evidence/research'), { recursive: true });
  const source = `# %% [markdown]
# ---
# title: Exact source fixture
# description: A static source-binding fixture.
# evidence-group: Fixtures
# evidence-review:
#   question: Can this fixture select its own presentation?
#   modeling: The fixture declares a numeric score without running any model.
#   note: This is a static test fixture.
#   comparison:
#     kind: Recorded measurement
#     metric: RMSE
#     definition: A numeric score for this fixture.
#     direction: Lower is better
#     conditions:
#     - label: Score
#       path: [score]
#       format: number
# evidence-readout:
#   - label: Score
#     metric: [score]
#     format: number
# ---
# The current source illustration.
#
# ## Example
# One static example.
# %%
# /// script
# dependencies = []
# [tool.relflow.evidence]
# capability = "computation"
# types = ["Number"]
# conditions = ["fresh-records"]
# ///
@experiment("P001", seed=0)
def run():
    return "current source bytes"
`;
  writeFileSync(join(directory, sourcePath), source);
  const record = structuredClone(snapshot.results[Object.keys(snapshot.results)[0]]);
  record.provenance = {
    ...record.provenance,
    evidence_revision: 'a'.repeat(40),
    script_sha256: createHash('sha256').update(source).digest('hex'),
  };
  record.metrics = { score: 0.75 };
  const document = {
    schema: 1,
    version: 'source-fixture',
    revision: 'a'.repeat(40),
    report: {},
    results: { P001: record },
  };
  const publication = join(workspace, 'publication.json');
  function read() {
    writeFileSync(publication, JSON.stringify(document));
    return loaded(publication, workspace).entries[0];
  }
  return { directory, workspace, sourcePath, source, record, document, read };
}

test('a matching current source is shown with its exact digest and derived readout', (t) => {
  const input = sourceFixture(t);
  const entry = input.read();
  assert.equal(entry.sourceMatches, true);
  assert.equal(entry.source, input.source);
  assert.equal(entry.sourceSha256, input.record.provenance.script_sha256);
  assert.ok(entry.code.includes('current source bytes'));
  assert.ok(entry.sections.intro.includes('current source illustration'));
  assert.equal(entry.readouts[0].value, 0.75);
});

test('current Evidence links retain real site routes and their anchors', (t) => {
  const input = sourceFixture(t);
  for (const name of ['advanced/data-modules', 'concepts/data-types']) {
    const file = join(input.workspace, `src/content/docs/${name}.mdx`);
    mkdirSync(join(file, '..'), { recursive: true });
    writeFileSync(file, '# Current guide\n');
  }
  const source = input.source.replace(
    '# The current source illustration.',
    '# [Data modules](/advanced/data-modules/#preprocessing) and [types](/concepts/data-types/).',
  );
  writeFileSync(join(input.directory, input.sourcePath), source);
  input.record.provenance.script_sha256 = createHash('sha256').update(source).digest('hex');
  const entry = input.read();
  assert.ok(entry.sections.intro.includes('[Data modules](/advanced/data-modules/#preprocessing)'));
  assert.ok(entry.sections.intro.includes('[types](/concepts/data-types/)'));
});

function archivedFixture(t) {
  const input = sourceFixture(t);
  const sha256 = input.record.provenance.script_sha256;
  const retained = join(input.workspace, 'source', input.sourcePath.replace('evidence/', ''));
  mkdirSync(join(retained, '..'), { recursive: true });
  writeFileSync(retained, input.source);
  rmSync(join(input.directory, input.sourcePath));
  function write() {
    writeFileSync(join(input.workspace, 'publication.json'), JSON.stringify(input.document));
  }
  write();
  return { ...input, sha256, retained, write };
}

test('deleting an experiment removes it independently of retained results and sources', (t) => {
  const input = archivedFixture(t);
  const path = join(input.workspace, 'publication.json');
  let actual = loaded(path, input.workspace);
  assert.deepEqual(actual.entries, []);
  assert.deepEqual(JSON.parse(actual.selection.raw).results.P001, input.record);
  rmSync(input.retained);
  actual = loaded(path, input.workspace);
  assert.deepEqual(actual.entries, []);
});

test('pending selections include current declarations without resurfacing archived-only proofs', (t) => {
  const input = archivedFixture(t);
  input.document.results = {};
  input.write();
  const actual = loaded(join(input.workspace, 'publication.json'), input.workspace);
  assert.deepEqual(actual.entries, []);
  assert.deepEqual(JSON.parse(actual.selection.raw).results, {});
});

test('new Group declarations stay unmeasured beside the selected historical records', () => {
  const entries = currentSnapshot().entries;
  for (const id of ['P108', 'P109', 'P110', 'P111', 'P112']) {
    const entry = entries.find((entry) => entry.id === id);
    assert.ok(entry, `${id}: retain its current declaration`);
    assert.equal(entry.result, null);
    assert.ok(entry.readouts.every((readout) => readout.value === null));
  }
});

test('archived source bytes override changed current code only when their digest matches', (t) => {
  const input = sourceFixture(t);
  const archived = input.source.replaceAll('current source', 'archived source');
  const digest = createHash('sha256').update(archived).digest('hex');
  input.record.provenance.script_sha256 = digest;
  const retained = join(input.workspace, 'source', input.sourcePath.replace('evidence/', ''));
  mkdirSync(join(retained, '..'), { recursive: true });
  writeFileSync(retained, archived);
  const entry = input.read();
  assert.equal(entry.sourceMatches, true);
  assert.equal(entry.source, archived);
  assert.equal(entry.sourceSha256, digest);
  assert.ok(entry.code.includes('archived source bytes'));
  assert.ok(!entry.code.includes('current source bytes'));
  assert.equal(entry.readouts[0].value, 0.75);
});

test('unavailable, malformed, and mismatched source fingerprints never borrow current code', (t) => {
  const input = sourceFixture(t);
  for (const digest of [undefined, 'invalid', 'b'.repeat(64)]) {
    input.record.provenance.script_sha256 = digest;

    const entry = input.read();
    assert.equal(entry.sourceMatches, false);
    assert.equal(entry.source, '');
    assert.equal(entry.code, '');
    assert.ok(Object.values(entry.sections).every((section) => section === ''));
    assert.deepEqual(entry.readouts, []);
    assert.deepEqual(
      entry.result,
      JSON.parse(JSON.stringify(input.record)),
      'The full execution record remains available when source is unavailable.',
    );
  }
});

test('Git fallback shows only the exact historical source at its recorded revision', (t) => {
  const input = sourceFixture(t);
  const historical = input.source.replaceAll('current source', 'historical source');
  writeFileSync(join(input.directory, input.sourcePath), historical);
  const git = (args) =>
    execFileSync('git', args, {
      cwd: input.directory,
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    }).trim();
  git(['init', '--quiet']);
  git(['add', input.sourcePath]);
  git([
    '-c',
    'user.name=Evidence fixture',
    '-c',
    'user.email=evidence-fixture@example.invalid',
    '-c',
    'commit.gpgsign=false',
    'commit',
    '--quiet',
    '-m',
    'Static source fixture',
  ]);
  const revision = git(['rev-parse', 'HEAD']);
  writeFileSync(join(input.directory, input.sourcePath), input.source);
  input.record.provenance.evidence_revision = revision;
  input.record.provenance.script_sha256 = createHash('sha256').update(historical).digest('hex');
  const entry = input.read();
  assert.equal(entry.sourceMatches, true);
  assert.equal(entry.source, historical);
  assert.equal(entry.sourceRevision, revision);
  assert.equal(entry.sourceSha256, input.record.provenance.script_sha256);
  assert.ok(entry.code.includes('historical source bytes'));
  assert.ok(!entry.code.includes('current source bytes'));
  input.record.provenance.script_sha256 = 'b'.repeat(64);
  const mismatch = input.read();
  assert.equal(
    mismatch.sourceMatches,
    false,
    'Git content must match the recorded digest before presentation.',
  );
  assert.equal(mismatch.source, '');
  assert.equal(mismatch.code, '');
});

test('a released plot bundle supplies exact historical proof source in a shallow checkout', (t) => {
  const input = sourceFixture(t);
  const historical = input.source.replaceAll('current source', 'released source');
  input.record.provenance.script_sha256 = createHash('sha256').update(historical).digest('hex');
  const directory = join(input.directory, 'docs/assets/evidence-plots');
  mkdirSync(directory, { recursive: true });
  const path = join(directory, 'P001-proof.py');
  writeFileSync(path, historical);
  const entry = input.read();
  assert.equal(entry.sourceMatches, true);
  assert.equal(entry.source, historical);
  assert.equal(entry.sourceRevision, input.record.provenance.evidence_revision);
  assert.deepEqual(entry.result, input.record);
  writeFileSync(path, historical + '\n# changed after publication\n');
  assert.equal(input.read().sourceMatches, false);
});

test('a renamed current proof keeps the recorded path from its retained CI report', (t) => {
  const input = sourceFixture(t);
  const retained = join(input.workspace, 'source/experiments/original/proof.py');
  mkdirSync(join(retained, '..'), { recursive: true });
  writeFileSync(retained, input.source);
  const entry = input.read();
  assert.equal(entry.sourcePath, 'evidence/experiments/original/proof.py');
  assert.ok(entry.sourceUrl.endsWith('/evidence/experiments/original/proof.py'));
  assert.equal(entry.sourceMatches, true);
});
