# relflow evidence

This directory owns the controlled synthetic experiments, their protocol,
hosted CPU runner, runtime tests, and current publication results.
Stable P identifiers identify experiments throughout the runner, reports, and
website. An `@experiment` declaration owns each script's ID and default seed;
frontmatter owns its group and presentation options. Measured
runs, checks, seeds, environment, source provenance, and report links live in
separate publication JSON assets.

Run from the repository root:

```sh
make check
uv run --project docs python evidence/experiments/run.py --list
uv run --project docs --with /path/to/matching-relflow.whl python evidence/experiments/run.py P001 --steps 2
uv run --project docs --with /path/to/matching-relflow.whl pytest -c docs/pyproject.toml evidence/tests/runtime
```

Supply an actual wheel. Use the isolated documentation project so the root
package environment cannot substitute the editable SDK for that wheel.
A two-step run checks execution only. Each execution
writes a new JSON file under evidence/runs/local; --output-dir selects another
directory. Execution never edits the reviewed publication selection.

## Define an experiment

Keep each experiment's typed data contract visible in its own `Model` and
`Branch` classes. Module-level declarations allocate no neural weights; the
experiment constructs the model when it runs. Programmatically generated field
mappings remain useful when an arm changes the schema.

Experiments that share the historical XS capacity use `CONTROL` from
`experiments/policies.py`:

```python
import relflow as rf
from policies import CONTROL


class Model(rf.Model):
    value = rf.Number
    target = rf.Boolean(mask=True)


# Inside the experiment's execution function:
model = Model(preset=CONTROL.name)
```

This Evidence-owned policy is registered as `evidence-control-v1`. It freezes
architecture defaults and supplies no training recipe. Each experiment still
owns its field definitions, overrides, independent data splits, masks, optimizer,
budget, controls, metrics and checks. Other experiments keep their explicitly
declared capacities rather than adopting this policy for uniformity.
See the [configuration contract](SPEC.md#experimental-contract).

The extraction changes policy-name metadata from `xs` to the internal name;
architecture and seeded initialization are checked separately. Source and runtime
fingerprints change, so old measurements keep their original provenance.
Historical results are never rewritten or presented as a fresh execution.

The default **core** lane runs in required hosted CI. Additional studies with
optional resources live in `research/<category>/`; they use the same experiment
format and appear in the docs with a Research label. List or run that lane
explicitly:

```sh
uv run --project docs python evidence/experiments/run.py --lane all --list
uv run --project docs --with /path/to/provisioned-relflow.whl \
  python evidence/experiments/run.py --lane research P091
```

Read the selected page's Reproduce tab for its declared `evidence-requires`.
P091 requires TorchJD, P092 requires a real offline Text encoder, and P099/P100
require a relflow 0.4.6 wheel exporting Duration/Ordinal. They remain in research
while the paired baseline predates these datatypes. A research selection does not download
models, substitute a mocked encoder, or replace a missing measurement with a pass.

Create an authoring file from the repository root:

```sh
uv run --project docs python evidence/experiments/run.py \
  --new aggregation/median --title "Can a model learn a median?"
```

This writes `evidence/experiments/aggregation/median.py` with the next stable
ID, a literal default seed of `0`, presentation metadata, example sections, and
the decorator and reporting entrypoint. The title defaults to words from the
filename; the category comes from the first path segment. The display group
reuses that category's existing group when unambiguous, otherwise its name.
Use `--title`, `--group`, `--id`, or `--seed` to choose those values.
Add `--lane research` to scaffold a separately provisioned study. IDs are
allocated across both lanes, and research sources must describe their resources
in `evidence-requires`.
Creation reads declarations without importing experiments or model packages,
and refuses to overwrite an existing file. IDs increase beyond the current
maximum instead of filling gaps.

The generated scientific implementation raises `NotImplementedError`. Fill in
the generator, model, independent training/validation/test splits, explicit
budgets, matched control, held-out metrics, and Boolean checks. Update the
example, model tree, and metric readout to match. The file joins discovery
immediately, so it cannot satisfy the full gate until it is complete; it may
also fail authoring checks while those sections are unfinished.

List the catalog, then smoke-test the new ID with the intended wheel:

```sh
uv run --project docs python evidence/experiments/run.py --list
uv run --project docs --with /path/to/matching-relflow.whl \
  python evidence/experiments/run.py P083 --steps 2
```

Replace `P083` with the ID printed by creation. `--steps 2` caps each training
stage when the implementation honors the cap; it records a smoke run and
cannot establish a learning result. Complete experiments keep the same explicit
training and evaluation protocol when run without that cap.

Decorate the standalone `run(seed, steps, accelerator)` function and pass it to
the shared reporter under a main guard:

```python
from reporting import experiment, report


@experiment("P083", seed=8301)
def run(seed, steps, accelerator):
    # Generate independent splits, fit the model, and evaluate held-out data.
    # Keep the experiment's budgets, controls, metrics, and checks in this file.
    ...
    return metrics, checks


if __name__ == "__main__":
    report(run)
```

Choose a new stable P identifier for a new experiment. The decorator keeps the
original function callable and attaches an immutable declaration; importing a
script never runs training. Discovery scans declarations as syntax, without
importing scripts, Torch, Lightning, or relflow. IDs and seeds must be literals,
and duplicate IDs or malformed declarations fail discovery. Existing publication
records and archived frontmatter-based suites retain their original identities.
The [authoring template](templates/experiment.py.txt) supplies structure rather
than a training framework: budgets, controls, and checks remain visible in the
experiment itself.

## Present recorded results

Generated pages put the behavioral criteria before the Example, with the
selected record's met/unmet statuses. Recorded measurements use plain readouts,
labeled groups, and scalar tables. Expanded details separate Measurements,
Method, Reproduce, and Code, with provenance and the complete raw record in
their own disclosures. These views are generated from the existing scripts and
selected publication, preserving scientific code, budgets, checks, and original
measurement provenance. Missing measurements stay unrecorded. See the
[presentation contract](SPEC.md#documentation) for authoring guidance.

## Required hosted execution

The library's workflow shares one candidate wheel across tests, Evidence, and
release, and builds an exact baseline wheel when needed. It
calls `.github/workflows/evidence-hosted.yml` at a reviewed full commit SHA,
passing a separate reviewed suite SHA. The helper pins the trusted executor.
Twelve GitHub-hosted CPU shards run one experiment per Docker container, with
both roles for each experiment assigned to the same runner. Each shard runs at
most two containers concurrently. Containers have bounded CPU/memory, no network, a read-only
root, and no host home or Docker socket. The full catalog runs for both wheels
with the original seeds and budgets. Every baseline-passing check must pass on
the candidate; baseline misses remain reported. A separate collector verifies
all shards and their actual execution identities. CI cancels superseded attempts
within each PR. Errors, missing records, invalid
provenance, smoke execution, and partial coverage fail CI / required.

The combined CI gate also requires the library test matrix and static
documentation validation. The suite normally comes from the candidate commit
in this repository. A new suite can be validated explicitly with
`gh workflow run ci.yml --repo relflow/relflow --ref main -f evidence_ref=<full-suite-commit>`.
This tests the selected suite without changing the trusted executor.

The dependency-only runtime image is prepared once and delivered with its
verified identity. Each shard records its actual CPU/kernel, image, and
resource/thread settings; baseline and candidate retain those measurements.
Both roles run freshly. A dependency-cache hit never supplies a learning result
or bypasses the required collector. Historical authenticated reports retain
their original measurements, revisions, and runtime identities.

Only infrastructure operations that have not produced a worker measurement
receive bounded retries. Failed checks, worker exceptions, OOMs, cancellations,
and learning timeouts remain failures, with their logs and attempts retained.

See [hosted.md](hosted.md) for workflow trust, parallelism, and storage.

## Reports and publication

Every push to `main` or `stage/**` validates the complete paired suite against
the pre-push revision. PRs and merge groups use their actual comparison base; manual runs
use the current commit's first parent by default; `evidence_base=<full-sha>`
selects an explicit ancestor for repeatable manual comparisons. Routine
reports remain Actions artifacts for 90 days, with tested input wheels and
requirements retained for seven days.

**Only package version releases publish new evidence selections.** The release
workflow passes its exact successful main CI run, source revision, and version
to `publish-evidence.yml`. After validating complete coverage, behavioral
checks, provenance, source/wheel digests, the runtime lock, and report integrity,
the publisher renders a review site from the selected records and freezes
`evidence-v<version>` before the package's `v<version>` tag is published.

New immutable Evidence releases use versioned downloads. For relflow 0.5.0:

- `relflow-0.5.0-evidence.json`: the frozen selection and original results.
- `relflow-0.5.0-evidence-measurements.md`: the comparison table.
- `relflow-0.5.0-evidence-reports.tar.gz`: records, logs, summary, and HTML report.
- `relflow-0.5.0-evidence-inputs.tar.gz`: exact wheels and runtime requirements.
- `relflow-0.5.0-evidence-plots.zip`: matched diagnostic plots, when available.
- `relflow-0.5.0-evidence-provenance.json` and
  `relflow-0.5.0-evidence-checksums.txt`: execution identity and integrity.
- `relflow-0.5.0-evidence-reuse.json`: a verified reuse receipt, when applicable.

Previously published names and bytes remain unchanged.
Publication never edits experiment headers or the source README, commits a
bot branch, or opens a follow-up PR. An already published package version
keeps its evidence selection frozen. New measurements for publication require
a new version. Report Releases do not replace the latest package Release.

The checked-in `results.json` preserves the historical snapshot for offline
builds. Its existing download URLs are generated from that canonical file.
Starlight combines the snapshot, or the publication selected by
`RELFLOW_EVIDENCE_PUBLICATION`, with canonical declarations through
`docs/starlight/src/evidence/data.ts`. Hosted tag docs require their exact
version's public release asset and source revision. Branch and PR builds use
the latest formal release, with its measured revision and proof sources visible.
CI never creates per-commit Evidence releases or requests docs rebuilds.
No model import, cloud account, or additional API credential is required.
To execute another complete CI run without publishing a new selection:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main
```

The Release workflow publishes Evidence as part of a formal version release.
To retry the release process for the exact checked current `main` commit:

```sh
gh workflow run release.yml --repo relflow/relflow --ref main
```

`publish-evidence.yml` is reusable only by the Release workflow, with its exact
successful main CI run, source revision, and package version. It has no ordinary
CI-completion trigger or standalone preview mode.
Keep actual metrics, checks, seeds, environment, tested revisions, and source
digests bound to the measured execution. A verified reused selection retains
its original identity and separate current-request receipt. Source edits and
reader changes do not establish fresh learning results.

## Runtime efficiency

Twelve shards distribute whole proof pairs using recorded durations as scheduling
hints, with two containers per runner. Wheels are installed in isolated image
layers rather than in every task. P025 keeps all three seeds and its existing
CPU behavior. Every report retains timing and CPU/memory telemetry.

Dependency caches reduce setup; they do not remove seeds, reduce budgets, or
substitute historical measurements. Shards execute both roles freshly. Complete
coverage and the paired regression gate remain required. See
[hosted.md](hosted.md#execution-cost-and-timing).
