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
uv run evidence/experiments/run.py --list
uv run evidence/experiments/run.py --wheel /path/to/matching-relflow.whl P001 --steps 2
uv run --project docs --with /path/to/matching-relflow.whl pytest -c docs/pyproject.toml evidence/tests/runtime
```

Each experiment declares its Python requirement and packages in standard
[PEP 723 metadata](https://packaging.python.org/en/latest/specifications/inline-script-metadata/).
uv creates a cached environment for that script, independent of the repository's
editable SDK and documentation environment. The ID runner is itself a uv script;
listing and scaffolding require only PyYAML, without model dependencies.

Supply an actual wheel with `--wheel` to test a development or baseline build.
The runner replaces the script's relflow requirement, preserving requested
extras, and resolves all its dependencies together. It clears inherited Python
import paths and records the wheel's SHA-256 in each result's provenance.
Without `--wheel`, uv installs the published relflow package declared by the script.
These class-based schemas require relflow 0.5.0 or later; use `--wheel` while
developing against the unreleased branch.
Runtime tests still use the separate documentation environment shown above.
A two-step run checks execution only. Each execution
writes a new JSON file under evidence/runs/local; --output-dir selects another
directory. Execution never edits the reviewed publication selection.

You can also invoke an experiment directly from any working directory using its
absolute path:

```sh
uv run /path/to/relflow/evidence/experiments/P001.py --steps 2
```

Keep the evidence checkout alongside the script: shared reporting and frozen
policies are source files, rather than packages downloaded from an index.
Local dependency resolution is convenient for development; exact reproduction
uses the wheel, dependency lock, source revision, and resources retained by CI.
Hosted CI continues to execute both supplied wheels in its frozen, offline
containers; it does not resolve packages from script headers during a proof.

Each experiment also owns its reviewed tags inside the same block:

```python
# [tool.relflow.evidence]
# capability = "computation"
# types = ["Number"]
# conditions = ["fresh-records", "transformations"]
```

The shared vocabulary in `docs/starlight/src/evidence/catalog.ts` defines valid
names and labels, with no experiment-ID membership lists. Documentation filters
and leaf reference links read assignments from each script. Choose exactly one
capability, zero or more primary leaf types, and at least one evaluation condition.
Scaffolds use `unreviewed` placeholders that must be replaced before docs checks
pass. Retained historical scripts without tags remain byte-for-byte unchanged;
retired entries appear as archived, without inferred tags.

## Define an experiment

Declare each experiment's typed data contract in an `rf.Model` subclass. Branches
may use named classes or inline `rf.Branch(...)` definitions; inline branches are
useful for small local structures. Module-level declarations allocate no neural
weights; the experiment constructs the model when it runs. Use subclasses for
fixed variants and function-local declarations when field options depend on an
experimental arm. Constructor arguments on declared classes configure architecture
and placement. Static checks require class-based models and permit inline branches.

Experiments that share the historical XS capacity use `CONTROL` from
`experiments/policies.py`:

```python
import relflow as rf
from policies import CONTROL


class Model(rf.Model):
    value = rf.Number
    history = rf.Branch(length=8, value=rf.Number)
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

Each file is a self-contained experiment. Its YAML frontmatter owns the reviewed
question, modeling explanation, primary comparison, criterion labels, diagnostic
plan, optional publication highlights, and optional scheduling estimates. Custom
`(bundle, spec)` plot functions live in that same file with their own imports.
Adding or removing it updates the catalog without editing any side list. See
[SPEC.md](SPEC.md#self-contained-experiment-files).

All studies live directly in `experiments/` as `Pxxx.py` and run in the required
hosted catalog. There are no lanes or optional exclusions. Each script owns its uv
dependencies and classification tags. Use `--list` to inspect the full catalog,
or pass IDs to run a local selection explicitly:

```sh
uv run evidence/experiments/run.py --list
uv run evidence/experiments/run.py --wheel /path/to/relflow.whl P099 P100
```

Create an authoring file from the repository root:

```sh
uv run evidence/experiments/run.py \
  --new --title "Can a model learn a median?" --group Aggregation
```

This writes `evidence/experiments/Pxxx.py` using the next stable ID, with a
literal default seed of `0`, presentation metadata, example sections, and the
decorator and reporting entrypoint. The filename matches the declared ID.
The title is required; `--group` defaults to an explicit `Unreviewed` placeholder.
Use `--id` or `--seed` to override those values. Tags and frontmatter supply
categorization independently of the file path.
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
uv run evidence/experiments/run.py --list
uv run evidence/experiments/run.py --wheel /path/to/matching-relflow.whl \
  P083 --steps 2
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
selects an explicit ancestor for repeatable manual comparisons. The active suite
requires relflow >=0.5.0. A pre-0.5 event base, or an authenticated forced push
with rewritten history, selects the fixed 0.5 anchor described in
[hosted.md](hosted.md). The report retains both the actual event base and the
measured baseline in its comparison receipt. Routine
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

Starlight combines canonical declarations with the CI publication selected by
`RELFLOW_EVIDENCE_PUBLICATION` through `docs/starlight/src/evidence/data.ts`.
The publication's sibling `source/` directory comes from its verified CI report.
Without a selection, local builds show pending measurements. Generated results,
proof sources, and historical tutorial archives are not checked into Git.
Hosted tag docs require their exact
version's public release asset and source revision. Branch and PR builds use
the latest formal release at version >=0.5.0, with its measured revision and
proof sources visible. Measurements remain pending until a supported release exists.
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
