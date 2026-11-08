# GitHub-hosted CPU evidence

The required Evidence workflow runs on standard GitHub-hosted Ubuntu CPU
runners. A trusted helper prepares the exact inputs, twelve shards run the full
paired catalog, and one collector validates the complete comparison. Shared
Docker operations live in `runtime.py`; `worker.py` executes one planned task
inside each isolated container.

## Workflow and trust

`ci.yml` remains the entry point for pull requests, merge groups, main pushes,
and explicit manual validation. Its shared package build supplies one candidate
wheel to package tests, Evidence, and release. The baseline comes from the
actual comparison revision: PR or merge-group base, pre-push commit, or the
manual first parent or explicit ancestor selected by `evidence_base`.

`evidence.yml` builds and retains the exact wheel inputs and locked dependency
export. It calls `evidence-hosted.yml` at a reviewed full commit SHA. The helper
pins a separately reviewed executor revision. Selected experiment sources are
data to the host coordinator; their Python executes only inside task containers.
Forks cannot replace that trusted code or gain a privileged execution path.
Checkouts do not retain repository credentials, and containers receive none.

The reviewed resolver uses pinned uv to resolve Linux CPU wheel metadata while
constraining selected distribution versions to the candidate's lock. Torch uses
the corresponding `+cpu` wheel; unused CUDA dependencies do not enter the CPU
runtime. Inputs retain `runtime-constraints.txt`, `requirements.txt`,
`resolution.json`, and `uv-version.txt`. The prepared image runs `pip check` and
records its installed package inventory and CPU Torch identity. Baseline and
candidate receive that same dependency image.

The helper has three phases:

1. **Prepare:** bind the input request to the GitHub event, freeze the complete
   manifest and twelve-shard assignment, and prepare the dependency-only runtime
   image. Retain the exact wheels, requirements, source snapshot, image identity,
   and archive hashes.
2. **Shard:** receive verified inputs, record the actual runner identity, and
   run both roles for every assigned proof with unchanged seeds and budgets.
   Baseline and candidate for a proof always run on the same VM.
3. **Collect:** verify every shard and the exact planned task union, then apply
   the original paired gate. The job is named **Collect full paired suite**.
   Missing, duplicate, altered, cancelled, or failed execution cannot become a
   passing `CI / required` result.

Keep helper and executor pins immutable. To update trusted mechanics, first
review and freeze the executor commit, then freeze a helper commit pointing to
it, then update the consumer to that helper. A suite-only revision does not
authorize changing host code. The consumer and publication verifier must agree
on the reviewed executor identity. Historical reports retain their original
executor and helper pins; do not rewrite their provenance during migration.

## Isolation and parallelism

Twelve shard jobs may run concurrently, subject to the repository's available
GitHub Actions concurrency. Each runner permits at most two active task
containers. Every container retains its two-CPU ceiling, 4 GiB memory limit,
512-PID bound, original timeout, and declared thread policy. P025 retains its
internal seed panel and CPU behavior.

Schedule individual tasks longest first within each shard. The two roles for a
proof may run concurrently on their shared VM, keeping both worker slots useful
when a long pair remains. The two-container limit applies across all roles.

Each container has no network, a read-only root, dropped capabilities, and no
new privileges. Its selected experiment and declared shared runtime files are
read-only; only its own result directory and bounded temporary storage are
writable. Other experiments, host home directories, Docker sockets, credentials,
and caches are absent from worker mounts. The supplied wheel is installed in an
offline image layer, with no dependency installation inside each task.

Different shard VMs may have different CPUs, kernels, or host identities.
Reports retain each actual identity and bind every task to its shard, rather
than treating the whole catalog as one machine. The collector checks within-pair
host/image/resource consistency and source, wheel, and task identity.

Durations guide scheduling only. They never establish a learning result,
change a seed, shorten a budget, or change a named check. The collector requires
the entire catalog, including baseline-unmet checks and every declared internal
seed panel.

`timings.json` retains per-role duration hints from a verified hosted run and
identifies its source commit, executor, and authenticated report artifact. Assign
whole pairs longest first using the sum of their role costs, then schedule
individual roles longest first within each runner. The plan freezes the timing
file digest and resolved costs so preparation, execution, and collection agree.
Missing timings use positive fallback costs, preventing unmeasured experiments
from accumulating on a runner at zero estimated cost. Refresh hints from validated
reports while retaining their source receipt; forecasts never replace measurements.

## Caches and attempts

The shared dependency-only runtime image uses a bounded Docker Buildx layer
cache to reduce preparation. Keys bind the runner OS/architecture, reviewed
executor, and locked requirements. Only trusted internal main pushes and
explicit dispatches save this cache. A cache miss builds the image. Preparation
ships the resulting image once as a one-day artifact; every shard verifies its
archive digest and expected image identity. Wheels and source snapshots remain
exact run inputs. Neither candidate wheels nor learning results enter the
dependency cache; both measurement roles execute freshly.

Historical authenticated reports remain valid descriptions of their original
executions. They cannot become hosted measurements by replacing a host field or
copying an old metric into a new record. Any future result reuse must preserve
original records and prove the current request's full execution equivalence.

Only bounded infrastructure setup failures and eligible Docker launch failures
before a worker record exists may retry. Retain every launch receipt and log.
A failed learning check, completed worker error, OOM, cancellation, or learning
timeout is terminal. Retry logic must not select a favorable later measurement.

GitHub workflow attempts retain their original identities. This hosted profile
rejects a shard or collector whose current attempt differs from its prepared
input request. Use **Re-run all jobs** so preparation, inputs, shards, and
collection share one attempt. Rerunning only failed jobs cannot silently mix
plans, relabel completed measurements, or overwrite earlier records. A failed
collector still retains the available shard logs and provenance for diagnosis.
A rerun cannot satisfy the gate until the complete authenticated comparison is
present.

## Validation and publication

Run the complete configured comparison without publishing a new selection:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main
```

To validate a prospective suite revision against main:

```sh
gh workflow run ci.yml --repo relflow/relflow --ref main \
  -f evidence_ref=<full-suite-commit>
```

Manual validation does not replace the required PR-triggered gate. Smoke and
partial local runs remain useful execution checks and cannot satisfy Evidence.
Use `uv run --project docs python evidence/hosted.py --help` for the coordinator's
planning, shard, and collection commands. Docker is required for actual tasks;
static docs and tooling tests do not install or import relflow, Torch, or
Lightning.

Reports include original task records and logs, the complete manifest and
source snapshot, wheel and dependency hashes, per-shard execution identity,
timing, summary, HTML, and a complete checksum inventory. Routine reports are
Actions artifacts for 90 days; tested wheel inputs and requirements are retained
for seven days. Plan, runtime, shard, and final report artifact names include
their workflow attempt, preserving earlier attempts. Artifacts support
investigation; immutable versioned Releases
preserve published evidence beyond routine artifact retention.

Only a package version release publishes a new evidence selection. The
publisher verifies the successful main CI run, original workflow attempts,
complete passing catalog, exact inputs, and report integrity before freezing
`evidence-v<version>`. Ordinary CI creates no publication commit, branch, or PR.
An already published version keeps its original evidence selection.

## Execution cost and timing

For public repositories, standard GitHub-hosted runner compute is free. GitHub
documents the public Ubuntu runner as four CPUs, 16 GB RAM, and 14 GB SSD;
dependency image archives and extracted layers must fit that disk budget. See
[runner specifications](https://docs.github.com/en/actions/reference/runners/github-hosted-runners).

GitHub Actions concurrency and storage have separate limits. The default cache
allowance is 10 GB per repository; using more after increasing the configured
allowance is billed. Image caches remain bounded and attempt artifacts expire
under their configured retention. See
[Actions limits](https://docs.github.com/en/actions/reference/limits) and
[cache usage and eviction](https://docs.github.com/en/actions/reference/workflows-and-actions/dependency-caching#usage-limits-and-eviction-policy).

Reports distinguish preparation, container execution, and total wall time.
Moving work onto several runners changes scheduling and hardware; timing alone
does not prove a learning improvement or a compute reduction. The required
behavioral comparison remains CPU-only with its full original budgets.

## Automatic proof plots

Each full paired worker retains readout observations in a separate bounded host
mount. The exact canonical result stays in the report; its observation receipt
binds the archive, recorder and manifest hashes. Both roles use the same recorder.
Raw archives are uploaded per shard for 14 days, outside report limits. The plots
job validates the complete paired union and renders the selected suite's figures
inside a read-only, network-disabled 2-CPU/4-GiB container, with host-disk temporary
storage and no credentials. It uploads the static bundle and frozen-host receipt
for 90 days. It performs no training or inference.

The post-CI publisher executes only the frozen verifier, authenticates the
original attempt, immutable helper/executor, reports, wheels and plot hashes,
and retains `evidence-ci-<id>-attempt-<n>` assets. Valid learning regressions may
be visualized; execution/provenance errors and incomplete coverage cannot publish.
No PR code executes with publication credentials. Version release publication
retains the same matched bundle in its frozen version assets. See
[diagnostics/README.md](diagnostics/README.md) for Read the Docs setup and local
review. Keep executor, helper and consumer pin updates in that order.
