# Proof diagnostics

Each proof has an authored visual brief in `views.json`: the question to make
visible, the comparison, and the limits of its interpretation. The core suite
uses actual full-budget CI captures. Optional research pages have generator
illustrations until their declared resources and learning records are available.

Capturing is an explicit operation, separate from the required paired Evidence
gate. Dispatch the documentation workflow on the review branch:

```sh
gh workflow run docs.yml --ref <branch> -f diagnostic_plots=true
gh run download <run-id> --name proof-visualization-bundles-attempt-1 --dir /tmp/proof-diagnostics
```

`diagnostic_ids=P025,P085` selects a named subset when repairing a recorder
failure. Keep original attempts and select replacement inputs explicitly; never
rerun an unmet learning check in search of a passing result. The current review
uses run `42432931869` for 94 core proofs and run `42437236022` for P025/P085,
after fixing spawn imports and gradient-safe serialization. All 96 executions
are full budget: 85 meet their checks and 11 do not. The earlier infrastructure
attempt is retained as run `42432383859`. No canonical proof or result was edited.

The prepare job builds the exact wheel and freezes its CPU image and catalog.
Twelve shards run two bounded offline containers each. Return-event monitoring
retains already-computed proof readouts without extra inference calls or random
draws. Arrays and Arrow tables are content-addressed IPC artifacts; completed
frames retain the original call context. Each bundle includes the unchanged
proof, complete record, source/wheel hashes, runtime identity, and observation
manifest. Process-pool trials retain their returned metrics; spawned workers do
not inherit monitoring. Capture overhead makes these unsuitable as performance
benchmarks.

Generate figures explicitly from the downloaded verified bundles:

```sh
uv run --project docs --with matplotlib==3.10.7 --with pyarrow==24.0.0 \
  python -m evidence.diagnostics.build --bundles /tmp/proof-diagnostics \
  --output /tmp/proof-plots --archive /tmp/proof-visualizations.zip
```

A complete diagnostic dispatch also uploads `proof-visualization-site-bundle-attempt-1`
with the ready-to-host ZIP. The builder checks bundle checksums before reading observations. Reviewed SVGs, figure/example JSON, exact records and source copies, and
compressed manifests are packaged as one immutable preview attachment outside
Git. `preview.json` pins its URL, size, and SHA256. Only the authored CSS and
JavaScript live in `docs/assets/evidence-plots/`; generated files there are
ignored. Keep PNG review images and large raw Arrow bundles outside Git too.
The bundle includes `review.json`, binding every figure and example to its
checksums. Review all figures for correct populations, units, control labels,
matched scales, and legible light/dark layouts. Keep projections fitted to
training/reference data and explain their scope. Missing learning results stay
Missing even when a generator diagram is available.

Read the Docs external PR builds fetch and verify the pinned static bundle,
then select its review records automatically. Release builds retain their
existing frozen publication path.
A figure renders only beside the identical captured record; released tables
cannot silently acquire figures from a different execution. To select the same
review locally, run from the repository root:

```sh
uv run --project docs python -c 'from pathlib import Path; from evidence.diagnostics.review import prepare, selection; prepare(Path.cwd()); print(selection(Path.cwd()))'
RELFLOW_EVIDENCE_PUBLICATION="$PWD/.cache/evidence/diagnostic-review.json" make dev
```

Static validation and rendering require only the isolated docs environment.
Matplotlib, Arrow, relflow, Torch, and Lightning are not imported by site builds.
The PR's diagnostic selection is an unpaired visualization run; package release
publication and the full paired CI gate retain their existing ownership.
