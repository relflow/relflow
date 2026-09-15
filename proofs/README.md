# Modeling proofs

Each proof is one Python file: a standalone, seeded synthetic experiment and
the source of its illustrated documentation page. Read from the explanation
through data generation, model, training, and controls in that same file.
Proofs demonstrate learned behavior; they are separate from the unit suite
and do not measure training speed.

Browse the [illustrated proof catalog](../docs/proofs.qmd) for model trees,
contrasting records, insights, results, and the complete source of every script.

## Layout

Scripts are grouped by the behavior they examine:

| Directory | Focus |
| --- | --- |
| `aggregation/` | Sums, moments, ranks, and weighted reductions |
| `calibration/` | Separating learnable signal from noise |
| `cluster/` | Cluster objectives and capacity |
| `identity/` | Equality for unseen identifiers |
| `mutations/` | Retention, extension, reset, deactivation, and deletion |
| `relational/` | Retrieval, grouped context, and field relationships |
| `state/` | Missing values and real zeroes |
| `structure/` | Reduction and nested cardinality |
| `temporal/` | Calendar relationships |

The shared runner, reporting, rendering, source readers, and `results.yaml` stay
at the root of `proofs/`. Directory categories are broader than the families
shown in the catalog; each script's page metadata defines its family.

## Mutation proofs

Five mutation experiments are registered in the 50-script catalog:

- **P046**: repeated neutral edits preserve a learned numerical and categorical task.
- **P047**: add a hidden output, measure immediate retention, and learn the new task.
- **P048**: reset one hidden output, measure localized loss, and relearn it.
- **P049**: deactivate an informative input and restore it through updates, overrides, and save/load.
- **P050**: delete an informative input and adapt toward the remaining-information limit.

Each starts from a trained model and includes state comparisons, prediction
checks, controls, and checkpoint round trips. Adaptation uses a fresh optimizer
and rehearses the original labels. Read the recorded outcomes before treating
a proposed preservation property as established.

```bash
uv run python proofs/run.py P046 P047 P048 P049 P050
```

The broader [mutation designs](SPEC.md#mutation-proofs) also cover informative
inputs, repeated context, masking-role changes, capacity growth, and
composed edits. Those additional scenarios remain planned.

## Run an experiment

```bash
uv run python proofs/run.py --list
uv run python proofs/run.py P014
PYTHONPATH=proofs uv run python proofs/aggregation/raw_value_weight_sum.py
```

The ID identifies the experiment even if its title or filename changes. Never
reuse an ID for a different claim. `proofs/results.yaml` maps IDs to scripts and
documentation pages. Prefer the ID command; direct script commands use
`PYTHONPATH=proofs` to find shared reporting and are run from the repository root.

```bash
uv run python proofs/run.py P014 P025 --accelerator gpu
uv run python proofs/run.py P014 --seed 42
uv run python proofs/run.py P014 --steps 2 --output /tmp/proof-smoke.yaml
make proofs
```

Without overrides, each script uses its original seed and full training budget
on CPU, with one Torch thread. `--accelerator gpu` selects CUDA. The suite runs
serially; completed experiments record every behavioral check, including ones
that were not met. Execution errors are recorded and return a nonzero exit code;
the suite continues with the remaining scripts.

A `--steps` cap is a smoke run. It checks that training and evaluation execute,
but it does not establish learning capability or replace full results in the
catalog. `--output` writes a separate evidence file for local work or isolated
lab runs. Omit it to append to the canonical results file.

## Read the scripts

A local generator yields ordinary nested Python records. Its RNG is seeded
inside the generator, so every invocation reproduces the split. A
`SyntheticDataModule` receives fresh generator factories for training,
validation, and testing. Held-out records may be materialized for evaluation
and matched corruption controls. There is no Arrow or Polars dataset boilerplate.

Each script defines its own model, training procedure, evaluations, and gates.
Only reporting is shared between experiments: command-line handling,
environment and source fingerprints, and atomic YAML recording. Importing
a script does not train it.

Keep the claim, data-generating relationship, visible and hidden fields, and
controls obvious in the code. Use short docstrings and comments to explain
experimental decisions. Store measured results in YAML, not in script comments.
Preserve the original target when a corruption is intended to break its
relationship to the inputs; update it only for an explicit algebraic
transformation.

## Author the page

Use `# %% [markdown]` cells for commented Markdown and `# %%` cells for Python.
The first Markdown cell contains the YAML page header: title, description,
family `categories`, `proof-id`, `code-fold: true`, and
`execute: {enabled: false, eval: false}`. Both execution settings also apply
to the docs project. Quarto renders this format directly without Jupyter or
running the experiment.

Keep the current insights, shared Typst tree, and contrasting single-record
YAML examples beside the code they explain. The existing scripts have four
foldable code sections: setup, data and controls, training and evaluation,
and the reporting entrypoint. The `proof` shortcodes insert recorded status,
evidence, and reproduction commands with a script download.

Edit the canonical file under `proofs/<category>/`. The docs build stages ignored copies
under `docs/proofs/<category>/<slug>.py`; the registry keeps their published URLs
stable. Do not author a second page or edit a generated copy.

`proofs/render.py` prepares those copies and recorded evidence. `make render`
removes the generated `docs/proofs/` directory afterward; the catalog stays in
`docs/proofs.qmd`.
Direct Quarto renders and previews retain their inputs; `make clean` removes
them and the rendered site.

## Recorded evidence

`results.yaml` keeps historical reports separate from actual script runs.
Historical gate clearance remains a statement that a gate was reported met;
it is never converted into a fabricated numerical score. Each new run records
its seed, accelerator, budget override, duration, package versions, source
fingerprint, measured metrics, and individual checks. Errors are recorded too.
Historical interpretations remain archived in YAML; the script contains the
current interpretation. A Python AST fingerprint detects code changes without
marking Markdown edits or formatting as a changed experiment. Full file hashes
remain in the run provenance. Moving scripts does not rewrite recorded hashes
or measurements.

The docs show the latest full result, with behavioral checks collapsed by
default and no historical comparison. A proof without a full run is marked
**Not run**. A new full run that misses a gate is shown as **Gates not met**;
an execution error is distinct from model quality. Smoke runs do not change
capability status. Editing or rendering the docs never executes a proof.

One seeded success remains provisional. Expected information-loss examples
are useful limitations, and mechanistic cluster diagnostics do not establish
held-out partition recovery. [SPEC.md](SPEC.md) describes the control and
multi-seed promotion protocol; [the reduction specification](../ATTENTION_AND_BRANCH_REDUCTION.md)
explains the architectural boundaries being explored.
