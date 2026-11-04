---
name: relflow-schema
description: Infer and validate a relflow model schema from a dataset, nested observations, or related tables. Use when choosing field representations, observation structure, histories, and prediction targets for relflow.
license: Apache-2.0
metadata:
  version: "0.1.0"
  relflow: "0.4.4"
---

# Infer a relflow schema

Produce a reviewable declaration from the data's structure, value semantics,
relationships, and intended use. Investigate ambiguous interpretations rather than
choosing a datatype solely from its physical type or column name. The result is a
proposal supported by observations; the data need not identify a unique schema.

## Establish the task

Find the observation unit, intended prediction or representation, target, prediction
cutoff, available-at-prediction fields, split/generalization unit, metric, and
resource budget in the request or dataset documentation. Ask only for missing
choices that materially affect the result. Continue independent profiling while
those choices are unresolved. Without a task, propose representations and leave
target selection explicitly unresolved.

## Inspect and interpret

Inspect source metadata before reading values. Use the host's data tools or the
bundled profiler for local JSON, JSONL, CSV, Parquet, and Arrow IPC:

```bash
python <skill-dir>/scripts/profile_data.py observations.parquet --rows 2048
```

Resolve `<skill-dir>` from this loaded SKILL.md's actual location. Scripts require
Python 3.12 and PyArrow 20 or newer; validation also requires relflow 0.4.4 and its
dependencies. Use the intended relflow environment. Dependency installation follows
the user's environment policy. Scripts do not call model providers.

The profiler emits JSON to stdout, reads a prefix, and labels statistics as sampled.
Use other bounded samples when ordering could conceal heterogeneity. Do not infer
globally unique keys, required fields, absent variants, exact caps, or business
meaning from that prefix. Keep all-null and empty-list types open. Pass
`--examples 0` to omit example values. Read the script's limits before large inputs.

Interpret fields jointly: identifiers and references, units, vocabulary reuse,
free text versus labels, coordinates versus sequences, and timestamps versus ages.
Inspect records and counterexamples when statistics are insufficient. Read
[schema decisions](references/schema-decisions.md) for relflow contracts. Read
[relational research](references/relarena.md) when proposing joins, histories,
reductions, or experiment recipes; respect its historical evidence boundaries.

An LLM may propose structures and preprocessing. If Jev is already available, it
can rank explicit compatible interpretations using profiles, samples, surrounding
fields, and the task. Keep alternatives or gather evidence when uncertain. Provider
confidence alone does not validate a schema. Jev and MCP are optional; this skill
does not require a provider account.

## Construct and check

Use `import relflow as rf`, `rf.Model(...)`, and `rf.Branch(...)`, or the installed
version's named YAML format. Use `fields` mappings for generated names and option
collisions. Load user-selected extensions through their documented registration
entrypoints before validation; do not replace them with a closed built-in type list.

Keep joins, filtering, sorting, derived ages, and grouping in application preparation
or an `rf.Preprocessor`. Audit row identity, cutoff availability, join multiplicity,
null/empty distinctions, and finite collection limits before materialization.

For a proposed YAML model and already-preprocessed observations:

```bash
python <skill-dir>/scripts/validate.py model.yaml --data observations.parquet --rows 256
```

This checks live schema contracts, finite declared branch geometry, and sampled
Arrow compatibility without constructing neural modules or downloading text models.
It is not an encoding or training proof. Use `--encode` for a separate explicit
model-construction and encoding check when its resources are available. Check
supervised targets against labeled values as well as visible context. Keep
additional validation data outside semantic selection.

## Deliver

Return the declaration and any required preprocessing, with:

- source sample scope and intended relflow version;
- evidence for consequential interpretations, alternatives, and unresolved choices;
- task assumptions, source candidate pools, model caps/overflow, and omissions;
- executed checks, their scope, and remaining uncertainty.

When saving artifacts, prefer `model.yaml`, `profile.json`, `validation.json`, and
an inference report in the user's chosen destination. Do not publish, start paid
experiments, or change public defaults unless requested. For requested empirical
search, fix task/split/external metric/budget, include a baseline, select on
validation, and retain candidates and failures. A later validation run does not
erase earlier test exposure in historical recipes.
