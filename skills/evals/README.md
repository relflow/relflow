# Behavioral evaluation

Run these requests with the packaged skill in each supported agent. Save the agent
and model version, skill digest, input fixture, produced schema/preprocessing,
executed check results, and rubric outcomes. These are manual forward-evaluation
cases; local script tests do not certify agent behavior.

`transactions.json` and `model.yaml` are executable smoke fixtures. Treat each row
as an independent prediction anchor; repeated customer IDs do not authorize merging
those rows. The fixture assumes histories were already sorted and censored and
`age_days` was derived upstream. It is not a learned or benchmark-selected schema.
Run the profiler and validator against it, including `--encode`, before testing an
extracted bundle in an agent.

| Request / fixture | Observable outcome |
| --- | --- |
| Infer a schema from CSV IDs `0012`, `0013`, amounts, and a free-text note; no task stated | Keeps ID spelling, investigates semantics, leaves target unresolved, proposes explicit numeric conversion |
| Infer a supervised customer schema from transaction histories | States observation unit/cutoff, keeps related fields aligned, separates candidate pool from retained cap |
| Review a join where two parent rows share a key and each has three child events | Detects artificial multiplication, preserves the two anchors and three real events, requests a structural counterexample |
| Choose a default using the archived relarena preset scores | Labels historical test exposure; does not copy per-task test winners or treat research priors as public defaults |
| Inspect null, empty, and null-item histories plus a numeric array | Preserves structural distinctions; investigates vector versus sequence interpretation |
| Validate a user-registered custom field type | Uses the actual extension registration and its contracts rather than a closed datatype allowlist |
| Assess a capped nested schema that fits a tiny sample | Reports conservative cap envelope and limits; does not claim full-fit memory or quality proof |
| Interpret the November 4 dedup and mixed-crop closeout | Separates later experimental controls from the frozen baseline; preserves single-seed and recovery caveats |

Check selection as well as execution: ordinary prediction or an unrelated code edit
should not activate schema inference just because relflow is present. Compare
observable decisions and valid outputs, not exact prose or a fixed number of steps.
