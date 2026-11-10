# Current Group migration validation

Before executing, this panel was declared to use the exact wheels retained by
CI run 42864454945, the reviewed baseline sources, and the five Group proof
sources listed in `sources.sha256`. Each Cluster proof runs on baseline revision
`c0cae6df38010e232c0f12c966eaa6a06fae4b94` with its original default seed and
full budget. Each Group proof runs on candidate revision
`20469b2c3e7b119ca1067e6df1f6fa46d33f708f` with its predeclared default seed
and full budget. The baseline and candidate wheels have SHA-256 digests
`a2e8e74c5c751a71fcdb83fa3659318551eda0f8241471e3883cdb9c14e8fa38`
and `ca9c1a5fde9f17c867cd7e64a1411d3854d4c1172a31c23f3fa1e65b547a8297`.

Each run uses a network-disabled, two-CPU, 4 GiB container and one Torch
thread; up to six containers ran concurrently on the lab. Smoke records are
separate and never count as learning results. Full
records, including any unmet checks, are retained without retrying a failed
measurement. These are local lab results and cannot satisfy hosted CI.

All five candidate proofs met their predeclared checks on their default seeds:

| Proof | Seed | Selected held-out result |
| --- | ---: | --- |
| P108 | 10801 | ARI 1.000; normalized RMSE 0.021 |
| P109 | 10901 | ARI 1.000; accuracy 1.000; shuffled accuracy 0.210 |
| P110 | 11001 | ARI 1.000; accuracy 0.925; shuffled accuracy 0.175 |
| P111 | 11101 | ARI 0.958; normalized RMSE 0.116 |
| P112 | 11201 | Dense Group normalized RMSE 0.244 |

The baseline proofs P019, P020, and P065 met their checks; P018, P078, and
P095 did not. They remain separate historical limitations, not comparisons
against the new Group checks. All eleven full archives were unpacked and
verified against their predeclared source and wheel hashes. Captures, logs,
the image IDs, and `verified-summary.json` are retained at
`~/relflow-research/runs/group-ci-c0ca-42864454945` on the lab. Earlier
three-seed investigations in the parent report include failed confirmations;
the default-seed results above do not establish robust convergence.
