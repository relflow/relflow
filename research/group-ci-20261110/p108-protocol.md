# P108 nonlinear-regime lab protocol

Declared before execution: use fresh seeds 10801, 10802 and 10803 and the
five-regime Group source with SHA-256
`1b2cb4f0d23a5e3471706d9cf2d0f84c229fc0f2c200be6b50f30bbc12639fe2`.
Train the stable and identity-shuffled arms for 4,800 updates each. Require
stable ARI ≥ 0.95, held-out normalized RMSE ≤ 0.35, no-regime ARI ≤ 0.15,
no-regime and shuffled-test normalized RMSE ≥ 0.85, and an error increase of
at least 0.40 on shuffled test identities. Also require finite measurements,
all 100 identities assigned, and unchanged assignments on evaluation. Run all
three seeds and retain every result. Baseline Cluster P018 remains unchanged;
these results are candidate-only research, not a paired comparison or CI run.
Use PR candidate wheel
`69f7162ed8ca5460036c96e5f26604d7ba72c8493392224e75c3553bbbc9a030`
in network-disabled two-CPU, 4 GiB containers with one Torch thread.
