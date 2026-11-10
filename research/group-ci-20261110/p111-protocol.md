# P111 extended-budget lab protocol

Declared before execution: use fresh seeds 11101, 11102 and 11103 and the
Group regime-recovery source with SHA-256
`438d4274651f175526862ee6fe0da2c1380651f32913024ac0f9a7086e62e813`.
Each stable and no-regime arm trains for 4,800 updates. Keep all existing
partition, prediction, shuffled-identity, state, and no-regime thresholds
unchanged. Run all three seeds and retain every result, including unmet checks.
The baseline Cluster P078 remains unchanged. This is a new candidate-only
Group experiment, not a paired comparison with P078 or a hosted CI result.
Use the exact PR candidate wheel
`69f7162ed8ca5460036c96e5f26604d7ba72c8493392224e75c3553bbbc9a030`
in network-disabled two-CPU, 4 GiB containers with one Torch thread.
