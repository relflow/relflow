# P110 masked-identity lab protocol

Declared before execution: use fresh seeds 11001, 11002 and 11003 and the
masked-label Group source with SHA-256
`affdf92de9d0f11d11c371c475ece741fcea75537894af564113bc304cde1caa`.
Train for 4,800 updates with a 50% Group reconstruction mask. Require final
assignment ARI ≥ 0.95, held-out known-identity accuracy ≥ 0.90,
shuffled-identity accuracy ≤ 0.35, all 50 identities assigned, and evaluation
state unchanged. Run all three seeds and retain every result. The baseline
Cluster P020 remains unchanged. This is a new candidate-only Group experiment,
not a paired comparison with P020 or a hosted CI result. Use the exact PR
candidate wheel `69f7162ed8ca5460036c96e5f26604d7ba72c8493392224e75c3553bbbc9a030`
in network-disabled two-CPU, 4 GiB containers with one Torch thread.
