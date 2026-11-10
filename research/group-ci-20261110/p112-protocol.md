# P112 extended-budget lab protocol

Declared before execution: use fresh seeds 11201, 11202 and 11203 and Group
tradeoff source SHA-256
`216d0e71af340646cca65473dc3d7c6b80435b5360bf688b48cc0fa237be92bc`.
Each of the twelve Group, Category, oracle and shuffled-control fits trains
for 2,400 updates. Keep the existing dense Group and oracle accuracy gates,
shuffled-control gate, and finite-result checks unchanged. Run all three seeds
and retain every result, including unmet checks. The baseline Cluster P095
remains unchanged. This is a new candidate-only Group experiment, not a paired
comparison with P095 or a hosted CI result. Use the exact PR candidate wheel
`69f7162ed8ca5460036c96e5f26604d7ba72c8493392224e75c3553bbbc9a030`
in network-disabled two-CPU, 4 GiB containers with one Torch thread.
