"""Choose a numeric reduction from a visible operation.

Claim
-----
One fixed-size Number branch can supply sum, mean, minimum, or maximum when a
root Category states which operation is requested. Each bag appears with four
contradictory answers, so neither its values nor an operation-specific constant
can solve the complete task alone.

User guidance
-------------
Keep every operation request for a bag in the same split and expose the
operation alongside the branch so the model must condition on it. Avoid hiding
the operation or accepting a position-dependent solution for reductions that
should be permutation invariant.

Why the distinction matters
---------------------------
The same bag has four contradictory targets. Hiding the request makes the task
unidentifiable, while reordering a true set should not change its answer.

Protocol and gate
-----------------
Train on 768 independently drawn length-eight bags, validate on 96, and test on
192 for 800 deterministic CPU steps. Re-evaluate after permuting items and
after replacing every operation with null. Report per-operation RMSE normalized
by the matching train-mean baseline. Every intact nRMSE must be at most 0.20,
every permuted nRMSE at most 0.25, permutation prediction delta at most 0.05
target SD, and at least three hidden-operation cells must have nRMSE >= 0.75
while their per-bag predictions collapse.

Status and evidence
-------------------
The fixed-length Stage A is provisional and passing: sum, mean, minimum, and
maximum each reach nRMSE <= 0.20; fresh permutations preserve predictions; and
hiding the operation makes paired requests contradictory and destroys
per-operation skill. Variable-length Stages B and C are not implemented.

Remaining work and promotion
----------------------------
Stage B must evaluate every operation at lengths 4 through 16, with every cell
at nRMSE <= 0.30. If sum alone fails, add an explicit visible-count control to
separate lost cardinality from general optimization failure. Stage C must
characterize degradation through length 32 and state the supported range.
Promotion requires three paired seeds across the declared core range and at
least ten seeds to freeze per-operation and per-length gates.

Run with::

    uv run pytest -n 0 proofs/relational/operation_conditioned_reduction -q
"""
