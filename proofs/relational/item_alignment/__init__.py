"""Item-aligned reconstruction from a repeated branch.

Claim
-----
A repeated target can use visible fields from the same item coordinate,
including when collection lengths vary. The target subtotal is the product of
its own quantity and unit price; other coordinates have identical marginals
and cannot determine that target.

User guidance
-------------
Keep related inputs and their target in the same item coordinate. Avoid
flattening items into unrelated rows or accepting only a bag-level score,
because either choice can hide a loss of coordinate alignment.

Protocol and gate
-----------------
Train on 4,096 orders, validate on 1,024, and evaluate on 2,048. Lengths cycle
from zero through five. A paired held-out control preserves every visible order
and permutes subtotals between coordinates. Aligned nRMSE must be at most 0.25,
permuted-target nRMSE must be at least 0.50 worse, and prediction shape and
inferred flags must exactly match valued and padded coordinates.

Status and evidence
-------------------
Provisional. Variable-length coordinate reconstruction and the permuted-target
control pass for one seed. The aligned subtotal reaches nRMSE <= 0.25,
permuting subtotals between otherwise unchanged items removes at least 0.50
nRMSE of skill, and inferred and padded prediction coordinates match branch
geometry.

Remaining work and promotion
----------------------------
Reconcile the exact product used here with the noisy product stated in
``SPEC.md``. Report performance by nonempty branch length so an aggregate
cannot hide a failed length. Promotion requires passing the geometry and
accuracy gates for three paired core seeds and calibrating the final threshold
with at least ten seeds.

Run with::

    uv run pytest -n 0 proofs/relational/item_alignment -q
"""
