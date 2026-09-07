"""Filter repeated Numbers by Category before reducing them.

Claim
-----
A visible root request can select one of three interleaved item groups and can
compose that selection with ``sum``, ``mean``, ``min``, or ``max``. RelFlow now
learns this interaction from the natural schema: ``group`` and ``value`` remain
sibling fields on each item, while ``selected_group`` and ``operation`` remain
ordinary visible root fields. Users do not declare attention Q/K/V paths or
precompute one feature per requested answer.

User guidance
-------------
Keep fields that describe one record on the same Branch coordinate. Branch
encoding automatically mixes direct sibling fields at that coordinate before
structural reduction, so the group label and Number can become one usable item
representation. Use ``reduction=None`` when every interleaved item must remain
available to the root. The decoder then incorporates visible sibling/root
fields into its learned query, allowing ``selected_group`` and ``operation``
to address the relevant retained content.

Treat ``Attention(n_outputs=...)`` as representational capacity, not as a list
of semantic result slots. Start with one fixed operation, check label
corruption, and add operations only after the simpler route is established.
Do not manually wire Q/K/V or duplicate equivalent data into one field per
group-operation pair.

Omitting a deterministic filter or aggregate is specific to this benchmark of
learned conditional selection. If the requested statistic is exact contractual
business logic, perform the filter and reduction in an
:class:`relflow.Preprocessor` or the application rather than relying on an
approximate model prediction.

Why the distinction matters
---------------------------
Correct group and value marginals do not reveal which values belong to the
requested group. Corrupting only their pairing proves that the learned result
uses record-level association before reduction.

Protocol and gates
------------------
The filtered-mean rung trains on 512 bags with two items per group for 800
deterministic CPU steps. The composed proof trains on 512 bags with four items
per group for 1,200 steps. Metrics are per-cell RMSE normalized by the matching
train-mean baseline. Each filtered-mean group must have nRMSE <= 0.55 and label
corruption must worsen it by at least 0.25. Every one of the twelve composed
group-operation cells must have nRMSE <= 0.30.

Status and evidence
-------------------
Both the filtered-mean rung and the twelve-cell group-by-operation proof pass
their deterministic normal gates. The composed proof passes all twelve cells
at the configured 1,200-step seed. Both are ordinary expected-pass regression
proofs.

The relevant behavior is automatic and structural. Direct sibling item fields
are mixed only at their shared coordinate, pass-through retains all item
evidence, and visible request fields condition the decoder query. Attention can
therefore select group-bearing content and learn the requested reduction
without assigning a permanent meaning to any anonymous reduction output. The
filtered-mean label-corruption control confirms that its successful prediction
depends on the Category/value association rather than their marginals alone.

Remaining work and promotion
----------------------------
Repeat the full twelve-cell gate across at least three paired core seeds and
ten lightweight calibration seeds before freezing thresholds. Add label,
selected-group, and operation corruptions to the composed proof, then vary
total length through 24 and test missing values, absent selected groups, and an
explicit empty-group convention. Sweep reduction width rather than assuming
that twelve requested cells require twelve outputs. Keep ``Mean`` at one output
and remember that ``reduction=None`` routes every configured encoded
field-coordinate slot with its presence mask.

Run with::

    uv run pytest -n 0 proofs/relational/category_conditioned_reduction -q
"""
