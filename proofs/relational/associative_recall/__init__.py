"""Distinguish same-branch identity recall from positional copying.

Claim
-----
An unseen Hash key can route a visible random Number to its masked target
coordinate within one repeated Branch. Values are regenerated per row and
positions are shuffled, so neither persistent identity memorization nor a
population statistic can recover them. RelFlow now performs the needed learned
content-addressed decoding from the natural record schema; users do not name
internal query, key, or value tensors.

User guidance
-------------
Keep ``entity_id``, role/query evidence, and ``value`` on the same repeated
coordinate. Use ``reduction=None`` while testing recall so every candidate
record survives to the decoder. RelFlow automatically mixes direct sibling
fields at each coordinate and uses visible fields aligned with a masked target
as a data-conditioned query over the retained memory.

Independently permute records, use disjoint Hash namespaces, and rotate target
keys at evaluation to prove that prediction follows identity rather than
position. Do not present aligned coordinates as associative recall: the
aligned positional control remains accurate even when its identities are
wrong. An explicit ``Reference`` may express a stronger application contract,
but it is not required for this learned interaction.

Omitting an exact keyed lookup is specific to this learned-recall benchmark.
When the mapping is known and exact retrieval is the application contract,
perform it in an :class:`relflow.Preprocessor` or the application.

Why the distinction matters
---------------------------
Aligned source and target positions permit copying without reading identity.
Independent shuffling plus a broken-key intervention distinguishes true
content-addressed recall from that easier shortcut.

Protocol and gates
------------------
Train a pass-through two-pair Branch on 2,048 rows, validate on 512, and test
on 1,024 for 800 deterministic CPU steps. Compare shuffled-key recall with
rotated target identities and separately train the compressed aligned control.
The control must have nRMSE <= 0.25 with valid or broken keys. Hash recall must
have nRMSE <= 0.25, broken-key nRMSE >= 0.80, and a gap >= 0.50.

Status and evidence
-------------------
The aligned positional control and shuffled same-branch Hash recall both pass
their normal deterministic gates. The recall prediction is accurate with valid
unseen keys and loses skill when only the target-key association is broken;
the aligned control deliberately does not. This separates content-addressed
lookup from copying a fixed coordinate.

The implementation first mixes fields belonging to the same coordinate. For a
repeated masked target, visible coordinate-aligned siblings seed one decoder
query per target coordinate. Decoder attention retains position for the aligned
control but can address shuffled pass-through memory by learned content. Source
eligibility and key/value alignment follow the schema geometry and masks; the
broken-key intervention proves when identity is causal, so users need neither
a hand-built join nor a public Q/K/V configuration.

Remaining work and promotion
----------------------------
Repeat the three gates across at least three paired core seeds and ten
lightweight calibration seeds. Add complete-record permutation/equivariance and
simultaneous unseen-Hash renaming controls, then sweep 2, 4, 8, and 16 pairs.
Exercise padded and missing keys, duplicate-key semantics, absent sources, and
Hash collisions. Keep the broken-key aligned control so positional copying
cannot be relabeled as recall, and test explicit ``Reference`` behavior as a
separate user contract.

Run with::

    uv run pytest -n 0 proofs/relational/associative_recall -q
"""
