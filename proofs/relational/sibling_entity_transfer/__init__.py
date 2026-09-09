"""Transfer random entity values between sibling branches.

Claim
-----
RelFlow can match each target entity to a source entity in a sibling Branch and
infer its masked Number by persistent Category or unseen Hash identity. Values
are resampled per row and source and target positions are independently
shuffled, leaving identity as the only valid communication path. The natural
two-branch schema is sufficient; users do not declare a join or attention
Q/K/V paths.

User guidance
-------------
Keep each ``entity_id`` and its ``value`` on the same branch coordinate. Use
``reduction=None`` along the sibling-to-root route when every unseen identity
and value must remain available. RelFlow automatically mixes aligned fields
within each record, exposes pass-through sibling memory to the target decoder,
and derives one query per masked target coordinate from its visible identity.
The resulting attention is learned and content-addressed rather than tied to
source or target position.

Persistent Categories in this bounded two-entity control also survive the
tested fixed-width compression, but that does not guarantee arbitrary
cardinality or unseen-Hash recall after compression. Pass-through is the simple
safe schema when individual associations matter. Independently permute sibling
orders and break target identities during evaluation; successful transport
without broken-key degradation is not evidence of a keyed transfer.

Omitting a deterministic keyed join is specific to this learned-transfer
benchmark. If the identity mapping and copied value must be exact in
production, perform the join in an :class:`relflow.Preprocessor` or the
application.

Why the distinction matters
---------------------------
Random values and independently shuffled branches remove positional and
population shortcuts. Breaking only target identities then tests whether the
prediction truly crossed branches through the entity key.

Protocol and gates
------------------
For each identity type, train matched compressed and full pass-through models
on 2,048 rows, validate on 512, and test on 1,024 for 600 deterministic CPU
steps. The persistent-Category compression control must have nRMSE <= 0.25.
The Hash pass-through route must improve nRMSE by at least 0.15 over early
compression. Both identity routes require transfer nRMSE <= 0.25, broken-key
nRMSE >= 0.80, and a gap >= 0.50.

Status and evidence
-------------------
Both persistent-Category and unseen-Hash sibling-transfer proofs pass their
normal deterministic gates. In the calibrated Category run, fixed-width
compression reaches 0.0198 nRMSE and pass-through reaches 0.0046. The Hash
route also clears its accuracy, broken-key, gap, and compression-contrast
gates. These are implemented architecture paths rather than temporary grafts.

Automatic coordinate mixing binds each source identity to its Number. The
pass-through root retains source evidence for the target decoder, whose visible
target identity conditions the query at the same output coordinate. Decoder
attention retains coordinate position as an available signal, while the
broken-key and independently shuffled-position controls demonstrate that this
route retrieves source content by identity. The ordinary Number head produces
the prediction. Category and Hash share this internal mechanism even though
their identity representations differ.

Remaining work and promotion
----------------------------
Repeat both identity routes over at least three paired core seeds and ten
lightweight calibration seeds, with no core seed above 0.40 nRMSE. Sweep 2, 4,
8, and 16 entities and verify independent sibling permutations plus target
equivariance. Add padded, missing, duplicate, and absent identities; define
duplicate-match behavior; and exercise Hash collisions. Test larger persistent
Category spaces and document when a shared identity geometry is required.
Keep explicit ``Reference`` declarations as a separately tested stronger
contract rather than making common users wire retrieval internals.

Run with::

    uv run pytest -n 0 proofs/relational/sibling_entity_transfer -q
"""
