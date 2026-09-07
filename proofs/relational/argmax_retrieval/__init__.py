"""Retrieve the payload attached to the maximum-scoring branch item.

Claim
-----
At length three, preserving the complete item-to-decoder route lets a model
select by one Number and return a different Number from the same item. The
answer is a fresh random payload identified only by the maximum-score
coordinate, so rotating payloads breaks the required binding.

User guidance
-------------
Keep score and payload together in one repeated Branch. Either preserve every
token with ``reduction=None`` or use
``reduction=Attention(n_outputs=candidates)`` when a learned fixed-width
summary is appropriate. The latter does not promise that output slot *i* is
candidate *i*; the outputs jointly retain evidence that the decoder can use.
Do not treat success at length two as evidence for longer retrieval or use a
target predictable from bag-level payload statistics.

Leaving the winning payload masked is specific to this learned-retrieval
benchmark. When scores and the argmax rule are known and the selected payload
must be exact, compute it in a ``Preprocessor`` or the application rather than
using model inference as the selection algorithm.

Why the distinction matters
---------------------------
A bag-level statistic can correlate with an answer without preserving which
payload belongs to the winning score. The rotated-payload intervention keeps
both marginals fixed and isolates that coordinate binding.

Protocol and gate
-----------------
Train matched multi-output and full pass-through models on 4,096 three-item
collections, validate on 1,024, and test on 2,048 for 25 deterministic CPU
epochs. Evaluate intact and rotated-payload versions of identical held-out
scores. Each intact route must have nRMSE <= 0.35; each broken-pair nRMSE must
be >= 0.90 and at least 0.35 worse than its intact counterpart.

Status and evidence
-------------------
Provisional first rung. Number payload retrieval passes for argmax with three
candidates through both full pass-through and learned multi-output reduction.
On the current deterministic seed they reach approximately 0.16 and 0.30
nRMSE, respectively. Rotating payloads relative to scores must remove the
skill in both routes. Broader operations, payload types, and lengths are
unimplemented.

Remaining work and promotion
----------------------------
Add argmin, optional selected-group filtering, and bounded Category payloads.
Sweep candidate count and state the supported capacity boundary. Retain
broken-pair and compressed-route controls at every promoted stage. Promotion
requires the gates across three paired core seeds and calibration over at least
ten seeds.

Run with::

    uv run pytest -n 0 proofs/relational/argmax_retrieval -q
"""
