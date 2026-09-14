# Modeling proofs

This directory contains deterministic synthetic-data checks of learned model
behavior. They are larger and slower than the unit and integration tests under
`tests/`, and default `pytest` runs do not collect them.

[`SPEC.md`](SPEC.md) defines the inference boundary, proof protocol, interaction
grammar, negative controls, and broader hypothesis matrix.
[The Branch attention and reduction specification](../ATTENTION_AND_BRANCH_REDUCTION.md)
explains the architecture exercised by these proofs in plain language.

Run the complete suite from the repository root:

```bash
make proofs
```

Run one proof family or one version while developing it:

```bash
uv run pytest -n 0 proofs/relational/operation_conditioned_reduction -q
uv run pytest -n 0 proofs/relational/operation_conditioned_reduction/test_fixed_length.py -q
```

## Expected user-facing change

The reduction work is meant to make one schema choice explicit without asking
users to become attention experts. Output count belongs to a parameterized
reducer, not to `Branch`:

```python
items=rf.Branch(
    length=16,
    reduction=rf.Attention(n_outputs=4),
    value=rf.Number,
)
```

The public choices are:

| Configuration | User-visible meaning |
| --- | --- |
| `rf.Attention(n_outputs=Q)` | Produce `Q` anonymous learned-query summaries, augmented internally with masked input mass and count. This preserves additive evidence without making the result an exact raw-value sum, one output per item, or a join. |
| `rf.Mean()` | Produce one presence-masked arithmetic mean of the encoded tokens. This is not the arithmetic mean of raw Number values. |
| `None` | Do not pool: route one slot per encoded child field-coordinate slot, in deterministic order, together with its presence mask. Same-coordinate fields may still interact first when branch attention is enabled. |

`Model(..., reduction=...)` configures the generated root in the same way.
`n_outputs` is invalid directly on `Branch` or `Model`, and `Mean` has no
output-count parameter. The default remains `rf.Attention()` with one output.
`Branch.length`, `overflow`, field declarations, and `mask=True` keep their
existing meaning. Branch `attention` still controls interaction within encoded
memory; `reduction` controls what representation is routed upward.
Reducer-specific settings live with that reducer:
`Attention(n_heads=..., n_layers=..., dropout=..., position=...)`.
Code that previously placed `n_outputs` or reduction-network `n_linear` on a
branch should migrate them to `rf.Attention(n_outputs=..., n_layers=...)`.

These reducers operate on encoded token slots, not directly on domain values.
With `reduction=None`, branch self-attention may already have contextualized a
slot before it is routed, and sibling fields contribute separate slots rather
than one automatically fused item token. With `Mean`, exact duplicate and
cardinality invariance requires the pre-reduction path not to encode count or
order—for example, the isolated proof uses `attention=None`.

Branch self-attention retains rotary position information by default. Set
`Attention(position=False)` to disable position in the learned-query reducer;
also use `Branch(attention=None)` when testing strict unordered invariance,
because the reducer setting does not change the branch's earlier attention.
Leaf `decoder_position=None` automatically disables decoder position for a
scalar target and enables it for a repeated target; `True` and `False`
override that default. Every set claim still needs a metamorphic permutation
gate.

For example, the new weighted-sum proof uses the ordinary schema; it does not
declare a multiplication operator or derived answer context:

```python
model = rf.Model(
    d_model=32,
    n_layers=2,
    n_heads=4,
    items=rf.Branch(
        length=6,
        reduction=rf.Attention(n_layers=2),
        value=rf.Number,
        weight=rf.Number,
    ),
    weighted_sum=rf.Number(mask=True),
)
```

Pass-through tokens are internal representations, not final decoder logits and
not raw values. RelFlow now mixes visible fields at the same coordinate before
branch-wide reduction, conditions each decoder query on visible fields at its
target coordinate, and gives decoders the resulting branch context. Users do
not wire attention Q/K/V, copy an answer into context, or manually precompute a
weighted contribution merely to make ordinary schema-local reasoning
learnable. This does not remove RelFlow's structural `query=` syntax:
queries and preprocessors still locate, rename, filter, join, sort, window, or
derive source data when observations do not already match the schema.

`mask=True` remains exactly
`Mask(skip=True, dropout=False, reconstruct=True)`: the example's target content
is retained for supervision but does not produce an encoder parcel. Decoders
already own learned target-shaped query slots. Any future encoder-visible or
data-conditioned target slot must be built without feeding that example's
hidden answer into the forward context; model parameters and datatype resources
are, of course, still learned from training targets.

The aggregation proofs measure learned approximations on declared bounded data
distributions. They are not replacements for exact accounting. If a product,
filter, join, or sum is known business logic and must be exact, calculating it
in a preprocessor or application remains the correct production design. A
supplied statistic is discouraged only when it would invalidate the learned
interaction that a particular proof is trying to measure.

The single-seed proofs currently suggest these provisional recipes:

| Need | Start with | Check before relying on it |
| --- | --- | --- |
| Normalized one-token latent | `rf.Mean()` | The target does not need count, total mass, or distribution detail; disable pre-pool attention if exact duplication invariance matters. |
| Fixed-width learned statistic or bounded sum | `rf.Attention()` | Its mass lane retains count and additive evidence, but this remains a learned approximation over encoded tokens; test unseen lengths and duplication. |
| Coordinate-sensitive retrieval | Start with `reduction=None`; then test `Attention(n_outputs=k)` | Pass-through is easiest to diagnose, while enough learned outputs can retain bounded candidate evidence. Target-coordinate queries retrieve by visible Category or unseen Hash identity. |
| Peer-relative item output | `reduction=None` or a tested learned item summary; keep the root route open | Aligned coordinate queries preserve each item's own evidence while branch outputs provide shared peer context. |
| Grouped weighted sum | Raw sibling `group`, `value`, and `weight` fields | Rotate labels, swap weights within groups, and permute complete items. |

These are guides to experiments, not universal guarantees; every proof
directory states its training scope, including length where relevant, and its
promotion work.

In short: users choose whether and how to compress; RelFlow remains responsible
for learning relationships among the fields and coordinates the schema already
declares. General all-pairs comparison across sibling collections remains a
known boundary. The retained [collection-overlap proof](relational/collection_overlap/)
records it, and [`RELATIONS.md`](../RELATIONS.md) specifies a deferred opt-in
feature without presenting it as current API.

## Proof organization

Each behavioral claim owns a semantically named directory under a broad
category. There are no sequence-number prefixes and no Markdown work sidecars.
Python module docstrings are the user guide and living work log:

```text
proofs/<category>/<proof_name>/
├── support.py                 # shared process plus proof-wide status, when needed
├── test_recommended_route.py  # exactly one collected scenario
└── test_footgun_route.py      # exactly one collected scenario
```

Every test file contains exactly one top-level test and explains that version
in its module docstring. Each docstring includes a small fenced YAML tree with
literal `input` and `expected_output` keys so the learned interaction is
visible at a glance. A single-file proof can keep the complete guide there;
a multi-version proof keeps common evidence, remaining work, and promotion
criteria in `support.py` or the package docstring. A unit-level layout contract
prevents numbered paths, `WORK.md`, undocumented versions, multi-test files,
and colliding test-module basenames from returning.

These are empirical regression proofs, not formal mathematical proofs. Each
one should generate data locally from fixed seeds, use public RelFlow training
and inference APIs, evaluate held-out behavior in natural units, include a
matched control, and print enough diagnostics to distinguish failed learning
from a violated proof precondition or a deliberately lossy configuration.

## Implemented coverage

| Proof family | Versions | Current result |
| --- | --- | --- |
| [Signal detection](calibration/signal_detection/) | signal versus independent noise | Passing provisional harness calibration |
| [Missingness](state/missingness/) | null versus real zero; null-prefill control | Passing provisional value-state proof |
| [Item alignment](relational/item_alignment/) | repeated subtotal reconstruction; shuffled targets | Passing through variable length five |
| [Hash equality](identity/hash_equality/) | unseen Hash equality; shuffled and Category-OOV controls | Passing provisional flat-identity proof |
| [Cluster convergence](cluster/cluster_convergence/) | reconstructing labels, hidden regimes, dormant plain input | Three mechanistic gates pass; partition and held-out gates remain |
| [Operation-conditioned reduction](relational/operation_conditioned_reduction/) | fixed-length sum, mean, min, and max | Passing at length eight; variable-length stages remain |
| [Cardinality generalization](aggregation/cardinality_generalization/) | Mean invariance; Attention and visible-count sum routes | All five gates pass, including unseen-length Attention mass and visible-count extrapolation |
| [Category-conditioned reduction](relational/category_conditioned_reduction/) | filtered mean; group-by-operation composition | Both filtering and all twelve group×operation cells pass |
| [Weighted aggregation](aggregation/weighted_aggregation/) | supplied contribution, raw weighted sum, variable weighted mean, Mean/count footgun | Four provisional gates pass, including pairing and metamorphic controls |
| [Grouped weighted aggregation](aggregation/grouped_weighted_aggregation/) | natural raw schema; supplied-contribution diagnostic | Both routes pass accuracy, label corruption, pairing, and complete-item permutation controls |
| [Distribution statistics](aggregation/distribution_statistics/) | raw and supplied variance; supplied and raw covariance | All four paths pass; raw covariance responds strongly to pairing corruption |
| [Order statistics](aggregation/order_statistics/) | minimum, quartiles, median, and maximum from one visible request | All rank, shape, permutation, hidden-request, and corrupted-request gates pass |
| [Sibling entity transfer](relational/sibling_entity_transfer/) | persistent Category and unseen Hash | Both keyed routes pass identity-corruption controls; Hash still requires pass-through for fresh IDs |
| [Argmax retrieval](relational/argmax_retrieval/) | Number payload at maximum score | Full pass-through and three-output Attention both pass at length three |
| [Associative recall](relational/associative_recall/) | aligned-position control; shuffled Hash lookup | Both position control and true unseen-key recall pass their corruption gates |
| [Peer-relative inference](relational/peer_relative_inference/) | supplied mean, ungrouped and grouped raw deviation, one-query summary | All four versions pass; both pass-through and compressed shared context support group selection and broadcast-back |
| [Collection overlap](relational/collection_overlap/) | flat unseen-Hash equality control; direct sibling collections | Equality passes; direct collection overlap deliberately records the current chance boundary |
| [Hierarchical statistics](structure/hierarchical_statistics/) | Mean/cardinality, fixed-width sum, and nested Attention | Three provisional reduction diagnostics pass |
| [DatePart periodicity](temporal/datepart_periodicity/) | month inference, weekday and leap ambiguity, two-coordinate business window | Four provisional gates pass, including constructive identifiability controls |

The current suite has 45 independently runnable, ordinary expected-pass
scenarios. Some are negative and footgun characterizations whose passing
outcome is to demonstrate a limitation; the module count is not a count of
universal affirmative capabilities.

Several outputs and full pass-through change how much evidence reaches a
parent, but output count still does not assign identities, operations, or
groups to learned slots. Automatic coordinate context, data-conditioned
queries, and additive mass provide those inductive paths. Recommended controls
and documented footguns remain beside each positive route in the same proof
directory.

Current gates are generally single-seed and provisional. The multi-seed
promotion protocol in [`SPEC.md`](SPEC.md) remains the bar before a threshold is
treated as a stable release claim.
