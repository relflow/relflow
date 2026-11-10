# Artifact cache protocol v1

The owner is a local, capability-authenticated process. Other clients can use
this protocol without Python, relflow, or a modality-specific dependency.
`src/wire.rs` and `src/store.rs` implement the contract.

A connection first sends the owner's 32-byte capability. The owner receives it
through stdin at launch, not an argument or log. Every connection has its own
compute-claim identity. Closing it releases unfinished claims. The owner listens
on loopback and exits when its parent's stdin closes or operation 6 succeeds.

All integers are little-endian. Requests contain:

```
u32 frame_bytes
u8 operation
u32 metadata_bytes
JSON metadata
operation-specific bytes
```

`frame_bytes` excludes its own four-byte header. It must be 5 through 32 MiB;
metadata is limited to 64 KiB. Replies contain `u32 frame_bytes`, a one-byte
status (0 success, 1 error), and the response bytes or UTF-8 error. The reply
body excluding status is limited to 32 MiB. A dispatch error releases the
connection's claims. A transport failure requires reconnecting/recreating the
client; replaying an unacknowledged operation is safe only with its immutable
identities and the claim rules below.

IDs are positive signed 64-bit integers. Alias and artifact IDs are separate
domains. A returned artifact ID of zero denotes an unprepared alias. A namespace
is a nonempty UTF-8 string of at most 1,024 bytes containing no NUL.

A `packed(n)` payload contains `u32 offsets[n+1]` followed by contiguous bytes.
Offsets start at zero, are monotonic, and end at the byte length. Equal offsets
represent empty artifacts. There is no implicit element dtype or shape.

| Op | Metadata | Request data | Success data |
| --- | --- | --- | --- |
| 1: lookup | `n` | SHA256 keys, 32 bytes each | `i64 aliases[n]`, then `i64 artifacts[n]` (zero if unprepared) |
| 2: bind | `n`, `namespace` | `i64 aliases[n]`, keys (32 bytes each), `packed(n)` prepared inputs | `i64 artifacts[n]` |
| 3: fetch | `n`, `namespace` | `i64 artifacts[n]` | `u8 statuses[n]`, `packed(hit_count)` results, `packed(owned_count)` prepared inputs |
| 4: publish | `n`, `namespace` | `i64 artifacts[n]`, `packed(n)` results | `packed(n)` committed results |
| 5: stats | `{}` | empty | JSON counters, counts, budget, hot weight, identity |
| 6: stop | `{}` | empty | empty, followed by owner shutdown |
| 7: release | `{}` | empty | empty; release all claims on this connection |
| 8: wait | `{}` | empty | empty after a completion notification or a 100 ms timeout |

`n` is at most 4,096. Split batches further when prepared inputs or results
would exceed the frame budget. The store reports oversized fetch replies as
errors so the caller can retry smaller batches. Streaming a single oversized
artifact is outside this protocol.

For operation 1, the Arrow client computes a key as
`SHA256(namespace UTF-8 || NUL || raw identity bytes)`. A raw identity must
identify immutable source content and preparation options. String identities
use their UTF-8 bytes. A caller supplying keys directly owns the same identity
contract. No source or modality interpretation occurs in the owner.

Binding verifies each alias/key pair. Its canonical artifact key is
`SHA256(preparation namespace UTF-8 || NUL || prepared payload bytes)`.
Identical payloads in the same preparation namespace deduplicate across aliases,
fields, and modalities. An alias cannot later bind a different preparation
namespace or payload. A conflict rolls back the entire batch; hot aliases are
published only after the transaction commits.

Fetch statuses are 0 for an owned miss, 1 for a hit, and 2 for a claim owned by
another connection. Hits and owned inputs each retain their relative order in
the requested batch. Results are keyed by artifact ID and transform namespace.
Publish requires matching claims and returns durable, immutable committed bytes.
Errors or disconnects make owned misses available again. A claim owner must
release its claims if its computation fails while the connection stays open.

Claims coordinate different connections. Handles in one worker normally share
a connection; serialize their fetch/compute/publish cycles with
`cache.resolve_lock`. Deduplicate IDs within a batch to avoid repeated work.
Text already performs both operations in its existing resolution loop.

The managed RAM tier uses one byte-weighted Moka budget for ready aliases and
results. Prepared inputs stay on disk until needed for a computation miss.
Eviction never deletes disk data. Separate owners may open the same durable
SQLite store, but compute claims and RAM budgets are local to each owner.
