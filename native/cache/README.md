# Native shared artifact cache

Optional Rust acceleration for the shared tensorfield cache in
`src/relflow/tensorfields/shared/cache/`. Ordinary `rf.Text` uses that helper;
other tensorfield extensions can reuse its opaque artifacts, stable IDs,
frequency-aware RAM tier, disk persistence, and computation claims.

The default relflow wheel remains a Python build. Without this add-on, the
helper launches its isolated Python owner, backed by Rust Moka and SQLite.
Installing this add-on selects the native owner and Arrow-aware PyO3 transport
automatically. Both implementations use the same database and wire protocol.
No additional tensorfield registration or model option is required.

The extension owns preprocessing identity, payload codecs, encoder versions,
pooling, precision, and any device representation. The shared store handles
bytes and never interprets tokens, images, audio, or video. See
[`PROTOCOL.md`](PROTOCOL.md) for the language-independent protocol.

## Build

From this directory, with Rust, Maturin, and the target Python available:

```bash
NATIVE_PYTHON=/path/to/python ./build.sh
/path/to/python -m pip install wheels/*.whl
```

The wheel bundles both the native transport and `cache-owner` executable. The
helper starts the owner lazily, binds loopback, and passes its capability through
stdin. `RELFLOW_CACHE_OWNER` can select an explicitly built executable instead.
Neither Rust nor Maturin is needed to install the ordinary relflow wheel.

## Resources and identity

The caller configures one RAM budget per local store/worker family. SQLite
persists all prepared artifacts and computed results; Moka admits frequently
observed results and aliases within its byte budget. Entries larger than that
budget remain on disk. Disk storage has no quota or pruning. The managed budget
excludes active request buffers, runtime overhead, device memory, and the OS
file cache. Each SQLite owner has a separate 4 MiB private page budget.

A request or reply is limited to 32 MiB and a batch to 4,096 entries. Consumers
must split large sequences into bounded clips, frames, windows, or chunks.
Compute claims and RAM budgets are shared within one owner; independently
started owners share disk rather than claims or a global RAM budget.

Use immutable source content/version identities and complete preprocessing and
encoder namespaces. Changing a tokenizer, model weights, crop, sample rate,
frame window, output precision, or payload codec must invalidate the relevant
artifact. Random augmentation belongs outside reusable deterministic stages.
Context-conditioned encoders also include their fitted context in identity.

## Validation

Shared cache tests live under `tests/tensorfields/shared/`; ordinary Text tests
live under `tests/tensorfields/extensions/`. With the optional wheel installed,
the same tests exercise native acceleration:

```bash
cargo test --locked
/path/to/python -m pytest -m integration \
  tests/tensorfields/shared/test_cache.py \
  tests/tensorfields/shared/test_cache_store.py \
  tests/tensorfields/extensions/test_text_cache.py \
  tests/tensorfields/extensions/test_text_tokens.py
```

The pytest command runs from the repository root. The core Rust tests cover
transactional binding, durable reopen, opaque payloads, deduplication, and
claim release. Experimental throughput scripts and measured artifacts remain
local and are excluded from Git tracking.
