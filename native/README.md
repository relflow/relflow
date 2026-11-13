# Native helpers for relflow

`relflow-native` is the optional native distribution for relflow. Python owns
the public API; focused Rust modules own substantial kernels and local runtime
state. The initial module is the shared tensorfield artifact cache. Future
helpers belong beside `src/cache/` in this crate, with their own bindings and
contracts; they do not require another Python distribution or build system.

The base relflow wheel remains a Python build. Without this add-on, the cache
launches its isolated Python owner backed by Moka and SQLite. Installing the
add-on automatically selects the Rust owner and Arrow-aware PyO3 transport.
An incompatible or broken installed add-on raises an error instead of silently
falling back. `Cache.stats()` reports `backend`, `transport`, and `protocol`.

## Build and install

From the repository root, with Rust and Python 3.12 or later available:

```bash
python -m pip install ./native
```

Standard PEP 517 builds produce the complete package. To inspect distributions:

```bash
uv build native --out-dir native-dist
```

There is one compiled extension. The cache launches it in a child process with
`python -m relflow_native.cache`; no executable-copy step or bundled executable
is required. The launcher imports neither relflow nor model libraries. Ordinary
compute helpers can run directly in the caller and should only use a process
when their state or isolation requires one.

The Cargo manifest is the package version's source of truth. `abi3-py312`
supports ordinary CPython 3.12 and later; it does not imply free-threaded Python
or cross-platform binary compatibility. CI builds Linux x86-64 manylinux 2.28
and Windows x86-64 wheels and tests them on Python 3.12 and 3.14. Other targets
can build from source but are not covered by the native wheel matrix.

## Adding a helper

Add a focused module under `src/` and register its Python bindings in
`src/lib.rs`. Keep the Python-facing API in relflow, batch operations across
the FFI boundary, and detach Python during substantial Rust computation or
blocking I/O. Keep Arrow buffers owned while detached; avoid per-value Python
callbacks. Add annotations to `python/relflow_native/native.pyi` and tests of
the installed package under `tests/` here. Stateful protocols declare a
compatibility version independently of the package version.

## Shared cache

The extension owns preparation identity, payload codecs, encoder versions and
device representation. The store handles opaque bytes and never interprets
text, images, audio or video. See [`src/cache/PROTOCOL.md`](src/cache/PROTOCOL.md).

One cache owner provides a RAM budget and compute claims for a local worker
family. SQLite persists prepared artifacts and computed results; Moka admits
frequently observed results and aliases within its byte budget. Oversized
entries remain on disk. Disk has no quota or pruning. Request buffers, runtime
overhead, device memory and OS file cache are outside the managed budget;
SQLite has a separate 4 MiB private page budget per owner.

Frames are limited to 32 MiB and batches to 4,096 entries. Consumers split large
inputs into clips, frames, windows or chunks. Separate owners share disk rather
than claims or a global RAM budget. Complete deterministic transformation
identities belong in namespaces; random augmentation stays outside the cache.

Both owner implementations use the same versioned wire protocol and database.
The local capability travels through stdin, and owners stop when the parent
closes stdin. `RELFLOW_CACHE_OWNER` may select an external executable implementing
the current protocol. An absent add-on uses the portable implementation;
protocol mismatches always fail with an actionable error.

## Validation

```bash
cargo fmt --manifest-path native/Cargo.toml --check
cargo test --manifest-path native/Cargo.toml --locked
cargo clippy --manifest-path native/Cargo.toml --locked --features python --all-targets -- -D warnings
python -m pytest --confcutdir=native/tests native/tests -m integration
```

The installed-package tests require both relflow and relflow-native distributions.
They load the installed cache code without importing the model stack, require
the Rust owner, and exercise both transports against both owner implementations.
The required CI gate also tests wheel installation and source-distribution
rebuilds, the minimum Rust version, and current stable Rust. Ordinary relflow
cache and Text tests remain under the repository's `tests/tensorfields/`.
