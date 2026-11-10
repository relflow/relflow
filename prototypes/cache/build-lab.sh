#!/usr/bin/env bash
set -euo pipefail
: "${NATIVE_PYTHON:?Set NATIVE_PYTHON to the benchmark Python executable}"
cargo build --locked --release --bin cache-owner
maturin build --locked --release --interpreter "$NATIVE_PYTHON" --out wheels
