#!/usr/bin/env bash
set -euo pipefail
: "${NATIVE_PYTHON:?Set NATIVE_PYTHON to the target Python executable}"
cargo build --locked --release --bin cache-owner
"$NATIVE_PYTHON" - <<'PYTHON'
import json
import shutil
import subprocess
import sys
from pathlib import Path
metadata = json.loads(subprocess.check_output(["cargo", "metadata", "--no-deps", "--format-version", "1"]))
name = "cache-owner.exe" if sys.platform == "win32" else "cache-owner"
shutil.copy2(Path(metadata["target_directory"]) / "release" / name, Path("python/relflow_cache_native") / name)
PYTHON
maturin build --locked --release --interpreter "$NATIVE_PYTHON" --out wheels
