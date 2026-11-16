"""Launch the shared Rust cache without importing relflow or model libraries."""

import sys

from .native import cache

if __name__ == "__main__":
    path, memory, capacity = sys.argv[1:]
    cache.run(path, int(memory), int(capacity))
