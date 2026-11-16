"""Enforce repository ownership, file types, and artifact boundaries before publication."""

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///

from __future__ import annotations

import argparse
import fnmatch
import subprocess
import tomllib
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def matches(parts: tuple[str, ...], pattern: tuple[str, ...]) -> bool:
    """Match anchored path segments, with ** spanning zero or more directories."""
    if not pattern:
        return not parts
    if pattern[0] == "**":
        return matches(parts, pattern[1:]) or bool(parts and matches(parts[1:], pattern))
    return bool(parts and fnmatch.fnmatchcase(parts[0], pattern[0]) and matches(parts[1:], pattern[1:]))


def violations(root: Path, policy: dict, paths: list[str]) -> list[str]:
    """Validate Git candidates; ignored outputs are never exempt once tracked."""
    errors = []
    for name in sorted(set(paths)):
        path = PurePosixPath(name)
        local = root / name
        if path.is_absolute() or ".." in path.parts:
            errors.append(f"{name}: paths must stay inside the repository")
            continue
        if not local.exists() and not local.is_symlink():
            continue  # Unstaged removals no longer contribute content to the worktree.
        if any(root.joinpath(*path.parts[:depth]).is_symlink() for depth in range(1, len(path.parts) + 1)):
            errors.append(f"{name}: repository inputs must be regular files, not symlinks")
            continue
        if not local.is_file():
            errors.append(f"{name}: repository inputs must be regular files")
            continue
        if any(matches(path.parts, tuple(pattern.split("/"))) for pattern in policy["blocked"]):
            errors.append(f"{name}: generated output belongs in an ignored artifact directory or CI asset")
            continue
        allowed = name in policy["files"] or any(
            path.suffix in suffixes and matches(path.parts, tuple(pattern.split("/")))
            for pattern, suffixes in policy["paths"].items()
        )
        if not allowed:
            errors.append(f"{name}: no owner permits this path and file type; use an existing source area")
            continue
        limit = policy.get("sizes", {}).get(name, policy["max-bytes"])
        size = local.stat().st_size
        if size > limit:
            errors.append(f"{name}: {size:,} bytes exceeds {limit:,}; retain large artifacts outside Git")
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tracked", action="store_true", help="Check Git-tracked files only, including ignored ones.")
    options = parser.parse_args()
    policy = tomllib.loads((ROOT / "pyproject.toml").read_text())["tool"]["relflow"]["tree"]
    arguments = ["git", "ls-files", "--cached", "-z"]
    if not options.tracked:
        arguments.extend(["--others", "--exclude-standard"])
    result = subprocess.run(arguments, cwd=ROOT, check=True, capture_output=True)
    paths = [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]
    errors = violations(ROOT, policy, paths)
    if errors:
        parser.exit(
            1,
            "Repository structure rejected:\n"
            + "\n".join(f"  {error}" for error in errors)
            + "\nSee [tool.relflow.tree] in pyproject.toml and CONTRIBUTING.md.\n",
        )
    print(f"Repository structure passed ({len(set(paths))} Git paths).")


if __name__ == "__main__":
    main()
