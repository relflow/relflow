"""Names shared by Evidence release writers and readers."""

import re
from collections.abc import Iterable

VERSION = re.compile(r"[0-9]+(?:\.[0-9]+)*(?:(?:a|b|rc)[0-9]+)?(?:\.post[0-9]+)?(?:\.dev[0-9]+)?")
LEGACY = {
    "publication": "publication.json",
    "reports": "evidence-reports.tar.gz",
    "inputs": "evidence-inputs.tar.gz",
    "provenance": "github-run.json",
    "plots": "proof-visualizations.zip",
    "measurements": "measurements.md",
    "checksums": "SHA256SUMS",
    "reuse": "evidence-reuse.json",
}


def filenames(version: str) -> dict[str, str]:
    if not VERSION.fullmatch(version):
        raise ValueError(f"Evidence assets require a normalized package version, got {version!r}")
    prefix = f"relflow-{version}-evidence"
    return {
        "publication": f"{prefix}.json",
        "reports": f"{prefix}-reports.tar.gz",
        "inputs": f"{prefix}-inputs.tar.gz",
        "provenance": f"{prefix}-provenance.json",
        "plots": f"{prefix}-plots.zip",
        "measurements": f"{prefix}-measurements.md",
        "checksums": f"{prefix}-checksums.txt",
        "reuse": f"{prefix}-reuse.json",
    }


def inventory(version: str, available: Iterable[str]) -> dict[str, str]:
    """Read old immutable releases; partial versioned inventories never fall back."""
    current = filenames(version)
    return current if set(current.values()).intersection(available) else LEGACY
