"""Static review selection and figures bound to the exact captured proof record."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from urllib import request


def prepare(root: Path, spec: dict) -> Path:
    """Fetch a checksum-bound static bundle from the retained CI or version release."""
    from evidence.visuals import LIMIT, install

    if (
        not re.fullmatch(r"[a-f0-9]{64}", spec.get("sha256", ""))
        or type(spec.get("size")) is not int
        or not 0 < spec["size"] <= LIMIT
        or not re.fullmatch(
            r"https://github.com/relflow/relflow/releases/download/evidence-(?:ci-\d+-attempt-\d+|v[0-9A-Za-z.]+)/proof-visualizations.zip",
            spec.get("url", ""),
        )
    ):
        raise ValueError("Evidence plots require a bounded, checksum-pinned repository release asset")
    cache = root / ".cache/evidence/diagnostics"
    cache.mkdir(parents=True, exist_ok=True)
    archive = cache / (spec["sha256"] + ".zip")
    if archive.is_file():
        payload = archive.read_bytes()
    else:
        query = request.Request(spec["url"], headers={"User-Agent": "relflow-docs"})
        with request.urlopen(query, timeout=60) as response:
            payload = response.read(spec["size"] + 1)
    if len(payload) != spec["size"] or hashlib.sha256(payload).hexdigest() != spec["sha256"]:
        raise ValueError("Evidence plot archive differs from its pinned checksum")
    if not archive.exists():
        archive.write_bytes(payload)
    target = root / "docs/assets/evidence-plots"
    install(archive, target)
    return target


def load(root: Path) -> dict | None:
    assets = root / "docs/assets/evidence-plots"
    path = assets / "review.json"
    if not path.is_file():
        return None
    from evidence.visuals import validate

    def asset(name: str) -> bytes:
        source = assets / name
        if source.is_symlink():
            raise ValueError(f"Review asset must be a regular file: {name}")
        return source.read_bytes()

    return validate(json.loads(path.read_text()), asset)
