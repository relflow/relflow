"""Static review selection and figures bound to the exact captured proof record."""

from __future__ import annotations

import hashlib
import html
import io
import json
import re
import shutil
import tempfile
from pathlib import Path
from urllib import request
from zipfile import ZipFile

from evidence.experiments.source import fingerprint


def prepare(root: Path) -> Path | None:
    """Fetch an immutable preview bundle; generated assets never belong in Git."""
    receipt = root / "evidence/diagnostics/preview.json"
    if not receipt.is_file():
        return None
    spec = json.loads(receipt.read_text())
    if (
        spec.get("schema") != 1
        or not re.fullmatch(r"[a-f0-9]{64}", spec.get("sha256", ""))
        or not 0 < spec.get("size", 0) <= 128 * 1024**2
        or not spec.get("url", "").startswith("https://github.com/relflow/relflow/releases/download/docs-preview-")
    ):
        raise ValueError("Diagnostic preview receipt must identify a bounded, checksum-pinned repository asset")
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
        raise ValueError(
            "Diagnostic preview archive differs from its pinned checksum; discard the cached bundle and fetch again"
        )
    with ZipFile(io.BytesIO(payload)) as bundle, tempfile.TemporaryDirectory(dir=cache) as staging:
        members = bundle.infolist()
        if len(members) > 1000 or sum(item.file_size for item in members) > 128 * 1024**2:
            raise ValueError("Diagnostic preview archive exceeds its bounded static asset inventory")
        names = [item.filename for item in members]
        if len(set(names)) != len(names) or any(
            len(Path(name).parts) != 1
            or (name != "review.json" and not re.fullmatch(r"P\d{3}[-.][A-Za-z0-9_.-]+", name))
            for name in names
        ):
            raise ValueError("Diagnostic preview archive must contain unique, flat generated asset names")
        staged = Path(staging) / "docs/assets/evidence-plots"
        staged.mkdir(parents=True)
        for item in members:
            (staged / item.filename).write_bytes(bundle.read(item))
        review = load(Path(staging))
        if review is None:
            raise ValueError("Diagnostic preview archive has no review manifest")
        expected = {"review.json"} | {name for entry in review["entries"].values() for name in entry["assets"]}
        if set(names) != expected:
            raise ValueError("Diagnostic preview archive inventory differs from its review manifest")
        target = root / "docs/assets/evidence-plots"
        target.mkdir(parents=True, exist_ok=True)
        for name in sorted(expected):
            shutil.copy2(staged / name, target / name)
    if not archive.exists():
        archive.write_bytes(payload)
    return target


def load(root: Path) -> dict | None:
    assets = root / "docs/assets/evidence-plots"
    path = assets / "review.json"
    if not path.is_file():
        return None
    review = json.loads(path.read_text())
    if review.get("schema") != 1 or not isinstance(review.get("entries"), dict):
        raise ValueError("Evidence review must declare schema 1 and per-proof entries")
    if review.get("errors") or set(review["entries"]) != set(review["expected"]):
        raise ValueError("Evidence review must contain its complete declared selection without rendering errors")
    for identifier, entry in review["entries"].items():
        if entry["data"] not in entry["assets"]:
            raise ValueError(f"{identifier}: example data must be bound to the review asset checksums")
        for name, expected in entry["assets"].items():
            source = assets / name
            if len(Path(name).parts) != 1 or Path(name).is_absolute() or source.is_symlink():
                raise ValueError(f"{identifier}: review asset must be a regular relative file: {name}")
            if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{identifier}: review asset changed: {name}; regenerate its diagnostic figures")
        data = json.loads((assets / entry["data"]).read_text())
        if data["id"] != identifier:
            raise ValueError(f"{identifier}: example data belongs to a different proof")
        if data.get("record") is not None:
            record = assets / f"{identifier}-record.json"
            if (
                hashlib.sha256(record.read_bytes()).hexdigest() != entry["record_sha256"]
                or json.loads(record.read_text()) != data["record"]
            ):
                raise ValueError(f"{identifier}: figure data must retain the exact captured record")
    return review


def selection(root: Path) -> Path | None:
    """Select explicitly captured review runs without changing released evidence."""
    review = load(root)
    if review is None:
        return None
    from evidence.experiments.source import registry

    catalog = registry(root / "evidence", measurements=False, lane="all")
    results = {}
    for identifier, entry in review["entries"].items():
        data = json.loads((root / "docs/assets/evidence-plots" / entry["data"]).read_text())
        source = root / "evidence" / catalog[identifier]["script"]
        if fingerprint(source.read_text()) != entry["code_sha256"]:
            raise ValueError(
                f"{identifier}: proof computation changed since capture; record new diagnostics before reviewing these figures"
            )
        record = data.get("record")
        if record is not None:
            if record.get("mode") != "full" or record.get("steps_override") is not None:
                raise ValueError(f"{identifier}: review figures require a full-budget record")
            results[identifier] = record
    path = root / ".cache/evidence/diagnostic-review.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": 1,
                "version": "diagnostic-review",
                "revision": review["revision"],
                "report": {},
                "results": results,
            },
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    return path


def fragment(root: Path, identifier: str, run: dict | None, review: dict | None) -> str:
    if review is None or identifier not in review["entries"]:
        return ""
    entry = review["entries"][identifier]
    assets = root / "docs/assets/evidence-plots"
    data = json.loads((assets / entry["data"]).read_text())
    if data.get("record") != run:
        return ""
    base = "../../assets/evidence-plots/"
    blocks = [
        f'<section id="proof-diagnostics" class="evidence-diagnostics" data-observations="{base}{html.escape(entry["data"], quote=True)}">',
        f"<h3>{html.escape(data['focus'])}</h3>",
        f"<p>{html.escape(data['comparison'])}</p>",
    ]
    if run is None:
        scope = "Generator illustration. No full learning result has been recorded for this optional study."
    else:
        scope = f"Full-budget diagnostic run · seed {run['seed']} · {run['accelerator']} · {run['started_at'][:10]}. "
        scope += "These figures and the selected measurements come from the same captured execution. This is an unpaired visualization run."
    blocks.append('<p class="diagnostic-scope">' + html.escape(scope) + "</p>")
    if run is None and data.get("requires"):
        blocks.append("<p>To measure this study: " + html.escape("; ".join(data["requires"])) + ".</p>")
    for figure in data["figures"]:
        name = html.escape(figure["name"], quote=True)
        alt = html.escape(figure.get("alt", data["focus"]), quote=True)
        blocks.append(
            f'<figure><div class="plot-canvas" tabindex="0" role="region" aria-label="Scrollable proof visualization">'
            f'<img class="plot-light" src="{base}{name}-light.svg" alt="{alt}">'
            f'<img class="plot-dark" src="{base}{name}-dark.svg" alt="{alt}"></div>'
            f'<figcaption><span class="plot-scroll-note">Scroll sideways to see the full figure.</span>{html.escape(figure["caption"])} <a href="{base}{name}-light.svg">Download SVG</a></figcaption></figure>'
        )
    if data.get("interactive", {}).get("kind") in {"numeric", "classification", "fusion", "anomaly"}:
        blocks.append('<div class="proof-explorer"></div>')
    blocks.append('<p class="diagnostic-limit">' + html.escape(data["limit"]) + "</p>")
    if run is not None:
        run_id = data["capture_run"]
        blocks.append(
            f'<p class="diagnostic-downloads"><a href="{base}{identifier}-record.json">Full run and checks</a> · '
            f'<a href="{base}{html.escape(data["manifest"], quote=True)}">Observation manifest (gzip)</a> · '
            f'<a href="{base}{identifier}-proof.py">Executed proof source</a> · '
            f'<a href="{base}{entry["data"]}">Figure and example data</a> · '
            f'<a href="https://github.com/relflow/relflow/actions/runs/{html.escape(str(run_id), quote=True)}">CI observation bundles</a></p>'
        )
    blocks.append(f'</section>\n<script src="{base}controls.js" defer></script>')
    return "\n".join(blocks) + "\n"
