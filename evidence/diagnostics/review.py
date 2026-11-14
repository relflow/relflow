"""Static review selection and figures bound to the exact captured proof record."""

from __future__ import annotations

import hashlib
import html
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
        scope += "These figures and the selected measurements come from the same captured execution. "
        scope += {
            "paired Evidence CI candidate": "This is the candidate role of the paired Evidence CI run.",
            "candidate-only Evidence CI proof": ("This new proof has no directly comparable baseline measurement."),
        }.get(data.get("scope"), "This is an unpaired visualization run.")
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
        downloads = (
            f'<p class="diagnostic-downloads"><a href="{base}{identifier}-record.json">Full run and checks</a> · '
            f'<a href="{base}{html.escape(data["manifest"], quote=True)}">Observation manifest (gzip)</a> · '
            f'<a href="{base}{identifier}-proof.py">Executed proof source</a> · '
            f'<a href="{base}{entry["data"]}">Figure and example data</a>'
        )
        if (
            data.get("scope") in {"paired Evidence CI candidate", "candidate-only Evidence CI proof"}
            and run_id is not None
        ):
            downloads += f' · <a href="https://github.com/relflow/relflow/actions/runs/{html.escape(str(run_id), quote=True)}">CI observation bundles</a>'
        blocks.append(downloads + "</p>")
    blocks.append(f'</section>\n<script src="{base}controls.js" defer></script>')
    return "\n".join(blocks) + "\n"
