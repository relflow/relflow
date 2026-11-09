"""Generate static review figures from existing proof artifacts; never train."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import math
import shutil
import tempfile
import traceback
from datetime import date, datetime
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

from evidence.diagnostics.data import Bundle

ROOT = Path(__file__).resolve().parents[2]


def plain(value):
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else {"nonfinite": str(value)}
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    raise TypeError(f"Plot data must be portable JSON; received {type(value).__name__}")


def save(path: Path, value) -> None:
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def typography(output: Path, entries: dict) -> None:
    """Keep SVG labels readable when the plotting machine's font is unavailable."""
    for entry in entries.values():
        for name in entry["assets"]:
            if not name.endswith(".svg"):
                continue
            source = output / name
            text = source.read_text()
            portable = text.replace("font-family: 'DejaVu Sans';", "font-family: 'DejaVu Sans', Arial, sans-serif;")
            if text != portable:
                source.write_text(portable)
            entry["assets"][name] = hashlib.sha256(source.read_bytes()).hexdigest()


def package(output: Path, destination: Path) -> None:
    """Archive only the complete checksum-bound static review, excluding PNGs."""
    review = json.loads((output / "review.json").read_text())
    if review["errors"] or set(review["entries"]) != set(review["expected"]):
        raise ValueError("Only a complete diagnostic review can become a preview bundle")
    assets = {name: digest for entry in review["entries"].values() for name, digest in entry["assets"].items()}
    assets["review.json"] = hashlib.sha256((output / "review.json").read_bytes()).hexdigest()
    with ZipFile(destination, "w", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, expected in sorted(assets.items()):
            if len(Path(name).parts) != 1 or Path(name).is_absolute():
                raise ValueError(f"Diagnostic archive asset must be a flat relative file: {name}")
            payload = (output / name).read_bytes()
            if hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError(f"Diagnostic archive asset differs from its reviewed checksum: {name}")
            info = ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = ZIP_DEFLATED
            archive.writestr(info, payload)


def build(bundles: Path, output: Path, identifiers: list[str] | None = None, *, ci: dict | None = None) -> dict:
    renderer_digest = hashlib.sha256(Path(__file__).with_name("plots.py").read_bytes()).hexdigest()
    from evidence.diagnostics.plots import render

    specs = json.loads(Path(__file__).with_name("views.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    paths = {}
    for manifest in bundles.rglob("manifest.json"):
        value = json.loads(manifest.read_text())
        identifier = value.get("id")
        if identifier not in specs:
            continue
        if identifier in paths:
            raise ValueError(f"Two diagnostic bundles claim {identifier}; select an immutable input set")
        paths[identifier] = manifest.parent
    if identifiers:
        paths = {key: value for key, value in paths.items() if key in identifiers}
    existing = (
        json.loads((output / "review.json").read_text()) if (output / "review.json").is_file() and identifiers else {}
    )
    errors, entries = dict(existing.get("errors", {})), dict(existing.get("entries", {}))
    for identifier, directory in sorted(paths.items()):
        try:
            errors.pop(identifier, None)
            bundle = Bundle(directory)
            record = bundle.record
            if record["outcome"] == "error":
                result = {
                    "id": identifier,
                    "focus": specs[identifier]["focus"],
                    "comparison": "This execution failed before complete evaluation. The exact error is retained in its run record.",
                    "limit": specs[identifier]["limit"],
                    "figures": [],
                    "interactive": {},
                }
            else:
                result = render(bundle, specs[identifier], output)
            result.update(
                record=record,
                record_sha256=hashlib.sha256((directory / "record.json").read_bytes()).hexdigest(),
                source_sha256=bundle.manifest["source_sha256"],
                code_sha256=bundle.manifest["code_sha256"],
                manifest=f"{identifier}-manifest.json.gz",
                capture_run=bundle.manifest["runtime"]["github_run_id"],
                scope="paired Evidence CI candidate" if ci else "full-budget unpaired diagnostic execution",
            )
            for source, name in [
                (directory / "record.json", f"{identifier}-record.json"),
                (directory / "proof.py", f"{identifier}-proof.py"),
            ]:
                shutil.copy2(source, output / name)
            (output / result["manifest"]).write_bytes(
                gzip.compress((directory / "manifest.json").read_bytes(), mtime=0)
            )
            save(output / f"{identifier}.json", result)
            entries[identifier] = {
                "data": f"{identifier}.json",
                "figures": result["figures"],
                "source_sha256": result["source_sha256"],
                "record_sha256": result["record_sha256"],
                "code_sha256": result["code_sha256"],
            }
            files = [
                f"{identifier}.json",
                f"{identifier}-record.json",
                result["manifest"],
                f"{identifier}-proof.py",
            ]
            files.extend(f"{figure['name']}-{theme}.svg" for figure in result["figures"] for theme in ["light", "dark"])
            entries[identifier]["assets"] = {
                name: hashlib.sha256((output / name).read_bytes()).hexdigest() for name in files
            }
            print(
                json.dumps({"id": identifier, "figures": len(result["figures"]), "outcome": record["outcome"]}),
                flush=True,
            )
        except Exception:
            entries.pop(identifier, None)
            errors[identifier] = traceback.format_exc()
            print(json.dumps({"id": identifier, "error": errors[identifier].splitlines()[-1]}), flush=True)
    if not identifiers:
        import matplotlib.pyplot as plt

        from evidence.diagnostics.illustrations import illustrate
        from evidence.experiments.source import fingerprint, registry

        catalog = registry(ROOT / "evidence", measurements=False, lane="all")
        for identifier, spec in specs.items():
            if spec["mode"] != "research" or identifier in paths:
                continue
            source = ROOT / "evidence" / catalog[identifier]["script"]
            note = ""
            for theme, bg, ink in [("light", "#f8faf7", "#243b35"), ("dark", "#181e1c", "#e6eee8")]:
                with plt.rc_context(
                    {
                        "figure.facecolor": bg,
                        "axes.facecolor": bg,
                        "savefig.facecolor": bg,
                        "text.color": ink,
                        "axes.labelcolor": ink,
                        "xtick.color": ink,
                        "ytick.color": ink,
                        "axes.edgecolor": "#85958c",
                        "grid.color": "#85958c",
                    }
                ):
                    fig, note = illustrate(identifier)
                    fig.savefig(output / f"{identifier}-{theme}.svg", bbox_inches="tight", metadata={"Date": None})
                    if theme == "light":
                        fig.savefig(output / f"{identifier}.png", bbox_inches="tight", dpi=130)
                    plt.close(fig)
            result = {
                "id": identifier,
                "focus": spec["focus"],
                "comparison": spec["comparison"],
                "limit": spec["limit"],
                "record": None,
                "interactive": {},
                "requires": catalog[identifier]["header"].get("evidence-requires", []),
                "figures": [{"name": identifier, "caption": note}],
                "scope": "Generator illustration; no recorded learning execution",
            }
            save(output / f"{identifier}.json", result)
            entries[identifier] = {
                "data": f"{identifier}.json",
                "figures": result["figures"],
                "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "code_sha256": fingerprint(source.read_text()),
                "record_sha256": None,
                "assets": {
                    name: hashlib.sha256((output / name).read_bytes()).hexdigest()
                    for name in [f"{identifier}.json", f"{identifier}-light.svg", f"{identifier}-dark.svg"]
                },
            }
    typography(output, entries)
    save(output / "build-errors.json", errors)
    save(
        output / "review.json",
        {
            "schema": 1,
            "entries": entries,
            "expected": sorted(specs),
            "errors": errors,
            "ci": ci,
            "revision": next(
                (
                    json.loads((path / "record.json").read_text())["provenance"]["evidence_revision"]
                    for path in paths.values()
                ),
                existing.get("revision", "unrecorded"),
            ),
            "renderer_source_sha256": renderer_digest,
        },
    )
    return {
        "rendered": len(entries),
        "failed": len(errors),
        "missing": sorted(set(specs) - set(entries)),
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports", type=Path, required=True)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/assets/evidence-plots")
    parser.add_argument(
        "--archive", type=Path, help="Write a complete static bundle for an external preview attachment"
    )
    options = parser.parse_args()
    from evidence.diagnostics.paired import prepare

    with tempfile.TemporaryDirectory() as temporary:
        bundles = Path(temporary)
        ci = prepare(options.reports, options.observations, bundles)
        result = build(bundles, options.output, ci=ci)
    print(json.dumps({key: value for key, value in result.items() if key != "errors"}))
    if options.archive and not result["failed"]:
        package(options.output, options.archive)
    raise SystemExit(1 if result["failed"] or result["missing"] else 0)


if __name__ == "__main__":
    main()
