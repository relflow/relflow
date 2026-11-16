"""The source licensing gate runs before package, native, and docs artifacts exist."""

import json
import shutil
from pathlib import Path

import pytest
import yaml

from scripts import release

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def checkout(tmp_path):
    for name in (
        "pyproject.toml",
        "LICENSE",
        "NOTICE",
        "CITATION.bib",
        "native/pyproject.toml",
        "native/Cargo.toml",
        "native/LICENSE",
        "native/NOTICE",
        "docs/pyproject.toml",
        "docs/starlight/package.json",
        "docs/starlight/package-lock.json",
    ):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, path)
    return tmp_path


def test_checked_out_sources_are_ready_for_agpl_builds(checkout):
    release.licensing(checkout)
    for name in ("LICENSE", "native/LICENSE"):
        path = checkout / name
        path.write_bytes(path.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    release.licensing(checkout)


@pytest.mark.parametrize(
    "name",
    [
        "pyproject.toml",
        "native/pyproject.toml",
        "native/Cargo.toml",
        "docs/pyproject.toml",
        "docs/starlight/package.json",
        "docs/starlight/package-lock.json",
        "CITATION.bib",
        "NOTICE",
        "native/NOTICE",
    ],
)
def test_source_gate_rejects_a_stale_permissive_declaration(checkout, name):
    path = checkout / name
    path.write_text(path.read_text().replace("AGPL-3.0-only", "MIT"))
    with pytest.raises(ValueError, match="AGPL"):
        release.licensing(checkout)


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE", "native/LICENSE", "native/NOTICE"])
def test_source_gate_requires_real_license_and_notice_files(checkout, name):
    (checkout / name).unlink()
    with pytest.raises(ValueError, match="regular file"):
        release.licensing(checkout)


def test_source_gate_rejects_a_custom_addition_to_the_agpl_text(checkout):
    path = checkout / "LICENSE"
    path.write_bytes(path.read_bytes() + b"\nOnly the founders may make private modifications.\n")
    with pytest.raises(ValueError, match="unchanged GNU AGPLv3"):
        release.licensing(checkout)


def test_docs_lockfile_license_is_its_root_package_license(checkout):
    path = checkout / "docs/starlight/package-lock.json"
    lock = json.loads(path.read_text())
    lock["license"] = "AGPL-3.0-only"
    lock["packages"][""]["license"] = "MIT"
    path.write_text(json.dumps(lock))
    with pytest.raises(ValueError, match="documentation package license"):
        release.licensing(checkout)


def test_ci_checks_source_licensing_before_building_distributions():
    workflows = ROOT / ".github/workflows"
    ci = yaml.safe_load((workflows / "ci.yml").read_text())["jobs"]
    assert any("scripts/release.py licensing" in step.get("run", "") for step in ci["structure"]["steps"])
    assert ci["package"]["needs"] == ci["docs"]["needs"] == "structure"
    package = yaml.safe_load((workflows / "package.yml").read_text())["jobs"]["build"]["steps"]
    guard = next(i for i, step in enumerate(package) if "scripts/release.py licensing" in step.get("run", ""))
    build = next(i for i, step in enumerate(package) if "uv build" in step.get("run", ""))
    upload = next(i for i, step in enumerate(package) if step.get("uses", "").startswith("actions/upload-artifact@"))
    assert guard < build < upload
    native = yaml.safe_load((workflows / "native.yml").read_text())["jobs"]["wheels"]["steps"]
    guard = next(i for i, step in enumerate(native) if "scripts/release.py licensing" in step.get("run", ""))
    build = next(i for i, step in enumerate(native) if step.get("uses", "").startswith("PyO3/maturin-action@"))
    assert guard < build
