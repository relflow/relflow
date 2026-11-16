from __future__ import annotations

import json
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

import pytest

import docs.publish as publishing


@pytest.fixture
def project(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    (root / "docs/starlight").mkdir(parents=True)
    (root / "docs/starlight/astro.config.mjs").write_text("export default {};\n")
    subprocess.run(["git", "init", "--quiet", str(root)], check=True)
    subprocess.run(["git", "add", "."], cwd=root, check=True)
    subprocess.run(
        [
            "git",
            "-c",
            "core.hooksPath=/dev/null",
            "-c",
            "commit.gpgsign=false",
            "-c",
            "user.name=Documentation test",
            "-c",
            "user.email=docs@example.invalid",
            "commit",
            "--quiet",
            "-m",
            "Documentation source",
        ],
        cwd=root,
        check=True,
    )
    return root


@pytest.fixture
def environment(project: Path) -> dict[str, str]:
    return {
        "READTHEDOCS_CANONICAL_URL": "https://relflow.readthedocs.io/en/latest/",
        "READTHEDOCS_OUTPUT": str(project.parent / "output"),
        "READTHEDOCS_GIT_COMMIT_HASH": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=project, text=True
        ).strip(),
    }


@pytest.fixture
def renderer(project: Path, monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    """Keep Git real while simulating the static Node build boundary."""
    execute = subprocess.run
    profiles = []

    def run(command, **options):
        if command[0] == "git":
            return execute(command, **options)
        assert command in (["npm", "run", "check"], ["npm", "test"], ["npm", "run", "build"])
        assert options["cwd"] == project / "docs/starlight"
        if command[-1] != "build":
            return subprocess.CompletedProcess(command, 0)
        config = options["env"]
        profiles.append(config)
        site_url = config["RELFLOW_DOCS_URL"]
        revision = config["RELFLOW_DOCS_REVISION"]
        prefix = urlsplit(site_url).path
        site = project / "docs/starlight/dist"
        (site / "assets").mkdir(parents=True, exist_ok=True)
        (site / "assets/site.js").write_bytes(b"// static asset\n")
        (site / "assets/font.bin").write_bytes(b"\x00\xff\x80font payload")
        (site / "index.html").write_text(
            f'<link rel="canonical" href="{site_url}">'
            + f'<script src="{prefix}assets/site.js"></script>'
            + f'<a href="{site_url}guide/#details">Guide</a>'
            + f'<a href="https://github.com/relflow/relflow/blob/{revision}/docs/starlight/src/content/docs/index.mdx">View source</a>'
        )
        (site / "guide").mkdir(exist_ok=True)
        (site / "guide/index.html").write_text(
            f'<link rel="canonical" href="{site_url}guide/">' + '<h1 id="details">Guide</h1>'
        )
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(publishing.subprocess, "run", run)
    return profiles


@pytest.mark.parametrize(
    "site_url",
    [
        "https://relflow.readthedocs.io/en/latest/",
        "https://relflow.ai/en/v0.4.2",
        "https://relflow--66.org.readthedocs.build/en/66/",
    ],
)
@pytest.mark.parametrize("output", ["external", "_readthedocs"])
def test_publish_preserves_validated_payload_and_binds_urls_to_checkout(
    project: Path, environment: dict[str, str], renderer: list[dict], site_url: str, output: str
) -> None:
    environment["READTHEDOCS_CANONICAL_URL"] = site_url
    if output == "_readthedocs":
        environment["READTHEDOCS_OUTPUT"] = output
    original = environment.copy()
    configuration = (project / "docs/starlight/astro.config.mjs").read_bytes()
    destination = publishing.publish(project, environ=environment)
    source = project / "docs/starlight/dist"
    assert {path.relative_to(source): path.read_bytes() for path in source.rglob("*") if path.is_file()} == {
        path.relative_to(destination): path.read_bytes() for path in destination.rglob("*") if path.is_file()
    }
    assert f'href="{site_url.rstrip("/")}/"' in (destination / "index.html").read_text()
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=project, text=True).strip()
    assert f"/blob/{revision}/docs/starlight/src/content/docs/index.mdx" in (destination / "index.html").read_text()
    assert renderer[0]["RELFLOW_DOCS_REVISION"] == revision
    assert (project / "docs/starlight/astro.config.mjs").read_bytes() == configuration
    assert environment == original


def test_publish_uses_checkout_revision_when_service_does_not_supply_one(project, environment, renderer):
    revision = environment.pop("READTHEDOCS_GIT_COMMIT_HASH")
    publishing.publish(project, environ=environment)
    assert renderer[0]["RELFLOW_DOCS_REVISION"] == revision


@pytest.mark.parametrize("matching", [True, False])
def test_tag_build_consumes_only_its_release_asset_and_passes_it_to_every_build_step(
    project, environment, renderer, monkeypatch, matching
):
    (project / "evidence/experiments").mkdir(parents=True)
    environment.update(READTHEDOCS_VERSION_TYPE="tag", READTHEDOCS_GIT_IDENTIFIER="v0.4.3")
    selected = project.parent / "publication.json"
    selected.write_text(
        json.dumps(
            {
                "schema": 1,
                "version": "0.4.3",
                "revision": environment["READTHEDOCS_GIT_COMMIT_HASH"] if matching else "a" * 40,
                "report": {},
                "results": {},
            }
        )
    )
    calls = []

    def fetch(root, version, *, required):
        assert root == project and version == "0.4.3" and required is True
        return selected

    execute = publishing.subprocess.run

    def run(command, **options):
        if command[0] == "npm":
            calls.append(options["env"]["RELFLOW_EVIDENCE_PUBLICATION"])
        return execute(command, **options)

    monkeypatch.setattr(publishing, "fetch", fetch)
    monkeypatch.setattr(publishing.subprocess, "run", run)
    if matching:
        publishing.publish(project, environ=environment)
        assert calls == [str(selected)] * 3
    else:
        with pytest.raises(ValueError, match="tag's source revision"):
            publishing.publish(project, environ=environment)
        assert not calls and not renderer


def test_publish_accepts_a_prepared_empty_output_directory(project, environment, renderer):
    destination = Path(environment["READTHEDOCS_OUTPUT"]) / "html"
    destination.mkdir(parents=True)
    assert publishing.publish(project, environ=environment) == destination
    assert (destination / "index.html").is_file()


def test_publish_rejects_a_service_revision_that_does_not_match_checkout(project, environment, renderer):
    environment["READTHEDOCS_GIT_COMMIT_HASH"] = "0" * 40
    with pytest.raises(ValueError, match="does not match checkout"):
        publishing.publish(project, environ=environment)
    assert renderer == []
    assert not (Path(environment["READTHEDOCS_OUTPUT"]) / "html").exists()


@pytest.mark.parametrize(
    "site_url",
    [
        "",
        "/en/latest/",
        "http://relflow.ai/en/latest/",
        "https://user:password@relflow.ai/en/latest/",
        "https://relflow.ai/en/latest/?query=true",
        "https://relflow.ai/en/latest/#fragment",
        "https://relflow.ai/en/ latest/",
        "https://[broken",
        "https://relflow.ai:invalid/en/latest/",
        "https://relflow.ai:70000/en/latest/",
        "https://relflow.ai:0/en/latest/",
    ],
)
def test_publish_rejects_malformed_canonical_urls_before_building(project, environment, renderer, site_url):
    environment["READTHEDOCS_CANONICAL_URL"] = site_url
    with pytest.raises(ValueError, match="READTHEDOCS_CANONICAL_URL"):
        publishing.publish(project, environ=environment)
    assert renderer == []
    assert not (Path(environment["READTHEDOCS_OUTPUT"]) / "html").exists()


def test_publish_preserves_the_cause_of_a_canonical_url_parse_error(project, environment, renderer):
    environment["READTHEDOCS_CANONICAL_URL"] = "https://[broken"
    with pytest.raises(ValueError, match="READTHEDOCS_CANONICAL_URL") as failure:
        publishing.publish(project, environ=environment)
    assert isinstance(failure.value.__cause__, ValueError)


@pytest.mark.parametrize("output", ["", ".", "docs", "src", "evidence", "tests", ".git", ".github"])
def test_publish_rejects_missing_output_and_source_tree_destinations(project, environment, renderer, output):
    environment["READTHEDOCS_OUTPUT"] = output
    with pytest.raises(ValueError, match="READTHEDOCS_OUTPUT"):
        publishing.publish(project, environ=environment)
    assert renderer == []


@pytest.mark.parametrize("kind", ["file", "directory"])
def test_publish_preserves_existing_output_instead_of_merging_or_overwriting(project, environment, renderer, kind):
    destination = Path(environment["READTHEDOCS_OUTPUT"]) / "html"
    destination.parent.mkdir()
    if kind == "directory":
        destination.mkdir()
        preserved = destination / "existing.html"
    else:
        preserved = destination
    preserved.write_bytes(b"previous publication")
    with pytest.raises(ValueError, match="output must be empty"):
        publishing.publish(project, environ=environment)
    assert preserved.read_bytes() == b"previous publication"
    assert renderer == []


def test_publish_rejects_an_external_symlink_that_resolves_into_source(project, environment, renderer):
    output = project.parent / "output-link"
    output.symlink_to(project / "docs", target_is_directory=True)
    environment["READTHEDOCS_OUTPUT"] = str(output)
    with pytest.raises(ValueError, match="outside the checkout"):
        publishing.publish(project, environ=environment)
    assert renderer == []


def test_publish_rejects_a_source_path_even_when_it_resolves_outside_checkout(project, environment, renderer):
    external = project.parent / "external-source"
    external.mkdir()
    (project / "src").symlink_to(external, target_is_directory=True)
    environment["READTHEDOCS_OUTPUT"] = "src"
    with pytest.raises(ValueError, match="outside the checkout"):
        publishing.publish(project, environ=environment)
    assert not (external / "html").exists()
    assert renderer == []


@pytest.mark.parametrize("phase", ["check", "test", "build"])
def test_publish_does_not_copy_stale_output_after_build_failure(project, environment, renderer, monkeypatch, phase):
    site = project / "docs/starlight/dist"
    site.mkdir()
    (site / "old.html").write_text("Old unvalidated output")
    execute = publishing.subprocess.run

    def fail(command, **options):
        if command[0] == "npm" and command[-1] == phase:
            raise subprocess.CalledProcessError(1, command)
        return execute(command, **options)

    monkeypatch.setattr(publishing.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError):
        publishing.publish(project, environ=environment)
    assert not (Path(environment["READTHEDOCS_OUTPUT"]) / "html").exists()


@pytest.mark.parametrize(
    "invalid",
    [
        '<link rel="canonical" href="https://relflow.readthedocs.io/en/latest/"><script src="missing.js"></script>',
        '<link rel="canonical" href="https://relflow.readthedocs.io/">',
        '<link rel="canonical" href="https://relflow.readthedocs.io/en/latest/wrong-page/">',
        '<link rel="canonical" href="https://relflow.readthedocs.io/en/latest/">'
        '<a href="guide/#missing">Absent fragment</a>',
        '<link rel="canonical" href="https://relflow.readthedocs.io/en/latest/">'
        '<a href="/installation/">Escaped version</a>',
        '<link rel="canonical" href="https://relflow.readthedocs.io/en/latest/">'
        '<a href="https://github.com/relflow/relflow/blob/main/docs/starlight/src/content/docs/index.mdx">Source</a>',
    ],
)
def test_publish_copies_nothing_when_rendered_assets_or_canonical_prefix_fail_validation(
    project, environment, renderer, monkeypatch, invalid
):
    execute = publishing.subprocess.run

    def invalidate(command, **options):
        result = execute(command, **options)
        if command == ["npm", "run", "build"]:
            (project / "docs/starlight/dist/index.html").write_text(invalid)
        return result

    monkeypatch.setattr(publishing.subprocess, "run", invalidate)
    with pytest.raises(ValueError, match="rendered site failed validation"):
        publishing.publish(project, environ=environment)
    assert not (Path(environment["READTHEDOCS_OUTPUT"]) / "html").exists()


@pytest.mark.parametrize("kind", ["external", "branch"])
def test_preview_builds_select_only_released_measurements(project, environment, renderer, monkeypatch, kind):
    (project / "evidence/experiments").mkdir(parents=True)
    environment.update(
        READTHEDOCS_VERSION_TYPE=kind,
        READTHEDOCS_GIT_IDENTIFIER="102" if kind == "external" else "latest",
        RELFLOW_EVIDENCE_PUBLICATION="/unused/ci-preview.json",
    )
    selected = project.parent / "released-publication.json"
    selected.write_text("{}")
    calls = []
    execute = publishing.subprocess.run

    def run(command, **options):
        if command[0] == "npm":
            calls.append(options["env"]["RELFLOW_EVIDENCE_PUBLICATION"])
        return execute(command, **options)

    def fetch(root, version, *, required):
        assert root == project and version is None and required is False
        return selected

    monkeypatch.setattr(publishing, "fetch", fetch)
    monkeypatch.setattr(publishing.subprocess, "run", run)
    publishing.publish(project, environ=environment)
    assert calls == [str(selected)] * 3


def test_unreleased_hosted_build_cannot_keep_an_ad_hoc_ci_override(project, environment, renderer, monkeypatch):
    (project / "evidence/experiments").mkdir(parents=True)
    environment.update(READTHEDOCS_VERSION_TYPE="branch", RELFLOW_EVIDENCE_PUBLICATION="/unused/ci-preview.json")
    monkeypatch.setattr(publishing, "fetch", lambda *args, **kwargs: None)
    publishing.publish(project, environ=environment)
    assert all("RELFLOW_EVIDENCE_PUBLICATION" not in options for options in renderer)


def test_read_only_check_validates_existing_output_without_fetching_or_building(
    project, environment, renderer, monkeypatch
):
    publishing.publish(project, environ=environment)
    monkeypatch.setenv("RELFLOW_DOCS_URL", environment["READTHEDOCS_CANONICAL_URL"])
    monkeypatch.setenv("RELFLOW_DOCS_REVISION", environment["READTHEDOCS_GIT_COMMIT_HASH"])
    monkeypatch.setattr(publishing, "ROOT", project)
    monkeypatch.setattr("sys.argv", ["docs.publish", "--check", "--directory", str(project / "docs/starlight/dist")])

    def forbidden(*args, **kwargs):
        raise AssertionError("Static validation must not fetch Evidence or rebuild the site")

    monkeypatch.setattr(publishing, "fetch", forbidden)
    monkeypatch.setattr(publishing.subprocess, "run", forbidden)
    publishing.main()
