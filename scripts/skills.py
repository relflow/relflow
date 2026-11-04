"""Check and package the canonical schema skill for local cross-agent distribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def contents(root: Path) -> tuple[dict, dict[str, bytes]]:
    """Validate metadata and resource containment before assembling either bundle."""
    manifest = json.loads((root / "skills/plugin.json").read_text(encoding="utf-8"))
    name = manifest["name"]
    if not re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name):
        raise ValueError(f"skill name must use lowercase words separated by hyphens, got {name!r}")
    if manifest.get("$schema") != "https://agent-plugins.org/schemas/1.0.0/plugin.schema.json":
        raise ValueError("plugin must declare the Agent Plugins 1.0.0 manifest schema")
    directory = root / "skills" / name
    source = (directory / "SKILL.md").read_text(encoding="utf-8")
    match = re.match(r"\A---\n(.*?)\n---\n", source, re.DOTALL)
    if match is None:
        raise ValueError("SKILL.md requires YAML frontmatter")
    metadata = yaml.safe_load(match[1])
    allowed = {"name", "description", "license", "compatibility", "metadata", "allowed-tools"}
    if not isinstance(metadata, dict) or metadata.keys() - allowed:
        raise ValueError("SKILL.md frontmatter must use Agent Skills standard fields")
    if metadata.get("name") != name or metadata.get("metadata", {}).get("version") != manifest["version"]:
        raise ValueError("skill name and version must match skills/plugin.json")
    description = metadata.get("description")
    if not isinstance(description, str) or not 1 <= len(description) <= 1024:
        raise ValueError("skill description must contain 1 to 1024 characters")
    files = {}
    for path in sorted(directory.rglob("*")):
        if "__pycache__" in path.parts or path.suffix == ".pyc":
            continue
        if path.is_symlink():
            raise ValueError(f"skill resource must be self-contained, not a symlink: {path}")
        if not path.is_file():
            continue
        relative = path.relative_to(directory).as_posix()
        files[relative] = path.read_bytes()
        if path.suffix == ".md":
            for target in re.findall(r"\[[^\]]*\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                linked = (path.parent / target.split("#")[0]).resolve()
                if not linked.is_relative_to(directory.resolve()) or not linked.is_file():
                    raise ValueError(f"skill reference {target!r} in {path.name} is missing or escapes the skill")
    files["LICENSE"] = (root / "LICENSE").read_bytes()
    return manifest, files


def archive(path: Path, files: dict[str, bytes]) -> None:
    """Write deterministic member names, ordering, timestamps, permissions, and bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as output:
        for name, content in sorted(files.items()):
            member = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            member.compress_type = zipfile.ZIP_DEFLATED
            output.writestr(member, content)


def build(output: Path, *, root: Path = ROOT) -> dict:
    """Stage one skill and one dual-manifest plugin without touching agent settings."""
    manifest, files = contents(root)
    name, version = manifest["name"], manifest["version"]
    output.mkdir(parents=True, exist_ok=False)
    skill = {f"{name}/{path}": content for path, content in files.items()}
    plugin = {f"skills/{name}/{path}": content for path, content in files.items()}
    plugin["plugin.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    plugin[".claude-plugin/plugin.json"] = (
        json.dumps({key: value for key, value in manifest.items() if key != "$schema"}, indent=2) + "\n"
    ).encode()
    plugin["LICENSE"] = files["LICENSE"]
    for path, content in plugin.items():
        destination = output / name / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    for kind, members in (("skill", skill), ("plugin", plugin)):
        archive(output / f"{name}-{version}-{kind}.zip", members)
    marketplace = {
        "name": "relflow",
        "owner": {"name": "relflow"},
        "plugins": [{"name": name, "source": f"./{name}"}],
    }
    catalog = output / ".claude-plugin/marketplace.json"
    catalog.parent.mkdir()
    catalog.write_text(json.dumps(marketplace, indent=2) + "\n", encoding="utf-8")
    receipt = {
        "name": name,
        "version": version,
        "skill_files": {path: hashlib.sha256(content).hexdigest() for path, content in sorted(files.items())},
        "artifacts": {
            path.relative_to(output).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(output.rglob("*"))
            if path.is_file()
        },
        "scope": "local packaging; marketplace publication and host behavior are not certified",
    }
    (output / "build.json").write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("check", "build"))
    parser.add_argument("--output", type=Path, default=ROOT / "dist/skills")
    args = parser.parse_args()
    try:
        if args.command == "check":
            manifest, files = contents(ROOT)
            result = {"name": manifest["name"], "version": manifest["version"], "files": len(files), "ok": True}
        else:
            result = build(args.output)
    except (OSError, ValueError, KeyError, yaml.YAMLError) as error:
        parser.exit(1, f"skill packaging failed: {error}\n")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
