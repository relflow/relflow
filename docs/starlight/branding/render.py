"""Prepare the selected vector logo assets without adding a website build dependency."""

import base64
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path

SVG = "http://www.w3.org/2000/svg"
XLINK = "http://www.w3.org/1999/xlink"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", XLINK)
ROOT = Path(__file__).resolve().parents[3]
ASSETS = ROOT / "docs/starlight/src/assets/branding"


def render():
    for theme in ("light", "dark"):
        destination = ASSETS / f"header.{theme}.svg"
        subprocess.run(
            [
                "typst",
                "compile",
                "--root",
                "docs",
                "--font-path",
                "docs/assets/fonts",
                "--ignore-system-fonts",
                "--input",
                f"theme={theme}",
                "docs/starlight/branding/header.typ",
                str(destination),
            ],
            cwd=ROOT,
            check=True,
        )
        root = ET.fromstring(destination.read_bytes())
        # Inline the imported SVG so the mark scales as paths alongside the wordmark.
        for parent in root.iter():
            for index, child in enumerate(list(parent)):
                if child.tag != f"{{{SVG}}}image":
                    continue
                source = child.attrib[f"{{{XLINK}}}href"]
                if not source.startswith("data:image/svg+xml;base64,"):
                    raise ValueError("The header logo must contain vector SVG artwork.")
                mark = ET.fromstring(base64.b64decode(source.split(",", 1)[1]))
                for key in ("x", "y", "width", "height", "preserveAspectRatio"):
                    if key in child.attrib:
                        mark.set(key, child.attrib[key])
                parent.remove(child)
                parent.insert(index, mark)
        destination.write_bytes(ET.tostring(root, encoding="utf-8", xml_declaration=True))

    light = ET.fromstring((ASSETS / "mark.light.svg").read_bytes())
    dark = ET.fromstring((ASSETS / "mark.dark.svg").read_bytes())
    changes = {
        before.attrib["fill"]: after.attrib["fill"]
        for before, after in zip(light.iter(), dark.iter(), strict=True)
        if before.attrib.get("fill") != after.attrib.get("fill")
    }
    style = ET.SubElement(light, f"{{{SVG}}}style")
    rules = "\n".join(f'[fill="{before}"] {{ fill: {after}; }}' for before, after in changes.items())
    style.text = "@media (prefers-color-scheme: dark) {\n" + rules + "\n}"
    ET.SubElement(light, f"{{{SVG}}}title").text = "relflow"
    (ROOT / "docs/starlight/public/favicon.svg").write_bytes(ET.tostring(light, encoding="utf-8", xml_declaration=True))


if __name__ == "__main__":
    render()
