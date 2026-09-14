"""Render themed PNGs and fixed or adaptive SVGs for RelFlow branding."""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def adaptive(light, dark):
    """Keep one drawing, using the second render only to derive dark paint rules."""
    paints = {}
    for first, second in zip(light.iter(), dark.iter(), strict=True):
        if first.tag != second.tag or first.text != second.text:
            raise ValueError("Theme renders must have identical geometry and text; change only their colors.")
        for name in first.attrib.keys() | second.attrib.keys():
            value, replacement = first.get(name), second.get(name)
            if value == replacement:
                continue
            if name not in ("fill", "stroke") or value is None or replacement is None:
                raise ValueError(f"Theme renders differ in {name!r}; only fill and stroke colors may differ.")
            key = (name, value)
            if paints.setdefault(key, replacement) != replacement:
                raise ValueError(f"Light paint {value!r} maps to multiple dark colors; use distinct light paints.")
    style = ET.Element(f"{{{SVG}}}style")
    rules = [f'  [{name}="{value}"] {{ {name}: {replacement}; }}' for (name, value), replacement in paints.items()]
    style.text = "\n@media (prefers-color-scheme: dark) {\n" + "\n".join(rules) + "\n}\n"
    light.insert(1, style)
    return light


def render():
    outputs = {}
    with tempfile.TemporaryDirectory(prefix="relflow-branding-") as temporary:
        for asset, title in (
            ("logo", "RelFlow"),
            ("banner", "relflow"),
            ("preview", "relflow"),
        ):
            directory = Path(f"{asset}s")
            source = directory / f"{asset}.typ"
            themes = {}
            for theme in ("light", "dark"):
                for extension in ("svg", "png"):
                    name = f"{asset}.{theme}.{extension}"
                    destination = Path(temporary) / name
                    subprocess.run(
                        [
                            "typst",
                            "compile",
                            "--root",
                            str(DIRECTORY),
                            "--ignore-system-fonts",
                            "--ppi",
                            "144",
                            "--input",
                            f"theme={theme}",
                            str(source),
                            str(destination),
                        ],
                        cwd=DIRECTORY,
                        check=True,
                    )
                    if extension == "png":
                        outputs[directory / name] = destination.read_bytes()
                        continue
                    root = ET.parse(destination).getroot()
                    root.set("role", "img")
                    root.set("aria-label", title)
                    ET.SubElement(root, f"{{{SVG}}}title").text = title
                    themes[theme] = root
                    outputs[directory / name] = ET.tostring(root, encoding="utf-8") + b"\n"
            outputs[directory / f"{asset}.svg"] = (
                ET.tostring(adaptive(themes["light"], themes["dark"]), encoding="utf-8") + b"\n"
            )
    for name, content in outputs.items():
        (DIRECTORY / name).write_bytes(content)
        print(f"Rendered {name}")


if __name__ == "__main__":
    render()
