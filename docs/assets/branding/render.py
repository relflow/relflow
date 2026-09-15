"""Render official RelFlow branding with its 35% canopy glow."""

import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

DIRECTORY = Path(__file__).resolve().parent
SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
ET.register_namespace("xlink", "http://www.w3.org/1999/xlink")


def glow(root, paint):
    """Duplicate the canopy within its native canvas so its glow scales with every layout."""
    parents = {child: parent for parent in root.iter() for child in parent}
    paths = [path for path in root.iter(f"{{{SVG}}}path") if path.get("fill") == paint]
    if not paths:
        raise ValueError(f"Logo canopy has no paths painted {paint!r}; pass the canopy color from the logo render.")
    groups = [parents[path] for path in paths]
    canvas = parents[groups[0]]
    if any(group.tag != f"{{{SVG}}}g" or len(group) != 1 or parents[group] is not canvas for group in groups):
        raise ValueError(
            "Logo canopy paths must have individual transform groups in one canvas; check the Typst layout."
        )
    position = list(canvas).index(groups[0])
    if list(canvas)[position : position + len(groups)] != groups:
        raise ValueError("Logo canopy paths must be consecutive before the branches; check the Typst draw order.")

    definitions = ET.Element(f"{{{SVG}}}defs")
    leaves = ET.SubElement(definitions, f"{{{SVG}}}g", {"id": "relflow-canopy"})
    for group in groups:
        canvas.remove(group)
        leaves.append(group)

    # Convert the 195 pt mark's 1.8 pt spread and 3 pt blur into CeTZ's local points.
    unit = 2.6 * 72 / 2.54 / 195
    filtering = ET.SubElement(
        definitions,
        f"{{{SVG}}}filter",
        {
            "id": "relflow-glow",
            "filterUnits": "userSpaceOnUse",
            "x": "0",
            "y": "0",
            "width": str(195 * unit),
            "height": str(195 * unit),
            "color-interpolation-filters": "sRGB",
        },
    )
    ET.SubElement(
        filtering,
        f"{{{SVG}}}feMorphology",
        {
            "in": "SourceGraphic",
            "operator": "dilate",
            "radius": str(1.8 * unit),
            "result": "expanded",
        },
    )
    ET.SubElement(
        filtering,
        f"{{{SVG}}}feGaussianBlur",
        {
            "in": "expanded",
            "stdDeviation": str(3 * unit),
        },
    )
    root.insert(0, definitions)
    canvas.insert(
        position,
        ET.Element(
            f"{{{SVG}}}use",
            {
                "href": "#relflow-canopy",
                "filter": "url(#relflow-glow)",
                "opacity": "0.35",
            },
        ),
    )
    canvas.insert(position + 1, ET.Element(f"{{{SVG}}}use", {"href": "#relflow-canopy"}))
    return root


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
    paints = {}
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
                name = f"{asset}.{theme}.svg"
                destination = Path(temporary) / name
                subprocess.run(
                    [
                        "typst",
                        "compile",
                        "--root",
                        str(DIRECTORY),
                        "--ignore-system-fonts",
                        "--input",
                        f"theme={theme}",
                        str(source),
                        str(destination),
                    ],
                    cwd=DIRECTORY,
                    check=True,
                )
                root = ET.parse(destination).getroot()
                if asset == "logo":
                    # The logo draws its tile first, then the canopy, then the two branches.
                    paints[theme] = list(root.iter(f"{{{SVG}}}path"))[1].get("fill")
                glow(root, paints[theme])
                root.set("role", "img")
                root.set("aria-label", title)
                ET.SubElement(root, f"{{{SVG}}}title").text = title
                themes[theme] = root
                data = ET.tostring(root, encoding="utf-8") + b"\n"
                outputs[directory / name] = data
                destination.write_bytes(data)
                png = destination.with_suffix(".png")
                subprocess.run(
                    ["rsvg-convert", "--dpi-x", "144", "--dpi-y", "144", "--output", str(png), str(destination)],
                    check=True,
                )
                outputs[directory / png.name] = png.read_bytes()
            outputs[directory / f"{asset}.svg"] = (
                ET.tostring(adaptive(themes["light"], themes["dark"]), encoding="utf-8") + b"\n"
            )
    for name, content in outputs.items():
        (DIRECTORY / name).write_bytes(content)
        print(f"Rendered {name}")


if __name__ == "__main__":
    render()
