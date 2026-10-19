"""Export scripts and playlist metadata from this checkout's rendered narration."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--framework", type=Path, default=HERE.parents[1] / "explainer-videos")
    p.add_argument("--mode", choices=["silent", "narrated"], default="narrated")
    p.add_argument("--quality", default="1080p30")
    a = p.parse_args()
    sys.path.insert(0, str(a.framework.resolve()))
    from tools.export_script import BOOKMARK
    from videos.ml.relflow.episodes import main as episodes
    from videos.ml.relflow.manifest import EPISODES, TITLE

    parts = [
        f"# {TITLE} — narration script\n",
        "_Exported from rendered narration metadata; archive labels and API traces retain their own provenance._\n",
    ]
    words = 0
    for e in EPISODES:
        path = HERE / "media" / a.mode / "videos" / e["module"] / a.quality / (e["cls"] + ".narration.json")
        if not path.exists():
            raise SystemExit(f"Missing narration metadata: {path}; render the full series first")
        lines = json.loads(path.read_text())["lines"]
        parts.append(f"\n## {e['number']}. {e['title']}\n\n_{e['provenance']}_\n")
        for line in lines:
            text = " ".join(BOOKMARK.sub("", line["text"]).split())
            parts.append("\n" + text + "\n")
            words += len(text.split())
    parts.append(f"\n{words:,} narrated words.\n")
    (HERE / "SCRIPT.md").write_text("".join(parts))
    # Durations come only from refreshed narrated episode files.
    episodes(src=HERE / "renders" / "episodes-narrated-1080p", out=HERE / "EPISODES.md", link="")
    print(f"Exported {len(EPISODES)} episodes, {words:,} words.")


if __name__ == "__main__":
    main()
