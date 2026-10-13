# relflow, Visualized

A series of short explainer videos about relflow: what it does, how each piece
works, and what it costs. There are 21 episodes of 1½ to 3½ minutes each. Every episode
stands on its own, opens with its own title card and ends on a card naming the
next one, so they work as a YouTube playlist and embedded next to the matching
docs pages.

The rendered episodes aren't committed here, to keep about 130 MB of video out of
a library repository; the commands below rebuild them (1080p, with subtitles and
chapters). [`EPISODES.md`](EPISODES.md) lists every episode with its length, a
description ready for YouTube, and the docs pages it belongs on. The full
narration is in [`SCRIPT.md`](SCRIPT.md).

Schematics are labeled `schematic`. Every other chart, grid and number comes from
running relflow 0.2.3 (commit `5077716`) on the computer that rendered the video,
an Apple M1, and the narration reads its numbers from that data.

## What's in this folder

| Path | What |
|---|---|
| `EPISODES.md` | titles, descriptions, lengths and docs pages |
| `SCRIPT.md` | the narration, exported from the scene code |
| `manifest.py` | episode order, titles, docs pages, and the music-bed settings |
| `s01_hook.py` … `s21_evidence.py` | one [Manim](https://www.manim.community/) scene per episode, with its narration inline |
| `endcards.py` | the end card after each episode |
| `common.py` | shared drawing helpers and the loader for `data/` |
| `compute.py` | every relflow experiment the series shows; writes `data/*.json` |
| `data/` | the committed results the scenes read |
| `assets/` | the relflow logo and the Libertinus Serif wordmark face (SIL OFL 1.1, `assets/OFL.txt`) |
| `episodes.py` | writes `EPISODES.md` from the manifest and the rendered files |
| `VIDEO.md` | the series' production notes: every episode's content and data, method notes, sound, sources |

## Rendering

The scenes are built on the Manim-based framework in
[Sir-Teo/explainer-videos](https://github.com/Sir-Teo/explainer-videos) (styles,
Kokoro narration, the build, music and publishing tools), and import it as
`explainer` and themselves as `videos.relflow`. This folder is a copy of
`videos/relflow/` on that repository's `claude/relflow-series` branch at commit `6c22be4`;
render from that checkout:

```bash
git clone https://github.com/Sir-Teo/explainer-videos && cd explainer-videos
git checkout 6c22be4
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e .                                  # needs ffmpeg and Manim's system dependencies
python -m videos.relflow.compute                     # optional: re-run every relflow experiment (~20 min on an M1)
python tools/build.py relflow --episodes             # 1080p30: the full cut and all 21 episodes
python tools/publish.py relflow --episodes           # GitHub-sized copies -> published/relflow-episodes/
python -m videos.relflow.episodes                    # EPISODES.md
```

The results the scenes read are committed, so the videos render without relflow
or torch. `compute.py` needs `relflow==0.2.3` with Polars < 2, as pinned in that
repository's `pyproject.toml`.
