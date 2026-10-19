"""Build this checkout using the current explainer-videos toolkit.

python explainer-video/render.py --framework ../explainer-videos -q l
python explainer-video/render.py --framework ../explainer-videos --episodes
Only a local ignored package symlink is created in the framework; source stays here.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--framework", type=Path, default=HERE.parents[1] / "explainer-videos")
    parser.add_argument("-q", "--quality", choices=["l", "m", "h", "p", "k"], default="h")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("-j", "--jobs", type=int, default=2)
    parser.add_argument("--only", nargs="+", help="Build and stitch just these scene classes")
    parser.add_argument("--episodes", action="store_true")
    parser.add_argument("--stitch-only", action="store_true")
    parser.add_argument(
        "--cards-only", action="store_true", help="Rebuild episode end cards while reusing scene movies"
    )
    args = parser.parse_args()
    if args.cards_only and not args.episodes:
        parser.error("--cards-only requires --episodes")
    framework = args.framework.resolve()
    if not (framework / "explainer/voiceover.py").exists():
        parser.error(f"No explainer-videos toolkit at {framework}; pass --framework")
    link = framework / "videos/ml/relflow"
    if link.exists() and link.resolve() != HERE:
        parser.error(f"{link} already contains a different video; choose another framework checkout")
    if not link.exists():
        link.symlink_to(HERE, target_is_directory=True)
    exclude = framework / ".git/info/exclude"
    if exclude.exists() and "/videos/ml/relflow" not in exclude.read_text():
        with exclude.open("a") as f:
            f.write("\n/videos/ml/relflow\n")
    os.environ["EXPLAINER_VIDEO"] = "relflow"
    os.environ.setdefault("EXPLAINER_LANG", "en")
    sys.path.insert(0, str(framework))
    # Manim subprocesses must also be able to resolve the current framework packages.
    os.environ["PYTHONPATH"] = str(framework) + os.pathsep + os.environ.get("PYTHONPATH", "")
    spec = importlib.util.spec_from_file_location("video_build", framework / "tools/build.py")
    build = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(build)
    from explainer.voiceover import subtitle_cues, to_srt
    from videos.ml.relflow.manifest import EPISODES

    selected = [e for e in EPISODES if not args.only or e["cls"] in args.only]
    if args.only and set(args.only) - {e["cls"] for e in selected}:
        parser.error("Unknown scene classes: " + ", ".join(set(args.only) - {e["cls"] for e in selected}))
    # Separate silent-layout QA from narrated masters and keep every product local to this branch.
    mode = "silent" if os.environ.get("EXPLAINER_TTS") == "silent" else "narrated"
    media = HERE / "media" / mode
    output = HERE / "renders"
    output.mkdir(exist_ok=True)
    scenes = [(e["module"], e["cls"]) for e in selected]
    if args.episodes:
        scenes += [tuple(e["end"]) for e in selected]
    if not args.stitch_only:
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            render_targets = [tuple(e["end"]) for e in selected] if args.cards_only else scenes
            futures = [
                pool.submit(build.render_scene, "relflow", m, c, args.quality, args.fps, media)
                for m, c in render_targets
            ]
            failed = []
            for future in as_completed(futures):
                cls, seconds, error = future.result()
                print(f"{'FAIL' if error else 'ok'} {cls}: {seconds:.0f}s", flush=True)
                if error:
                    print(error, flush=True)
                    failed.append(cls)
        if failed:
            raise SystemExit("Failed scenes: " + ", ".join(failed))
    selection = hashlib.sha1(" ".join(e["cls"] for e in selected).encode()).hexdigest()[:8] if args.only else "full"
    work = media / "stitch" / f"{args.quality}{args.fps}" / selection
    work.mkdir(parents=True, exist_ok=True)
    normalized = {}
    cues = {}
    durations = {}
    for m, c in scenes:
        source = build.scene_output(media, m, c, args.quality, args.fps)
        if not source.exists():
            raise SystemExit(f"Missing render: {source}")
        meta = source.with_suffix(".narration.json")
        lines = json.loads(meta.read_text())["lines"] if meta.exists() else []
        build.check_narration_audio(source, lines)
        dest = work / f"{c}.mp4"
        build.normalize(source, dest)
        normalized[c] = dest
        durations[c] = build.ffprobe_duration(dest)
        cues[c] = subtitle_cues(lines)

    def stitch(entries, dest):
        paths = []
        subtitles = []
        chapters = []
        offset = 0.0
        for cls, title in entries:
            paths.append(normalized[cls])
            chapters.append((offset, title))
            subtitles += [(a + offset, b + offset, t) for a, b, t in cues[cls]]
            offset += durations[cls]
        playlist = work / "concat.txt"
        # ffconcat quoting for absolute paths containing apostrophes.
        playlist.write_text("".join("file '" + str(p).replace("'", "'\\''") + "'\n" for p in paths))
        joined = work / "joined.mp4"
        build.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                str(playlist),
                "-c",
                "copy",
                str(joined),
            ]
        )
        srt = dest.with_suffix(".srt")
        srt.write_text(to_srt(subtitles))
        chapter_file = work / "chapters.ffmeta"
        text = ";FFMETADATA1\n"
        for i, (start, title) in enumerate(chapters):
            end = chapters[i + 1][0] if i + 1 < len(chapters) else offset
            text += f"[CHAPTER]\nTIMEBASE=1/1000\nSTART={round(start * 1000)}\nEND={round(end * 1000)}\ntitle={title}\n"
        chapter_file.write_text(text)
        filt = (
            build.HIGHPASS
            if mode == "silent"
            else build.master_filter(
                [
                    "ffmpeg",
                    "-v",
                    "info",
                    "-i",
                    str(joined),
                    "-vn",
                    "-af",
                    build.LOUDNORM + ":print_format=json",
                    "-f",
                    "null",
                    "-",
                ]
            )
        )
        build.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-i",
                str(joined),
                "-i",
                str(srt),
                "-i",
                str(chapter_file),
                "-map",
                "0:v",
                "-map",
                "0:a",
                "-map",
                "1:0",
                "-map_metadata",
                "2",
                "-map_chapters",
                "2",
                "-c:v",
                "copy",
                "-af",
                filt,
                "-c:a",
                "aac",
                "-b:a",
                "192k",
                "-c:s",
                "mov_text",
                "-movie_timescale",
                "1000",
                "-movflags",
                "+faststart",
                str(dest),
            ]
        )
        dest.with_suffix(".chapters.txt").write_text(
            "\n".join(f"{int(t // 60):02d}:{int(t % 60):02d} {title}" for t, title in chapters) + "\n"
        )
        print(dest, flush=True)

    suffix = f"-selection-{selection}" if args.only else ""
    full = output / f"relflow-{mode}-{build.QUALITY[args.quality]}{suffix}.mp4"
    stitch([(e["cls"], e["title"]) for e in selected], full)
    if args.episodes:
        folder = output / f"episodes-{mode}-{build.QUALITY[args.quality]}"
        folder.mkdir(exist_ok=True)
        for e in selected:
            stitch(
                [(e["cls"], e["title"]), (e["end"][1], "Next episode")], folder / f"{e['number']:02d}-{e['slug']}.mp4"
            )


if __name__ == "__main__":
    main()
