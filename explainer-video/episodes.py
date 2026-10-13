"""Write EPISODES.md: one entry per published episode, with its file, length, a description for YouTube,
and the relflow docs pages it belongs on.

    python tools/build.py relflow --episodes      # render and stitch first
    python -m videos.relflow.episodes             # then describe what was built
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from videos.relflow.manifest import DOCS, EPISODES, SERIES

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent

DESCRIPTIONS = {
    "records-have-shape": "Most data worth learning from has shape: lists inside records. relflow is a model factory "
    "for structured data: describe the shape of one record and it builds the PyTorch model that reads it. The series "
    "road map.",
    "three-lines": "The happy path: load a schema into a model, build a data module, fit. A real run from those three "
    "lines, what each training step does, Lightning's four loops, and saving, loading and predicting.",
    "schema": "Addresses, fixed shapes, what each leaf and branch owns, where the parameters live, and the five size "
    "presets.",
    "records-to-rectangles": "How nested Arrow records become dense tensors: offsets and child arrays, one placement "
    "rule, head and tail overflow, nested branches, and a state beside every value.",
    "null-is-not-zero": "Five value states (valued, null, padded, masked, other) with learned vectors, and a proof "
    "of why filling nulls with zero loses information.",
    "numbers": "Streaming standardization with exactly merged moments, 13 Fourier frequencies, a raw ruler lane for "
    "extrapolation, and decoding in standard deviations.",
    "vocabulary": "Category vocabularies that grow while the model trains: worker-local dictionaries, geometric "
    "growth, and optimizer state carried across every resize.",
    "hash": "Identifiers that need equality, not identity: salted hash lanes, a new salt every training batch, and "
    "the proofs that show where this works and where it doesn't.",
    "quantile-and-enum": "Quantile: a number as its percentile in a learned t-digest, and when that beats "
    "standardizing. Enum: labels declared up front, fixed rows, and strict validation.",
    "time-vectors-text": "DateParts puts calendar positions on circles; Vector projects supplied embeddings; Text "
    "runs strings through a frozen Hugging Face encoder and learns how to use the result.",
    "inside-a-branch": "Coordinate attention among an item's fields, the unrolled sequence, rotary positions, "
    "pre-norm layers, and a real test that the model reads SKUs and quantities as pairs.",
    "packing": "Padding costs the answers nothing, but it costs the hardware. Packing gathers live tokens, groups "
    "rows by length and attends without masks: timed for real.",
    "softmax-cant-count": "Learned-query cross-attention summarizes a branch, but softmax weights sum to one, so it "
    "can't count. relflow's additive mass and count lanes, tested in three real training runs.",
    "climbing-the-tree": "The customer model bottom up: the shapes at every hand-off and the depth-first execution "
    "plan.",
    "decoders": "Decoders read the tree back down: heritage paths, target-shaped learned queries, and a head and "
    "loss for every datatype.",
    "masks": "One mechanism for targets, dropout, self-supervision and ablation: learned masks versus skipping, "
    "branch-level masks, deterministic SHA-256 sampling, and what each stage does.",
    "pretraining": "Learning without labels: pretrain by reconstructing hidden values, fine-tune with a few labels, "
    "and a real comparison against training from scratch.",
    "processors": "Polars preprocessors reshape records before the schema applies; postprocessors turn predictions "
    "into application output. A real model trained through both.",
    "watching-it-learn": "A real training run, its reduction attention read out head by head, exported embeddings, "
    "and the prediction envelope.",
    "benchmarks": "Training and inference speed and peak memory for every preset, CPU against the M1's GPU, and what "
    "longer contexts cost: measured on a laptop.",
    "proofs": "relflow's proof catalog: what passes, what is limited, what is still missing, and a recap of the "
    "series.",
}


def duration(path: Path) -> float:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return float(out.strip())


def main(src: Path = ROOT / "published" / "relflow-episodes", out: Path = HERE / "EPISODES.md",
         link: str = "../../published/relflow-episodes/"):
    assert set(DESCRIPTIONS) == {e["slug"] for e in EPISODES}
    lines = [f"# {SERIES}: episodes", "",
             f"The series cut into {len(EPISODES)} standalone videos, one per scene, each ending on a card that names "
             "the next episode. Upload them in order as a playlist; each entry below has the title, the description "
             f"and the pages of the [relflow docs](https://{DOCS}/) it belongs on.", "",
             "| # | Title | Length | Docs pages |", "|---|---|---|---|"]
    for e in EPISODES:
        name = f"{e['number']:02d}-{e['slug']}"
        path = src / f"{name}.mp4"
        length = duration(path) if path.exists() else None
        mm = f"{int(length // 60)}:{int(length % 60):02d}" if length else "–"
        docs = ", ".join(f"[`{d}`](https://{DOCS}/{d.replace('.qmd', '.html')})" for d in e["docs"])
        title = f"[{e['title']}]({link}{name}.mp4)" if link else f"{e['title']} (`{name}.mp4`)"
        lines.append(f"| {e['number']} | {title} | {mm} | {docs} |")
    lines += [""]
    for e in EPISODES:
        lines += [f"## {e['number']}. {e['title']}", "", f"**YouTube title:** {e['title']} | {SERIES} #{e['number']}", "",
                  DESCRIPTIONS[e["slug"]], ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=HERE / "EPISODES.md")
    ap.add_argument("--link", default="../../published/relflow-episodes/",
                    help="where EPISODES.md finds the files; empty to name them without linking")
    args = ap.parse_args()
    main(out=args.out, link=args.link)
