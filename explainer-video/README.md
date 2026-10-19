# relflow, Visualized

25 narrated Manim episodes explaining how nested records become a model. This
refresh follows the [current docs](https://relflow.github.io/docs/) at
`b46f1b0` and the Relflow 0.3.2 runtime at `2bb35ad`.

[Watch the complete video (480p MP4)](relflow-narrated-480p.mp4), including
narration, embedded subtitles and chapter markers.
The [25 individual chapter videos](chapters-480p/README.md) are in a separate
folder, with narration and subtitles preserved.

Start with one payment and a hidden dispute label, then explore the model tree,
Arrow layout, parent-relative queries, value states, field representations,
branch attention, reduction, output, masks, pretraining, and measured limits.
The four new episodes cover query paths, jitter, Set/Cluster, and prediction
format 2. The schema episode now animates changing batch widths around an
unchanged model. The training episode now follows one real 0.3.2 run through
11 checkpoints: attention, learned parameters, paired predictions, gradients,
AdamW updates and moving representations, followed by controlled input changes.

The visual storytelling pass now spans all 24 other chapters. Records and tokens
retain their identity through placement, packing and output; messages visibly
travel through encoders and decoder heritage. New measured views show SKU lookup,
local contextualization, residual additions, and query/key scores becoming
softmax weights and weighted values. Pretraining compares full-vector cosine
neighborhoods and group means, excluding self-pairs and showing their overlap.

## Build from this branch

Use the current [explainer-videos toolkit](https://github.com/Sir-Teo/explainer-videos)
and its environment (Manim 0.22+, ffmpeg, LaTeX, Kokoro). Source stays in this
folder; the runner creates an ignored local `videos/ml/relflow` symlink in the
framework. No old framework commit or copied scene tree is required.

```bash
# From the relflow repository, with a sibling explainer-videos checkout:
../explainer-videos/.venv/bin/python explainer-video/render.py -q l
../explainer-videos/.venv/bin/python explainer-video/render.py --episodes

# Fast layout QA, separate from narrated output:
EXPLAINER_TTS=silent ../explainer-videos/.venv/bin/python explainer-video/render.py -q l --fps 15

# Rebuild end cards while reusing completed scene movies:
../explainer-videos/.venv/bin/python explainer-video/render.py --episodes --cards-only

# A selection can be built and stitched independently:
../explainer-videos/.venv/bin/python explainer-video/render.py -q l --only QueryPaths Jitter Output
```

Pass `--framework /absolute/path/to/explainer-videos` for another location.
Outputs are ignored under `explainer-video/renders/`; scene movies, narration
metadata and segments are under `explainer-video/media/`. Masters include
normalized narration, subtitles and chapters. `--episodes` adds end cards and
writes individual episode movies. This refresh uses narration without the old
music bed, which the current shared toolkit no longer provides.

## Evidence stays attached to its version

- `data/current.json`: freshly executed **0.3.2** geometry, output, query and
  jitter traces, plus one train/validation batch exercising `fit`. The output
  example is explicitly **untrained**, and is evidence of shape and API behavior.
- `data/schema_current.json`: current schema geometry, presets, and per-node parameter census (21.3M for the medium customer model).
- `data/learning_current.json`: a fresh **0.3.2** training run, seed 0, 4,096
  training orders, 512 held-out orders, 256 AdamW updates. Includes all four
  summary heads on three fixed probes, one fixed 8×8 parameter slice, measured
  gradients/updates and root embeddings in a common final PCA basis. Animated
  transitions interpolate recorded checkpoints; they are not extra measurements.
- `data/flow_current.json`: post-training **0.3.2** intermediate tensors on the
  same three probes: coordinate/sequence/reduction attention, raw SKU vectors,
  residual deltas, summaries and root vectors. Also includes a separate seed-73
  **untrained** Vector projection fixture. `compute_learning.py` records both.
- The original 17 experiment JSON files: unchanged **0.2.3** recordings on an Apple M1.
  They remain useful learning illustrations, with archive labels on episode
  cards and measured charts. Their throughput, parameter counts and learning
  results are not new 0.3.2 measurements. See `data/provenance.json`.
- `compute_current.py` records current behavior; `compute.py` reproduces the
  historical experiments only in a separate `relflow==0.2.3` environment. For the old proof snapshot, point `RELFLOW_REPO` at an archive checkout of `5077716`; canonical proofs now live in the docs repository.

```bash
.venv/bin/python explainer-video/compute_current.py
.venv/bin/python explainer-video/compute_learning.py
../explainer-videos/.venv/bin/python explainer-video/check.py --docs /path/to/relflow-docs
../explainer-videos/.venv/bin/python explainer-video/export.py
```

[`EPISODES.md`](EPISODES.md) has the playlist and matching doc pages;
[`SCRIPT.md`](SCRIPT.md) is exported narration; [`VIDEO.md`](VIDEO.md) describes
visual design, source mappings, provenance and validation.
