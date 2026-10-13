# relflow, Visualized: How a Nested Schema Becomes a Transformer

A 14-chapter explainer of [relflow](https://github.com/relflow/relflow), a library
that builds PyTorch/Lightning models straight from nested records: you write the
shape of one observation (fields, repeated branches, targets) and that schema *is*
the network. The video follows one batch through the whole machine, with the
emphasis on how each piece works and why:

* **Records to rectangles.** Arrow's offsets and child arrays; relflow's single
  placement rule (`parent × length + slot`), head/tail overflow, nested branches;
  a value *state* (valued, null, padded, masked) beside every value, and why a
  null must not become a zero.
* **Embedders.** Number: streaming mean/variance merged exactly, 13 octave-spaced
  Fourier frequencies (π/256 … 16π), and a raw "ruler" lane for extrapolation.
  Category: an online vocabulary that grows geometrically while training, carrying
  Adam's moments across every resize. Hash: per-batch salted fingerprints that
  give equality without identity.
* **Inside a branch.** Coordinate attention among the fields of one item, the
  unrolled sequence, rotary positions (an item's SKU and its own quantity are
  always exactly `length` positions apart), pre-norm layers, and packing, which
  skips padding exactly.
* **Reduction.** Learned-query cross-attention, why a softmax average cannot
  count, and relflow's additive *mass* and *count* lanes that let it.
* **Routing and decoding.** The depth-first execution plan, the shapes at every
  hand-off, decoders that pool their ancestor path with target-shaped learned
  queries, and masks as one mechanism for targets, dropout and self-supervision
  (with deterministic, SHA-256-seeded selection).
* **Evidence.** A real training run whose attention can be read, exported
  embeddings, and relflow's own proof catalog, limits included.

Schematics are labeled `schematic`. Every other chart, grid and number is produced
by running relflow itself (release 0.2.3) on the computer that rendered the video,
and the numbers the narration states are asserted against that data at render time.

**Watch:** [`relflow.mp4`](relflow.mp4) (1080p, about 33 min; subtitles and chapters embedded,
also as [`relflow.srt`](relflow.srt)). The full narration is in [`SCRIPT.md`](SCRIPT.md).

<details><summary>Chapters</summary>

- `0:00` Records have shape
- `2:40` The schema is the architecture
- `4:56` From records to rectangles
- `7:07` Null is not zero
- `9:10` How a number becomes a vector
- `11:49` A vocabulary that grows while it trains
- `14:25` Equality without identity
- `16:29` Inside a branch
- `19:18` Don't pay for padding
- `21:23` Why softmax can't count
- `24:36` Climbing the tree
- `26:13` Asking the tree a question
- `29:03` Watching it learn
- `31:23` What the proofs show, and what they don't

</details>

## What's in this folder

| Path | What |
|---|---|
| `relflow.mp4`, `relflow.srt` | the published video and its subtitles |
| `SCRIPT.md` | the narration, exported from the scene code |
| `manifest.py` | scene order, chapter titles and the music-bed settings |
| `s01_hook.py` … `s14_evidence.py` | one [Manim](https://www.manim.community/) scene per chapter, with its narration inline |
| `common.py` | shared drawing helpers and the loader for `data/` |
| `compute.py` | every relflow experiment the video shows; writes `data/*.json` |
| `data/` | the committed results the scenes read (about 380 kB) |
| `assets/` | the relflow logo and the Libertinus Serif wordmark face (SIL OFL 1.1, `assets/OFL.txt`) |

## Rendering

The scenes are built on the Manim-based framework in
[Sir-Teo/explainer-videos](https://github.com/Sir-Teo/explainer-videos) (styles, Kokoro narration, the build and music tools), and import
it as `explainer` and themselves as `videos.relflow`. This folder is a copy of `videos/relflow/` at
[`e67f761`](https://github.com/Sir-Teo/explainer-videos/tree/e67f7619437c289bf277f6352c4143010700c58c/videos/relflow); render from that checkout:

```bash
git clone https://github.com/Sir-Teo/explainer-videos && cd explainer-videos
git checkout e67f761
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install --index-url https://download.pytorch.org/whl/cpu torch
uv pip install -e .                         # needs ffmpeg and Manim's system dependencies
python -m videos.relflow.compute            # optional: re-run every relflow experiment (about 5 min on 4 cores)
python tools/build.py relflow -q l          # preview
python tools/build.py relflow               # final 1080p30 + subtitles + chapters + music bed
python tools/check_mix.py relflow           # Whisper: does the music bed cost intelligibility?
python tools/export_script.py relflow       # regenerate SCRIPT.md from the code
python tools/check_narration.py relflow     # Whisper listen test of every narration line
python tools/publish.py relflow             # GitHub-sized copy -> published/relflow.mp4
```

The results the scenes read (`data/*.json`) are committed, so the video renders without relflow
or torch. `compute.py` needs `relflow==0.2.3` with Polars < 2 (it hashes with
`Series.hash(seed_1=...)`), as pinned in that repository's `pyproject.toml`.

## Outline

| # | Scene | What it establishes | Real data on screen |
|---|---|---|---|
| – | `Hook` | A customer record as YAML; what flattening loses; the `rf.Model(...)` code becoming a tree of cards and then a network; the road map. | The customer model's 12,156,172 parameters (preset `md`). |
| 1 | `SchemaIsArchitecture` | Addresses (`/orders/line_items/sku`); branch lengths fix every shape; what a leaf and a branch own; where the parameters live; the five presets. | Per-node parameter counts of the real model; the SKU embedder's 1,536 parameters with an empty vocabulary; `rf.presets` XS–XL. |
| 2 | `RaggedToDense` | Arrow's offsets + child arrays; `placement = parent × length + slot`; head vs tail overflow; nested placement; the state grid. | `model.encode(...)` and relflow's own `ragged.descend` on four orders: placements `[0, 1, 4, 12, 13, 14, 15]`, V/N/P states, batch-local SKU ids. |
| 3 | `ValueState` | Five state tokens with learned vectors, added to content; null is a coordinate, padding is not; empty vs null lists; unknown labels. | Proof P038: AUC 1.00 with nulls kept, 0.50 when nulls are filled with 0.0 first, 0.51 with random nulls. |
| 4 | `Numbers` | Streaming standardization with exact moment merging; 13 Fourier frequencies; why a weighted sum of them can make sharp effects; the raw ruler lane; the clamp; decoding in standard deviations. | relflow's `GlobalOnlineNormalizer` over six real batches matching the exact mean and variance; frequencies and the 199-σ clamp read from the embedder. |
| 5 | `Vocabulary` | Online admission; worker-local dictionaries bound before forward; geometric growth; optimizer state carried across a resize; validation and serving never admit labels; `p_unavailable`. | A real run with new merchants arriving: rows 1 → 4 → 8 → 16 → 32 → 64, with old rows and Adam moments identical and new moments exactly 0 at every growth. |
| 6 | `Hashing` | Why identifiers need equality, not identity; salted hash lanes through fixed Fourier features; a new salt every training batch; salt 0 at test/predict. | The real Hash embedder's vectors and similarity matrices in three batches; P021 (AUC 0.9999 vs 0.50 for Category on unseen IDs) and its limit P028 (0.47). |
| 7 | `InsideABranch` | Coordinate attention; the unrolled 24-position sequence; rotary positions make score depend only on distance; pre-norm layers; MHA/GQA/MQA/None. | Toeplitz scores from relflow's `RotaryEmbedding`; a trained model at AUC 1.000 that falls to 0.906 when quantities are shuffled among each order's items. |
| 8 | `Packing` | Attention cost on padding; gather live tokens, group rows by length, attend without masks, scatter back. | relflow's `pack`/`unpack` on a real batch (72% of pair work would be padding); dense vs packed encoder timed on this CPU: 248 → 14 ms (17×), 231 → 37 ms, 201 → 95 ms; outputs equal to ~10⁻⁶. |
| 9 | `Reduction` | Learned-query cross-attention; softmax weights sum to one, so duplicates cannot be counted; the mass and count lanes (zero-initialized); the three reduction choices. | Three real training runs on bags of 1–6 equal amounts: `Mean` answers 4.39 and attention without the lanes 4.22 for 1–10 copies; with the lanes 1.26, 2.47, 3.70, 4.94, 6.18, 7.44, then flattening past the trained range. |
| 10 | `Routing` | The customer model bottom-up: 32 tokens per order → 4 summaries; 64 × (1 + 4) = 320 tokens → 4; 7 root slots → 4; the depth-first plan. | Input widths, outputs and the plan from `model.schema`. |
| 11 | `Decoders` | Heritage paths; target-shaped learned queries nudged by aligned siblings; state + content heads and the loss; masks for targets, dropout, self-supervision, data-driven and subtree selection; deterministic sampling; stages. | Heritage paths from the schema; real mask selections at 15% for two runs of epoch 0, epoch 1 and validation. |
| 12 | `RealRun` | A model learns which line item makes an order come back; reading its reduction attention; exported embeddings; the prediction envelope. | 4,096 orders, 9 s on CPU, validation AUC 0.96 → 1.00; two of four heads put ≈ 1.0 of their weight on the decisive item's SKU; PCA of the exported root vectors separates the classes; a real output row. |
| 13 | `Evidence` | The proof catalog: 78 proofs, 55 passing, 13 limited, 10 partial, 7 missing their own gates in the latest run; highlights and recorded limits; recap. | `proofs/results.yaml` at relflow 5077716: P040, P076, P044, P028, P066, P071. |

## Sound

The narrator is Kokoro-82M `af_heart`, as in the other videos, with new lexicon
entries for relflow, PyTorch, Parquet, YAML, SKU, RoPE, SDPA and friends (checked
with a Whisper listen test). This is the first video in the repository with music:

* **A generated music bed** ([`tools/music.py`](https://github.com/Sir-Teo/explainer-videos/blob/e67f7619437c289bf277f6352c4143010700c58c/tools/music.py)): additive
  "felt piano" notes with inharmonic partials and two-stage decays, a detuned pad,
  a sine bass and a shaped-noise convolution reverb, on a slow I–vi–IV–V loop in
  D major at 66 bpm. Each chapter restarts the phrase with a rolled chord. No
  melody, nothing above a soft register, and a dip around 2.5 kHz so it stays out
  of the voice's presence band.
* **Ducking that anticipates the voice.** The whole narration is known in
  advance, so the bed drops *before* each phrase (fast) and recovers slowly: about
  22 dB below the voice while it speaks, 16 dB below in longer pauses, and 9 dB
  below over the title, the chapter cards and the ending.
* **A light voice chain**: 70 Hz high-pass, 2:1 compression that only touches
  peaks, and a 2 dB dip at 7 kHz for sibilants. The mix is then mastered to
  −16 LUFS integrated (EBU R128, −1.5 dBTP) as before.
* **Checked, not just tuned.** On 24 narration lines sampled across the first
  four chapters, Whisper (`small.en`) transcribes the raw voice at 17.70% word
  error against the script, the processed voice at 17.70%, and the final mix
  with the music bed at 17.68% (the absolute rate counts things like "12" for
  "twelve"; the comparison is the point). The bed itself was checked for pitch
  content (all energy on D-major pitch classes), clicks (none: the largest
  sample-to-sample jump is half the local level) and spectrum (it sits 15 dB
  down above 1 kHz and 40 dB down above 3 kHz, below the consonants).
  `tools/check_mix.py` repeats the voice-vs-mix comparison on a finished build.

## Data (`compute.py` → `data/*.json`)

| Item | What | Used in |
|---|---|---|
| `schema` | the customer model (`rf.Model.md`) node by node: kinds, shapes, heritage, decoder context, encoder input widths and outputs, parameters per module; the README order model; `rf.presets` | Hook, 1, 10, 11 |
| `encode` | four orders through `ragged.root`/`ragged.descend` and `model.encode` (head and tail); deterministic mask selection on 12 orders × 16 slots | 2, 11 |
| `number` | the Number embedder's frequencies and clamp; `GlobalOnlineNormalizer` fed six lognormal batches, against exact statistics | 4 |
| `vocab` | 14 training batches with three new merchants each; `Model.on_before_batch_transfer`'s binding wrapped to compare the embedding table and Adam's `exp_avg`/`exp_avg_sq` before and after every bind | 5 |
| `hash` | one observation with seven IDs encoded in three training batches and once for prediction: lanes, vectors, cosine similarities | 6 |
| `packing` | `relflow.architecture.packed.pack` on 16 rows; a two-layer `BranchEncoder` (d = 128, 256 × 128 slots) timed dense vs packed at three fill levels, median of 7, with the outputs compared | 8 |
| `counting` | three models (`Mean`; `Attention` with the mass/count lanes deleted; `Attention`) trained 40 epochs on 4,096 bags; predictions on held-out bags of 1–6 and 7–10 items and on 1–10 copies of 1.25 | 9 |
| `returns` | the order model trained 4 epochs on 4,096 synthetic orders; test AUC intact and with quantities shuffled within orders; recomputed reduction attention for four test orders; PCA of exported root embeddings; one real output row | 7, 12 |
| `rope` | scores of one query/key pair at every pair of 24 positions through `RotaryEmbedding(16)` | 7 |
| `proofs` | status and latest full-run outcome of all 78 proofs in `proofs/results.yaml`, and the metrics of the ten cited | 3, 6, 9, 13 |

### Method notes

* **Synthetic data.** The customer, order and bag datasets are generated in
  `compute.py` with fixed seeds; they exist to make one mechanism visible at a
  time, not to benchmark relflow. Re-running `compute.py` reproduces every file
  (the training runs are deterministic on CPU; the timings are not).
* **The returns rule.** An order is returned iff it contains SKU S03 or S11 with
  quantity ≥ 3; 20 SKUs, 1–8 items per order, lognormal prices; 24% of orders are
  returned. The "misaligned" control permutes quantities among the items of each
  test order, keeping the SKUs and the multiset of quantities.
* **Attention weights.** SDPA does not return them, so `compute.py` hooks the
  reduction's first cross-attention block and recomputes
  `softmax(rotate(W_q q) · rotate(W_k k)ᵀ / √d)` with the model's own projections
  and padding mask. Tokens 0–7 are SKU slots, 8–15 quantity slots, 16–23 price slots.
* **The mass-lane ablation** (chapter 9) sets `mass_projection` and `mass_direction`
  to `None` in every `LearnedQueryCrossAttention`; it is not a relflow option.
  Branch attention is off (`attention=None`) in all three arms, as in proof P041.
* **Proof numbers** are relflow's recorded results (its `results.yaml`), not
  re-runs. "Limited" and "Partial" are relflow's own labels.
* **Machine.** A 4-vCPU cloud VM; relflow 0.2.3, PyTorch 2.14 (CPU), Lightning 2.6,
  PyArrow 24, Polars 1.44.

## Sources

* relflow: [github.com/relflow/relflow](https://github.com/relflow/relflow) at
  commit `5077716` (Sep 29, 2026), its [documentation](https://relflow.github.io/relflow/),
  `AGENTS.md`, `proofs/SPEC.md` and `proofs/results.yaml`. The logo and the
  Libertinus Serif wordmark face (SIL OFL 1.1, `assets/OFL.txt`) are from the
  relflow repository (Apache-2.0).
* Vaswani et al. (2017), *Attention Is All You Need*; Su et al. (2021), *RoFormer:
  Enhanced Transformer with Rotary Position Embedding*; Xiong et al. (2020), *On
  Layer Normalization in the Transformer Architecture* (pre-norm).
* Tancik et al. (2020), *Fourier Features Let Networks Learn High Frequency
  Functions in Low Dimensional Domains*.
* Chan, Golub & LeVeque (1979), *Updating Formulae and a Pairwise Algorithm for
  Computing Sample Variances* (merging moments).
* Kingma & Ba (2015), *Adam*; Loshchilov & Hutter (2019), *Decoupled Weight Decay
  Regularization* (AdamW).
* Lee et al. (2019), *Set Transformer* (pooling by multihead attention with
  learned seeds), and Zaheer et al. (2017), *Deep Sets*, for the reduction view.
* Apache Arrow columnar format (list offsets and child arrays).
* ITU-R BS.1770 / EBU R128 for loudness.
