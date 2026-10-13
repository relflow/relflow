# relflow, Visualized: A Model Factory for Structured Data

A 21-episode explainer series about [relflow](https://github.com/relflow/relflow), a model
factory for structured data: you describe the shape of one record (its fields, the
lists inside it, and the values to predict) and relflow builds the PyTorch/Lightning
model that reads it. Each episode is one scene, 1½ to 3½ minutes long, and stands on its
own: it opens on its own title card, avoids leaning on earlier episodes, and ends
on a card naming the next one. They're published as separate videos for a YouTube
playlist and for embedding next to the matching pages of relflow's docs (see
[`EPISODES.md`](EPISODES.md)); the same renders also stitch into one full cut.

The series starts with the happy path (a schema, a data module and a fit, in
three lines) and then takes the machine apart: records to rectangles and value
states; how numbers, labels, identifiers, ranks, calendar time, vectors and text
become vectors; branch encoders, packing and reductions; decoders; masks and
pretraining; preprocessing and postprocessing; a real run; what it all costs in
time and memory; and relflow's own proof catalog.

Schematics are labeled `schematic`. Every other chart, grid and number is produced
by running relflow itself (release 0.2.3) on the computer that rendered the video,
an Apple M1, and the numbers the narration states are read from that data, with the
claims they support asserted at render time.

**Watch:** the episodes listed in [`EPISODES.md`](EPISODES.md), or the
full cut, `published/relflow.mp4` in [explainer-videos](https://github.com/Sir-Teo/explainer-videos/blob/6c22be4/published/relflow.mp4) (1080p; subtitles and chapters embedded).

```bash
python -m videos.relflow.compute            # optional: re-run every relflow experiment (about 20 min on an M1)
python tools/build.py relflow -q l          # preview
python tools/build.py relflow --episodes    # final 1080p30: the full cut and all 21 episodes, with the music bed
python tools/publish.py relflow             # GitHub-sized copy -> published/relflow.mp4
python tools/publish.py relflow --episodes  # -> published/relflow-episodes/NN-slug.mp4
python -m videos.relflow.episodes           # EPISODES.md: titles, descriptions, lengths, docs pages
python tools/export_script.py relflow       # regenerate SCRIPT.md from the code
python tools/check_narration.py relflow     # Whisper listen test of every narration line
python tools/check_mix.py relflow           # Whisper: does the music bed cost intelligibility?
```

The results the scenes read (`data/*.json`) are committed, so the video renders
without relflow or torch. `compute.py` needs `relflow==0.2.3` with Polars < 2 (it
hashes with `Series.hash(seed_1=...)`), as pinned in `pyproject.toml`; the Text
experiment downloads relflow's default encoder, `google/bert_uncased_L-2_H-128_A-2`
(about 17 MB), from Hugging Face on first use.

## Episodes

| # | Scene | What it establishes | Real data on screen |
|---|---|---|---|
| 1 | `Hook` | relflow as a model factory for structured data; records with lists inside (a ticket, a transfer, a sensor, a customer); what flattening loses; the `rf.Model(...)` code becoming a tree and then a network; the series road map. | The customer model's 12,156,172 parameters (preset `md`). |
| 2 | `ThreeLines` | The happy path: `rf.Model.sm(**schema, batch_size=..., optimizer=...)`, `rf.ArrowDataModule(...)`, `lit.Trainer(...).fit(...)`; what one training step does; field-owned learning during training only; Lightning's four loops; save, load and predict; the two settings the presets leave to you. | Those three lines run on 4,096 orders: 1.5M parameters, 12 s, validation AUC 0.74 then 1.00; `trainer.test` on 2,048 orders; a 5.9 MB artifact; real prediction rows. |
| 3 | `SchemaIsArchitecture` | Addresses (`/orders/line_items/sku`); branch lengths fix every shape; what a leaf and a branch own; where the parameters live; the five presets. | Per-node parameter counts of the real model; the SKU embedder's 1,536 parameters with an empty vocabulary; `rf.presets` XS–XL. |
| 4 | `RaggedToDense` | Arrow's offsets + child arrays; `placement = parent × length + slot`; head vs tail overflow; nested placement; the state grid. | `model.encode(...)` and relflow's own `ragged.descend` on four orders: placements `[0, 1, 4, 12, 13, 14, 15]`, V/N/P states, batch-local SKU ids. |
| 5 | `ValueState` | Five state tokens with learned vectors, added to content; null is a coordinate, padding is not; empty vs null lists; unknown labels. | Proof P038: AUC 1.00 with nulls kept, 0.50 when nulls are filled with 0.0 first, 0.51 with random nulls. |
| 6 | `Numbers` | Streaming standardization with exact moment merging; 13 Fourier frequencies; why a weighted sum of them can make sharp effects; the raw ruler lane; the clamp; decoding in standard deviations. | relflow's `GlobalOnlineNormalizer` over six real batches matching the exact mean and variance; frequencies and the 199-σ clamp read from the embedder. |
| 7 | `Vocabulary` | Online admission; worker-local dictionaries bound before forward; geometric growth; optimizer state carried across a resize; validation and serving never admit labels; `p_unavailable`. | A real run with new merchants arriving: rows 1 → 4 → 8 → 16 → 32 → 64, with old rows and Adam moments identical and new moments exactly 0 at every growth. |
| 8 | `Hashing` | Why identifiers need equality, not identity; salted hash lanes through fixed Fourier features; a new salt every training batch; salt 0 at test/predict. | The real Hash embedder's vectors and similarity matrices in three batches; P021 (AUC 0.9999 vs 0.50 for Category on unseen IDs) and its limit P028 (0.47). |
| 9 | `QuantileEnum` | Quantile: a value's percentile in a t-digest kept as Arrow data, eight Fourier pairs, clipping at the learned range, and when it beats standardizing (P077, P068). Enum: declared labels, fixed rows, strict validation, and why declaring isn't learning (P070). | 4,000 heavy-tailed amounts with 8 outliers: Number's σ inflated to 1,275 with 97% of values inside 0.1 σ, against Quantile's percentiles (20 → 0.32, 33 → 0.51, 100 → 0.86); an Enum trained without ever seeing `gold` (3 rows, 0 exposures) beside a Category that learned two labels; the real error for an undeclared value. |
| 10 | `TimeVectorText` | DateParts: calendar positions as (sin, cos) on circles, nine parts, choosing them (P044, P043). Vector: supplied fixed-width vectors, projected, null as a state, no normalizer. Text: a frozen Hugging Face encoder, mean pooling, a learned projection; what Text doesn't do. | Hour-of-day coordinates from `model.encode` for a whole day (23:30 and 01:30 at distance 0.52, noon at 1.98); the real wrong-length error; frozen BERT similarities of four support messages (0.81 for the two about a missing transfer, 0.67 against the password question). |
| 11 | `InsideABranch` | Coordinate attention; the unrolled 24-position sequence; rotary positions make score depend only on distance; pre-norm layers; MHA/GQA/MQA/None. | Toeplitz scores from relflow's `RotaryEmbedding`; a trained model at AUC 1.000 that falls to 0.90 when quantities are shuffled among each order's items. |
| 12 | `Packing` | Attention cost on padding; gather live tokens, group rows by length, attend without masks, scatter back. | relflow's `pack`/`unpack` on a real batch (72% of pair work would be padding); dense vs packed encoder timed on the M1: 153 → 10 ms (15×), 153 → 22 ms, 152 → 51 ms; outputs equal to ~10⁻⁶. |
| 13 | `Reduction` | Learned-query cross-attention; softmax weights sum to one, so duplicates cannot be counted; the mass and count lanes (zero-initialized); the three reduction choices. | Three real training runs on bags of 1–6 equal amounts: `Mean` answers 4.37 and attention without the lanes 4.23 for 1–10 copies; with the lanes 1.31, 2.59, 3.85, 5.10, 6.30, 7.48, then flattening past the trained range. |
| 14 | `Routing` | The customer model bottom-up: 32 tokens per order → 4 summaries; 64 × (1 + 4) = 320 tokens → 4; 7 root slots → 4; the depth-first plan. | Input widths, outputs and the plan from `model.schema`. |
| 15 | `Decoders` | Heritage paths; target-shaped learned queries nudged by aligned siblings; state + content heads and the loss. | Heritage paths and decoder contexts from the schema. |
| 16 | `Masks` | One mechanism: select, hide, optionally reconstruct; the policies (`True`, a rate, `skip`, `reconstruct`, `query`, branch masks); learned masks vs removal; deterministic SHA-256 sampling; stage behavior; P051 and P053. | One real batch encoded under five policies (visible, `0.25`, `skip=True`, `True`, a branch mask); real selections at 15% for epoch 0 twice, epoch 1 and validation. |
| 17 | `Pretraining` | Self-supervised pretraining with sampled reconstruction; what it learns; fine-tuning with `Model.load`, `update` and `extend`; comparing against training from scratch; habits that keep it honest (P062). | 200 products in 8 hidden groups, 20,000 unlabeled orders, 6 epochs of pretraining: hidden-SKU accuracy 2.9% (chance 0.5%, group-level ceiling ≈ 3.3%) and group structure in the SKU embeddings; fine-tuned vs from-scratch test AUC at 32–1,024 labels, 3 seeds each (64 labels: 0.77 → 0.89; equal by 1,024). |
| 18 | `Processors` | Where pre- and postprocessors run; a Polars preprocessor that turns named accounts into a repeated `parties` branch; the schema of the processed shape; a postprocessor that applies a business threshold; attaching both to training, prediction, batch writing and serving; scope and retention. | A real transfer before and after `prepare`; a model trained through it (99.95% accuracy; watchlisted payees flagged 100%, the same accounts as payers 0%); real postprocessed output rows. |
| 19 | `RealRun` | A model learns which line item makes an order come back; reading its reduction attention; exported embeddings; the prediction envelope. | 4,096 orders, 5 s on CPU, validation AUC 0.96 → 1.00; two of four heads put ≈ 1.0 of their weight on the decisive item's SKU; PCA of the exported root vectors; a real output row. |
| 20 | `Benchmarks` | Training and inference throughput and peak memory for every preset; CPU vs the M1 GPU; the cost of more room in a branch; what to measure on your own workload. | `rf.Model.xs`…`xl` on the getting-started customer schema, batches of 64: 476 → 18 rows/s training, 1,263 → 69 predicting, 0.9 → 4.2 GB peak; the CPU ahead in 5 of 6 comparisons with MPS; orders length 16 → 256 costs 2.7× the time and 2.2× the memory. |
| 21 | `Evidence` | The proof catalog: 78 proofs, 55 passing, 13 limited, 10 partial, 7 missing their own gates in the latest run; highlights and recorded limits; recap. | `proofs/results.yaml` at relflow 5077716: P040, P076, P044, P028, P066, P071. |

## Sound

The narrator is Kokoro-82M `af_heart`, as in the other videos, with new lexicon
entries for relflow, PyTorch, Parquet, YAML, SKU, RoPE, SDPA and friends (checked
with a Whisper listen test). This is the first video in the repository with music:

* **A generated music bed** ([`tools/music.py`](https://github.com/Sir-Teo/explainer-videos/blob/6c22be4/tools/music.py)): additive
  "felt piano" notes with inharmonic partials and two-stage decays, a detuned pad,
  a sine bass and a shaped-noise convolution reverb, on a slow I–vi–IV–V loop in
  D major at 66 bpm. Each episode restarts the phrase with a rolled chord. No
  melody, nothing above a soft register, and a dip around 2.5 kHz so it stays out
  of the voice's presence band.
* **Ducking that anticipates the voice.** The whole narration is known in
  advance, so the bed drops *before* each phrase (fast) and recovers slowly: about
  22 dB below the voice while it speaks, 16 dB below in longer pauses, and 9 dB
  below over the title, the episode cards and the end cards.
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
| `schema` | the customer model (`rf.Model.md`) node by node: kinds, shapes, heritage, decoder context, encoder input widths and outputs, parameters per module; the README order model; `rf.presets` | 1, 3, 14, 15 |
| `happy` | the three lines on 4,096 orders (preset `sm`, 6 epochs): per-step loss, per-epoch validation, `trainer.test`, the saved artifact's size, predictions from the reloaded model, the presets' default batch size and optimizer | 2 |
| `encode` | four orders through `ragged.root`/`ragged.descend` and `model.encode` (head and tail); deterministic mask selection on 12 orders × 16 slots | 4, 16 |
| `number` | the Number embedder's frequencies and clamp; `GlobalOnlineNormalizer` fed six lognormal batches, against exact statistics | 6 |
| `vocab` | 14 training batches with three new merchants each; `Model.on_before_batch_transfer`'s binding wrapped to compare the embedding table and Adam's `exp_avg`/`exp_avg_sq` before and after every bind | 7 |
| `hash` | one observation with seven IDs encoded in three training batches and once for prediction: lanes, vectors, cosine similarities | 8 |
| `quantile_enum` | Number and Quantile fields trained side by side on 4,000 lognormal amounts with 8 outliers ×400: the field statistics, percentiles over a probe grid; an Enum and a Category trained on a column that never says `gold`, and the error for `platinum` | 9 |
| `time_vector_text` | DateParts coordinates for a day and a week from `model.encode`; a Vector's encoding and its wrong-length error; four messages through the default frozen Text encoder (mean pooling) and their cosine similarities | 10 |
| `packing` | `relflow.architecture.packed.pack` on 16 rows; a two-layer `BranchEncoder` (d = 128, 256 × 128 slots) timed dense vs packed at three fill levels, median of 7, with the outputs compared | 12 |
| `counting` | three models (`Mean`; `Attention` with the mass/count lanes deleted; `Attention`) trained 40 epochs on 4,096 bags; predictions on held-out bags of 1–6 and 7–10 items and on 1–10 copies of 1.25 | 13 |
| `masks` | eight orders encoded in train, validate and predict under five quantity policies and a branch-level policy: states, presence, `inferred` | 16 |
| `pretraining` | pretraining on 20,000 unlabeled orders (sampled SKU and price reconstruction), the SKU embedding table, and fine-tuned vs from-scratch test AUC at six label budgets × 3 seeds | 17 |
| `processors` | a transfer model trained through a Polars preprocessor; predictions with the pre- and postprocessor attached; accuracy by kind of transfer | 18 |
| `returns` | the order model trained 4 epochs on 4,096 synthetic orders; test AUC intact and with quantities shuffled within orders; recomputed reduction attention for four test orders; PCA of exported root embeddings; one real output row | 11, 19 |
| `rope` | scores of one query/key pair at every pair of 24 positions through `RotaryEmbedding(16)` | 11 |
| `benchmarks` | every preset on the getting-started customer schema, each configuration in a fresh process: training and prediction time per batch of 64 and peak resident memory, on CPU and on MPS (three presets), and the small preset with `orders` length 16–256 | 20 |
| `proofs` | status and latest full-run outcome of all 78 proofs in `proofs/results.yaml`, and the metrics of the nineteen cited | 5, 8, 9, 10, 13, 16, 17, 21 |

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
* **The mass-lane ablation** (episode 13) sets `mass_projection` and `mass_direction`
  to `None` in every `LearnedQueryCrossAttention`; it is not a relflow option.
  Branch attention is off (`attention=None`) in all three arms, as in proof P041.
* **Proof numbers** are relflow's recorded results (its `results.yaml`), not
  re-runs. "Limited" and "Partial" are relflow's own labels.
* **Pretraining.** 200 SKUs in 8 groups of 25; each order (2–8 items) draws 80% of
  its items from one group; prices are set by group, and group A (the one that
  matters) shares its price level with group B. Returned iff a group-A item has
  quantity ≥ 3 (17% of orders). Pretraining hides 15% of SKUs and prices
  (`Mask(rate=0.15, reconstruct=True)`) for 6 epochs; fine-tuning and the
  from-scratch control both get 300 steps of batch 32 at each budget, and are scored
  on the same 2,000 test orders.
* **Processors.** 400 accounts, 8 on a watchlist; 8% of transfers pay a watched
  account, 8% are sent from one; amounts lognormal in cents. Flagged iff the payee
  is watched or the amount is over 2,500. `account` is a `Category` (persistent
  identity), not a `Hash`.
* **Benchmarks.** Each configuration runs in its own process so peak resident
  memory (`ru_maxrss`) belongs to it alone; about 500 MB of every peak is the
  interpreter, PyTorch, Lightning and relflow. Customers have Poisson(10) orders of
  1–6 items; time is the median of 9 batches after 3 warm-up batches, through
  Lightning for training and `model.predict` for inference. MPS runs synchronize
  before reading the clock. AdamW uses `fused=None` so the same code runs on both
  devices.
* **Machine.** An Apple M1 laptop (16 GB), 4 PyTorch threads; relflow 0.2.3,
  PyTorch 2.14.1 (CPU and MPS), Lightning 2.6.6, PyArrow 26, Polars 1.44.2,
  transformers 5.19. The first cut of this video ran the same experiments on a
  4-vCPU cloud VM; all of `data/` was regenerated on the M1, so a few numbers
  differ from that cut (packing 248 → 14 ms there, 153 → 10 ms here).

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
