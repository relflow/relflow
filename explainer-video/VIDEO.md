# Production notes: 0.3 refresh

The 25-episode cut follows `relflow/docs` main at
`b46f1b0d6e5fc511993bc56fcdba77e56441d926` (2026-10-19 checkout), whose API contract
is pinned to 0.3.1. Changed behavior is checked against package main at
`2bb35adb87b6fefe615864b170c196e7509f81fe`, release 0.3.2. The video branch includes
that package revision through a merge. No docs or runtime source is modified
for the video.

## Visual grammar

Validated framework checkout: `Sir-Teo/explainer-videos` at `b52316d4b4112fcea723453f56d37e5197adbf96`, Manim 0.22.0.

Use the shared explainer toolkit's dark background, semantic colors,
bookmark-driven voiceover, cached vector rendering, and normalized narration.
The Relflow palette remains local: green root, blue branch, neutral fields,
amber targets; cyan numbers, orange categories, violet hashes; distinct null,
padded and masked states. Keep object identity while transforming a record,
and reveal a mechanism at the word that explains it. Schematics must say so;
measured illustrations state their version.

| Mechanism | Visual implementation | Evidence |
|---|---|---|
| Supervised learning | payment cards; hide the dispute label while amount and merchant flow into the model | current README, getting-started, `compute_current.py` fit smoke run |
| Review cutoff | transaction timeline separates known inputs from the next 30 days of outcome labels; every retained label is hidden | current transaction-history tutorial |
| Automatic lengths | morph a 3-row tensor grid through widths 2 → 5 → 1 around a stationary parameter block | three actual `model.encode` shapes; parameter identities asserted unchanged |
| Query paths | colored routes from payload to a sliced transaction list to gross amounts; whole rows remain aligned | actual selection `[56.2, 84.5]` from `transactions[-2:]` |
| Jitter | a dot fans into a stacked sample cloud while the pristine target stays fixed; evaluation lanes stay still | 512 seeded `Jitter.apply` samples, displaying 128; raw add 0.1 / multiply 0.02 |
| Set | reorder tags and remove a duplicate, leaving the same two memberships | schematic; current Set source and docs |
| Cluster | label-to-group routes paired with a reconstruction objective | schematic; never presented as learned groups |
| Prediction geometry | null → null, empty → empty, mixed list → predictions with the same hole | actual untrained nested output; assertions on validity and list counts |
| Representations | highlight ordinal 2, then connect it to `parents: [2]`; separate three output planes | actual representation records with paths, ordinals, intrinsic index, normalized vectors |
| Direct batching | preprocess once → chunk before encode → combine → postprocess once | runtime prediction contract and tests |
| Branch blocks | RMSNorm and node-local `ffnx` (default 8), tanh GELU, residual connections | `architecture/blocks.py`, `encoder.py`, `presets/base.py` |
| Learning microscope | same order, four attention rows, fixed signed parameter slice, paired prediction rails and held-out loss share one checkpoint clock | `compute_learning.py`; 11 current 0.3.2 snapshots |
| Backpropagation | forward value pulse, reverse gradient arrows, same coordinate highlighted in before/gradient/actual AdamW-change grids | schematic route; actual update 16 tensors, each panel's stated scale |
| Representation learning | 512 persistent dots move in a common final PCA basis, center and axis ranges | actual exported 64-d root vectors at every checkpoint |
| Controlled input changes | freeze parameters; change only S11 quantity 4→1, restore it, then move that same item to the first slot | three fixed label-free probes; measured final probabilities |

The Arrow-buffer, Fourier, rotary, token-packing and mass/count explanations
retain their empirical foundations, with added identity and flow cues. Their
learning experiments remain tied to the original 0.2.3 data. Episode 23's
attention and embedding footage uses the synchronized 0.3.2 learning microscope.
Capped examples that reach their limits
remain valid worked shapes; they no longer imply every batch executes at the
cap. The output scene replaces slash-keyed examples with format-2 records.
The processor example reads `struct.field("flagged")`; fine-tuning configures
the next run through `fit`.

## Series-wide storytelling pass

All 24 chapters outside the already polished training episode were revised.
Use the same semantic colors across a hand-off. Move an existing item or its
copy when identity matters; avoid replacing it with an unrelated glyph. Packets
show information access and direction, without asserting neural latency or
physical travel. Numerical cells and captions carry their stated scales.

| Episode | Visual story |
|---|---|
| 1 | Leaf messages aggregate by depth; root context reaches the target decoders. |
| 2 | Visible payment fields enter the model; a forward pass reaches the loss and AdamW updates the tree. |
| 3 | Address traversal follows the actual schema edges; the preset introduction stays visible during its setup. |
| 4 | The same source token reaches placement 13 before all retained values scatter. |
| 5 | Query selection follows source routes while whole records preserve alignment. |
| 6 | Content and valued-state vectors actually add to the illustrated result, using a shared scale. |
| 7 | Moving the numeric probe changes its analytic Fourier fingerprint; the scalar lane remains explicit. |
| 8 | Jitter moves a point on the training rail while evaluation rails remain fixed. |
| 9 | A real trained SKU row is looked up twice; resized weight rows remain distinct from zero-initialized Adam moments. |
| 10 | One identifier becomes salted lanes, then a deterministic vector. |
| 11 | A percentile probe changes its features; observed labels feed Enum and Category differently. |
| 12 | A duplicate arrives at an existing membership without increasing it; Cluster's reconstruction objective receives the group signal. |
| 13 | A real untrained Vector token replaces a decorative strip; frozen Text features feed the learned projection. |
| 14 | A measured residual block keeps the old vector on its shortcut; identical SKU lookups become distinct item-context vectors. |
| 15 | One cyan token keeps row 3, position 1 through gather, grouping and scatter. |
| 16 | Real scaled query/key scores become softmax weights, then gather value vectors; equal averages still lose count. |
| 17 | Child summaries become parent fields; only present root inputs enter the final reduction. |
| 18 | Queries receive their own siblings before reading heritage; shared parameters can produce different reads. |
| 19 | Nulls, empty lists, holes and ancestor ordinals retain their identity through output writing. |
| 20 | A pristine target takes a separate route to loss; copied batch grids show exactly which tokens become masked or absent. |
| 21 | Full-vector cosine neighborhoods reveal overlap; signed group means exclude self-comparisons. |
| 22 | Actual payer/payee identities move to their corresponding party records; the amount changes units. |
| 24 | Throughput bars grow from their common baselines while recorded values and memory remain legible. |
| 25 | One batch visibly changes representation through records, embedders, encoders, summaries and decoders. |

The reusable vector instruments live in `story.py`. The attention, embedding
lookup, residual and contextualization views consume actual 0.3.2 tensors.
Architectural routes, packet motion, the mask-learning diagram and the closing
batch glyph are schematic. No random cells are presented as trained weights.

## Measurement boundaries

The old 17 data files are unchanged, including all learning scores and timings.
`data/provenance.json` records their SHA-256 hashes and original version. The
78-entry proof overview is explicitly an archived snapshot, not the latest
proof run. Historical proof status is empirical evidence under the recorded
conditions, not a mathematical guarantee or a 0.3.2 regression check.

`data/schema_current.json` replaces the old schema census in the opening, architecture, routing and decoder scenes. It records 21,340,172 parameters for the current medium customer model, with component totals checked against the whole model.

`data/current.json` records current runtime and docs commits, package/dependency
versions, seed and machine. Geometry and query traces require no learning.
Prediction values are from an untrained model and are not used to imply useful
probabilities. The fit check uses one training and one validation batch; it
makes no accuracy or throughput claim. No pretrained text model is downloaded.

`data/learning_current.json` is a separate seeded 0.3.2 learning experiment,
recorded by `compute_learning.py` using public `model.fit(...)`, CPU float32,
batch size 128, learning rate 0.002, weight decay 0.01, eight epochs. All 4,096
training examples and 512 test examples are synthetic. No test labels or probe
labels enter the inputs or optimizer. Test data is observed at checkpoints,
without tuning or early stopping. The final test accuracy is 100% on this
small, deterministic task; no general performance or timing claim follows.

Each of three probes has six live items, so the reducer reads 18 field-major
tokens. Its actual projected/rotated queries and keys reconstruct all four
softmax heads with the actual padding mask. Attention uses a fixed 0–1 scale.
The unselected top-left 8×8 Q-projection slice uses a common signed scale over
the whole training movie. Update 16's gradient and actual parameter delta use
separately stated scales; neither is inferred from the other. The highlighted
coordinate is the largest absolute gradient within the shown slice (zero-based
indices). The backpropagation routing is schematic. Attention weights are
read operations, not a causal decomposition of a prediction.

All representation checkpoints use the final test embeddings' PCA basis and
center, and a single range spanning all checkpoints. Labels only color the
dots; PCA uses no labels. Transitions interpolate measurements. The quantity
intervention changes only S11's quantity, while the position intervention moves
its complete item. These test the model's input sensitivity, not real-world
causality. Source inspiration: AI Built part 3's forward/backward graph, part 8's
fixed examples during training, and part 10's persistent embedding trajectories.
Numeric readouts switch directly to the destination checkpoint during each
transition; glyphs never morph into unreadable intermediate numbers.
The recorder was run twice: all 256 losses, 11 snapshots, three probes and the
recorded gradient/update tensors matched exactly on this environment.

`data/flow_current.json` records three additional eval-only forward passes after
that same fit. Hooks capture raw SKU lookup vectors, local coordinate outputs,
sequence attention and FFN deltas, branch summaries and root representations.
All heads' actual rotated Q/K projections, V projections, softmax weights and
weighted contexts are retained for the coordinate, sequence and reduction blocks.
The source checker independently reconstructs softmax and weighted contexts, and
verifies each displayed residual addition. The two quantity probes have identical
raw SKU lookup vectors and different contextualized SKU vectors.
The contextual difference readout is the Euclidean norm over all 64 coordinates;
the cropped strips display 16 coordinates on one shared scale.

The Vector fixture has its own seed (73), untrained parameters and explicit
0.3.2 label. Its projected features plus state equal its emitted token; an actual
encoder call records valued/null states and rejects an incorrect input width.

Pretraining geometry is derived from the unchanged archived embedding rows.
The neighborhood view places other SKUs by full-vector cosine to the first
vocabulary label; categorical rows and colors are display organization. The four
nearest labels span several groups. Group averages exclude the diagonal of
each within-group pair matrix, removing trivial self-similarity. A common signed
scale shows positive and negative cosine. These views show useful aggregate
structure and overlapping individual neighborhoods, without asserting perfect
group recovery or inventing an unrecorded training trajectory.

## Build and QA

`render.py` delegates scene rendering, narration metadata, audio-length checks,
normalization and subtitle splitting to the current shared toolkit. It supports
independent selections, full cuts, and per-episode end cards. Silent layout QA
and narrated output use separate media directories. Each output has embedded
subtitles and chapters, plus standalone SRT and chapter text.

Run `compute_current.py` and `compute_learning.py` with the Relflow environment. Run `check.py` with the
framework environment to check the manifest, imports, docs mappings, narration
bookmarks, archived-data hashes and current-trace invariants. Generate contact
sheets with the shared `tools/contact_sheet.py`, inspect dense frames, and
re-render corrections. `SCRIPT.md` and `EPISODES.md` are regenerated from source;
durations remain unknown until the refreshed narrated episodes exist.

## Archived 0.2.3 measurement methods

These methods and machine details describe the preserved historical measurements.
The new API traces and parameter census use their own 0.3.2 provenance above.
Reproduction requires the archived environment and sources; `compute.py` enforces
its package version separately from `compute_current.py`.

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
* **The mass-lane ablation** (the reduction episode) sets `mass_projection` and `mass_direction`
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
  4-vCPU cloud VM; the original 17 experiment files were regenerated on the M1, so a few numbers
  differ from that cut (packing 248 → 14 ms there, 153 → 10 ms here).

## Validation of this refresh

The full narrated cut is 62:03.905 at 1920×1080, with audio, subtitles and 25
chapters. All 25 standalone episodes have audio, subtitles and two chapters
(content and end card); FFprobe reports no container errors. Contact sheets
were inspected for the full cut and the revised scenes. The current API recorder
and source/docs/provenance checker pass, as do 21 targeted runtime tests for
automatic lengths and record-shaped prediction output. All 17 historical data
files retain their original SHA-256 hashes.

The series-wide storytelling pass re-rendered all 24 revised content scenes at
1080p30 and retained the previously polished training scene. Dense mechanism
frames and a fresh 25-chapter contact sheet were inspected. The additional
tensor checks reconstruct every captured head's softmax and value mixture and
verify the measured residual additions and Vector token composition.

Static end cards are captured at the requested output resolution before fading,
and each process gets its own SVG logo cache copy. The build sets MP4 movie
timescale 1000 to preserve QuickTime chapter tracks with FFmpeg 9. A machine-readable
media check is saved with generated output as `renders/validation.json`.

## Sources

- [Current docs](https://github.com/relflow/docs/tree/b46f1b0d6e5fc511993bc56fcdba77e56441d926): getting started, model tree, query paths, jitter, Set, Cluster, prediction output, performance and training.
- [Runtime](https://github.com/relflow/relflow/tree/2bb35adb87b6fefe615864b170c196e7509f81fe): architecture, ragged geometry, field extensions, prediction format and loop methods.
- Archived learning experiments: runtime `5077716`, 0.2.3; original `compute.py`, unchanged data files and the recorded proof snapshot.
- Manim Community and the explainer-videos framework; Kokoro narration. The bundled logo and Libertinus Serif wordmark retain their original licenses.
- Mechanism references retained from the original series: Attention Is All You Need; RoFormer; Fourier Features Let Networks Learn High Frequency Functions; Chan/Golub/LeVeque moment merging; Adam/AdamW; Set Transformer and Deep Sets; Apache Arrow columnar format; EBU R128 loudness.
