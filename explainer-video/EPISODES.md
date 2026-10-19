# relflow, Visualized: episodes

The series cut into 25 standalone videos, one per scene, each ending on a card that names the next episode. Upload them in order as a playlist; each entry below has the title, the description and the pages of the [relflow docs](https://relflow.github.io/docs/) it belongs on.

| # | Title | Length | Docs pages |
|---|---|---|---|
| 1 | Records have shape (`01-records-have-shape.mp4`) | 3:16 | [`index.qmd`](https://relflow.github.io/docs/index.html) |
| 2 | Three lines to a trained model (`02-three-lines.mp4`) | 2:14 | [`getting-started.qmd`](https://relflow.github.io/docs/getting-started.html), [`tutorials/transaction-history.qmd`](https://relflow.github.io/docs/tutorials/transaction-history.html), [`guides/lightning.qmd`](https://relflow.github.io/docs/guides/lightning.html) |
| 3 | The schema is the architecture (`03-schema.mp4`) | 2:41 | [`core-concepts/model-tree.qmd`](https://relflow.github.io/docs/core-concepts/model-tree.html) |
| 4 | From records to rectangles (`04-records-to-rectangles.mp4`) | 2:22 | [`core-concepts/binding-data.qmd`](https://relflow.github.io/docs/core-concepts/binding-data.html), [`data-types/branch.qmd`](https://relflow.github.io/docs/data-types/branch.html) |
| 5 | Follow the source, keep the schema (`05-query-paths.mp4`) | 1:07 | [`core-concepts/querypaths.qmd`](https://relflow.github.io/docs/core-concepts/querypaths.html) |
| 6 | Null is not zero (`06-null-is-not-zero.mp4`) | 2:08 | [`core-concepts/data-types.qmd`](https://relflow.github.io/docs/core-concepts/data-types.html) |
| 7 | How a number becomes a vector (`07-numbers.mp4`) | 2:44 | [`data-types/number.qmd`](https://relflow.github.io/docs/data-types/number.html) |
| 8 | Teach tolerance, keep the target (`08-jitter.mp4`) | 1:11 | [`core-concepts/jitter.qmd`](https://relflow.github.io/docs/core-concepts/jitter.html) |
| 9 | A vocabulary that grows while it trains (`09-vocabulary.mp4`) | 2:40 | [`core-concepts/vocabulary.qmd`](https://relflow.github.io/docs/core-concepts/vocabulary.html), [`data-types/category.qmd`](https://relflow.github.io/docs/data-types/category.html) |
| 10 | Equality without identity (`10-hash.mp4`) | 2:09 | [`data-types/hash.qmd`](https://relflow.github.io/docs/data-types/hash.html) |
| 11 | Ranks and fixed labels (`11-quantile-and-enum.mp4`) | 3:03 | [`data-types/quantile.qmd`](https://relflow.github.io/docs/data-types/quantile.html), [`data-types/enum.qmd`](https://relflow.github.io/docs/data-types/enum.html) |
| 12 | Membership and learned groups (`12-sets-and-clusters.mp4`) | 1:24 | [`data-types/set.qmd`](https://relflow.github.io/docs/data-types/set.html), [`data-types/cluster.qmd`](https://relflow.github.io/docs/data-types/cluster.html) |
| 13 | Time, vectors and text (`13-time-vectors-text.mp4`) | 2:26 | [`data-types/dateparts.qmd`](https://relflow.github.io/docs/data-types/dateparts.html), [`data-types/vector.qmd`](https://relflow.github.io/docs/data-types/vector.html), [`data-types/text.qmd`](https://relflow.github.io/docs/data-types/text.html) |
| 14 | Inside a branch (`14-inside-a-branch.mp4`) | 3:05 | [`data-types/branch.qmd`](https://relflow.github.io/docs/data-types/branch.html) |
| 15 | Don't pay for padding (`15-packing.mp4`) | 2:10 | [`guides/performance.qmd`](https://relflow.github.io/docs/guides/performance.html) |
| 16 | Why softmax can't count (`16-softmax-cant-count.mp4`) | 3:20 | [`data-types/branch.qmd`](https://relflow.github.io/docs/data-types/branch.html) |
| 17 | Climbing the tree (`17-climbing-the-tree.mp4`) | 1:49 | [`core-concepts/model-tree.qmd`](https://relflow.github.io/docs/core-concepts/model-tree.html) |
| 18 | Asking the tree a question (`18-decoders.mp4`) | 1:40 | [`guides/prediction-output.qmd`](https://relflow.github.io/docs/guides/prediction-output.html) |
| 19 | A prediction keeps its place (`19-prediction-output.mp4`) | 1:51 | [`guides/prediction-output.qmd`](https://relflow.github.io/docs/guides/prediction-output.html), [`core-concepts/embeddings.qmd`](https://relflow.github.io/docs/core-concepts/embeddings.html) |
| 20 | One mask, many objectives (`20-masks.mp4`) | 3:15 | [`core-concepts/dynamic-masking.qmd`](https://relflow.github.io/docs/core-concepts/dynamic-masking.html) |
| 21 | Learning without labels (`21-pretraining.mp4`) | 2:58 | [`guides/pretraining.qmd`](https://relflow.github.io/docs/guides/pretraining.html) |
| 22 | Before and after the model (`22-processors.mp4`) | 2:31 | [`guides/preprocessors.qmd`](https://relflow.github.io/docs/guides/preprocessors.html), [`guides/postprocessors.qmd`](https://relflow.github.io/docs/guides/postprocessors.html) |
| 23 | Watching it learn (`23-watching-it-learn.mp4`) | 3:37 | [`guides/evaluation.qmd`](https://relflow.github.io/docs/guides/evaluation.html), [`core-concepts/embeddings.qmd`](https://relflow.github.io/docs/core-concepts/embeddings.html) |
| 24 | What it costs: time and memory (`24-benchmarks.mp4`) | 2:41 | [`guides/performance.qmd`](https://relflow.github.io/docs/guides/performance.html) |
| 25 | What the proofs show, and what they don't (`25-proofs.mp4`) | 2:25 | [`proofs.qmd`](https://relflow.github.io/docs/proofs.html) |

## 1. Records have shape

**YouTube title:** Records have shape | relflow, Visualized #1

Most data worth learning from has shape: lists inside records. relflow is a model factory for structured data: describe the shape of one record and it builds the PyTorch model that reads it. The series road map.

## 2. Three lines to a trained model

**YouTube title:** Three lines to a trained model | relflow, Visualized #2

One payment, a hidden dispute target, transaction context at a common review cutoff, model.fit, and selected Arrow output planes. Current API smoke-tested; no learning-quality claim.

## 3. The schema is the architecture

**YouTube title:** The schema is the architecture | relflow, Visualized #3

Addresses, batch-dependent execution widths, finite caps, what each leaf and branch owns, where the parameters live, and the five size presets.

## 4. From records to rectangles

**YouTube title:** From records to rectangles | relflow, Visualized #4

How nested Arrow records become dense tensors: offsets and child arrays, one placement rule, head and tail overflow, nested branches, and a state beside every value.

## 5. Follow the source, keep the schema

**YouTube title:** Follow the source, keep the schema | relflow, Visualized #5

Animate parent-relative query routing, retained records, and the boundary between structural selection and preprocessing.

## 6. Null is not zero

**YouTube title:** Null is not zero | relflow, Visualized #6

Five value states (valued, null, padded, masked, other) with learned vectors, and a proof of why filling nulls with zero loses information.

## 7. How a number becomes a vector

**YouTube title:** How a number becomes a vector | relflow, Visualized #7

Streaming standardization with exactly merged moments, 13 Fourier frequencies, a raw ruler lane for extrapolation, and decoding in standard deviations.

## 8. Teach tolerance, keep the target

**YouTube title:** Teach tolerance, keep the target | relflow, Visualized #8

Actual triangular jitter samples: visible training inputs move while pristine targets, evaluation and normalization statistics stay fixed.

## 9. A vocabulary that grows while it trains

**YouTube title:** A vocabulary that grows while it trains | relflow, Visualized #9

Category vocabularies that grow while the model trains: worker-local dictionaries, geometric growth, and optimizer state carried across every resize.

## 10. Equality without identity

**YouTube title:** Equality without identity | relflow, Visualized #10

Identifiers that need equality, not identity: salted hash lanes, a new salt every training batch, and the proofs that show where this works and where it doesn't.

## 11. Ranks and fixed labels

**YouTube title:** Ranks and fixed labels | relflow, Visualized #11

Quantile: a number as its percentile in a learned t-digest, and when that beats standardizing. Enum: labels declared up front, fixed rows, and strict validation.

## 12. Membership and learned groups

**YouTube title:** Membership and learned groups | relflow, Visualized #12

Set membership, duplicates, unavailable labels, and the reconstruction objective required to train Cluster assignments; schematic.

## 13. Time, vectors and text

**YouTube title:** Time, vectors and text | relflow, Visualized #13

DateParts puts calendar positions on circles; Vector projects supplied embeddings; Text runs strings through a frozen Hugging Face encoder and learns how to use the result.

## 14. Inside a branch

**YouTube title:** Inside a branch | relflow, Visualized #14

Coordinate attention among an item's fields, the unrolled sequence, rotary positions, pre-norm layers, and a real test that the model reads SKUs and quantities as pairs.

## 15. Don't pay for padding

**YouTube title:** Don't pay for padding | relflow, Visualized #15

Padding costs the answers nothing, but it costs the hardware. Packing gathers live tokens, groups rows by length and attends without masks: timed for real.

## 16. Why softmax can't count

**YouTube title:** Why softmax can't count | relflow, Visualized #16

Learned-query cross-attention summarizes a branch, but softmax weights sum to one, so it can't count. relflow's additive mass and count lanes, tested in three real training runs.

## 17. Climbing the tree

**YouTube title:** Climbing the tree | relflow, Visualized #17

The customer model bottom up: the shapes at every hand-off and the depth-first execution plan.

## 18. Asking the tree a question

**YouTube title:** Asking the tree a question | relflow, Visualized #18

Decoders read the tree back down: heritage paths, target-shaped learned queries, and a head and loss for every datatype.

## 19. A prediction keeps its place

**YouTube title:** A prediction keeps its place | relflow, Visualized #19

Format 2: nested records preserve null lists, empty lists and structural holes; embeddings carry explicit paths and parent ordinals.

## 20. One mask, many objectives

**YouTube title:** One mask, many objectives | relflow, Visualized #20

One mechanism for targets, dropout, self-supervision and ablation: learned masks versus skipping, branch-level masks, deterministic SHA-256 sampling, and what each stage does.

## 21. Learning without labels

**YouTube title:** Learning without labels | relflow, Visualized #21

Learning without labels: pretrain by reconstructing hidden values, fine-tune with a few labels, and a real comparison against training from scratch.

## 22. Before and after the model

**YouTube title:** Before and after the model | relflow, Visualized #22

Polars preprocessors reshape records before the schema applies; postprocessors turn predictions into application output. A real model trained through both.

## 23. Watching it learn

**YouTube title:** Watching it learn | relflow, Visualized #23

A freshly recorded relflow 0.3.2 training run: four attention heads, learned query weights, paired predictions and held-out loss on one checkpoint clock. Follow real gradients and AdamW changes, watch exported embeddings move in a fixed PCA view, then freeze the weights and change quantity or position.

## 24. What it costs: time and memory

**YouTube title:** What it costs: time and memory | relflow, Visualized #24

Archived relflow 0.2.3 training and inference speed and peak memory for every preset, CPU against the M1's GPU, and what longer contexts cost: measured on a laptop.

## 25. What the proofs show, and what they don't

**YouTube title:** What the proofs show, and what they don't | relflow, Visualized #25

Archived relflow 0.2.3 proof catalog: what passes, what is limited, what is still missing, and a recap of the series.
