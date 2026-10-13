# relflow, Visualized: episodes

The series cut into 21 standalone videos, one per scene, each ending on a card that names the next episode. Upload them in order as a playlist; each entry below has the title, the description and the pages of the [relflow docs](https://relflow.github.io/relflow/) it belongs on.

| # | Title | Length | Docs pages |
|---|---|---|---|
| 1 | Records have shape (`01-records-have-shape.mp4`) | 3:14 | [`index.qmd`](https://relflow.github.io/relflow/index.html) |
| 2 | Three lines to a trained model (`02-three-lines.mp4`) | 2:28 | [`getting-started.qmd`](https://relflow.github.io/relflow/getting-started.html), [`guides/lightning.qmd`](https://relflow.github.io/relflow/guides/lightning.html) |
| 3 | The schema is the architecture (`03-schema.mp4`) | 2:32 | [`core-concepts/model-tree.qmd`](https://relflow.github.io/relflow/core-concepts/model-tree.html) |
| 4 | From records to rectangles (`04-records-to-rectangles.mp4`) | 2:17 | [`core-concepts/binding-data.qmd`](https://relflow.github.io/relflow/core-concepts/binding-data.html), [`data-types/branch.qmd`](https://relflow.github.io/relflow/data-types/branch.html) |
| 5 | Null is not zero (`05-null-is-not-zero.mp4`) | 2:08 | [`core-concepts/data-types.qmd`](https://relflow.github.io/relflow/core-concepts/data-types.html) |
| 6 | How a number becomes a vector (`06-numbers.mp4`) | 2:44 | [`data-types/number.qmd`](https://relflow.github.io/relflow/data-types/number.html) |
| 7 | A vocabulary that grows while it trains (`07-vocabulary.mp4`) | 2:40 | [`core-concepts/vocabulary.qmd`](https://relflow.github.io/relflow/core-concepts/vocabulary.html), [`data-types/category.qmd`](https://relflow.github.io/relflow/data-types/category.html) |
| 8 | Equality without identity (`08-hash.mp4`) | 2:09 | [`data-types/hash.qmd`](https://relflow.github.io/relflow/data-types/hash.html) |
| 9 | Ranks and fixed labels (`09-quantile-and-enum.mp4`) | 3:03 | [`data-types/quantile.qmd`](https://relflow.github.io/relflow/data-types/quantile.html), [`data-types/enum.qmd`](https://relflow.github.io/relflow/data-types/enum.html) |
| 10 | Time, vectors and text (`10-time-vectors-text.mp4`) | 2:26 | [`data-types/dateparts.qmd`](https://relflow.github.io/relflow/data-types/dateparts.html), [`data-types/vector.qmd`](https://relflow.github.io/relflow/data-types/vector.html), [`data-types/text.qmd`](https://relflow.github.io/relflow/data-types/text.html) |
| 11 | Inside a branch (`11-inside-a-branch.mp4`) | 2:55 | [`data-types/branch.qmd`](https://relflow.github.io/relflow/data-types/branch.html) |
| 12 | Don't pay for padding (`12-packing.mp4`) | 2:09 | [`guides/performance.qmd`](https://relflow.github.io/relflow/guides/performance.html) |
| 13 | Why softmax can't count (`13-softmax-cant-count.mp4`) | 3:20 | [`data-types/branch.qmd`](https://relflow.github.io/relflow/data-types/branch.html) |
| 14 | Climbing the tree (`14-climbing-the-tree.mp4`) | 1:43 | [`core-concepts/model-tree.qmd`](https://relflow.github.io/relflow/core-concepts/model-tree.html) |
| 15 | Asking the tree a question (`15-decoders.mp4`) | 1:32 | [`guides/prediction-output.qmd`](https://relflow.github.io/relflow/guides/prediction-output.html) |
| 16 | One mask, many objectives (`16-masks.mp4`) | 3:15 | [`core-concepts/dynamic-masking.qmd`](https://relflow.github.io/relflow/core-concepts/dynamic-masking.html) |
| 17 | Learning without labels (`17-pretraining.mp4`) | 2:58 | [`guides/pretraining.qmd`](https://relflow.github.io/relflow/guides/pretraining.html) |
| 18 | Before and after the model (`18-processors.mp4`) | 2:31 | [`guides/preprocessors.qmd`](https://relflow.github.io/relflow/guides/preprocessors.html), [`guides/postprocessors.qmd`](https://relflow.github.io/relflow/guides/postprocessors.html) |
| 19 | Watching it learn (`19-watching-it-learn.mp4`) | 2:36 | [`guides/evaluation.qmd`](https://relflow.github.io/relflow/guides/evaluation.html), [`core-concepts/embeddings.qmd`](https://relflow.github.io/relflow/core-concepts/embeddings.html) |
| 20 | What it costs: time and memory (`20-benchmarks.mp4`) | 2:40 | [`guides/performance.qmd`](https://relflow.github.io/relflow/guides/performance.html) |
| 21 | What the proofs show, and what they don't (`21-proofs.mp4`) | 2:21 | [`proofs.qmd`](https://relflow.github.io/relflow/proofs.html) |

## 1. Records have shape

**YouTube title:** Records have shape | relflow, Visualized #1

Most data worth learning from has shape: lists inside records. relflow is a model factory for structured data: describe the shape of one record and it builds the PyTorch model that reads it. The series road map.

## 2. Three lines to a trained model

**YouTube title:** Three lines to a trained model | relflow, Visualized #2

The happy path: load a schema into a model, build a data module, fit. A real run from those three lines, what each training step does, Lightning's four loops, and saving, loading and predicting.

## 3. The schema is the architecture

**YouTube title:** The schema is the architecture | relflow, Visualized #3

Addresses, fixed shapes, what each leaf and branch owns, where the parameters live, and the five size presets.

## 4. From records to rectangles

**YouTube title:** From records to rectangles | relflow, Visualized #4

How nested Arrow records become dense tensors: offsets and child arrays, one placement rule, head and tail overflow, nested branches, and a state beside every value.

## 5. Null is not zero

**YouTube title:** Null is not zero | relflow, Visualized #5

Five value states (valued, null, padded, masked, other) with learned vectors, and a proof of why filling nulls with zero loses information.

## 6. How a number becomes a vector

**YouTube title:** How a number becomes a vector | relflow, Visualized #6

Streaming standardization with exactly merged moments, 13 Fourier frequencies, a raw ruler lane for extrapolation, and decoding in standard deviations.

## 7. A vocabulary that grows while it trains

**YouTube title:** A vocabulary that grows while it trains | relflow, Visualized #7

Category vocabularies that grow while the model trains: worker-local dictionaries, geometric growth, and optimizer state carried across every resize.

## 8. Equality without identity

**YouTube title:** Equality without identity | relflow, Visualized #8

Identifiers that need equality, not identity: salted hash lanes, a new salt every training batch, and the proofs that show where this works and where it doesn't.

## 9. Ranks and fixed labels

**YouTube title:** Ranks and fixed labels | relflow, Visualized #9

Quantile: a number as its percentile in a learned t-digest, and when that beats standardizing. Enum: labels declared up front, fixed rows, and strict validation.

## 10. Time, vectors and text

**YouTube title:** Time, vectors and text | relflow, Visualized #10

DateParts puts calendar positions on circles; Vector projects supplied embeddings; Text runs strings through a frozen Hugging Face encoder and learns how to use the result.

## 11. Inside a branch

**YouTube title:** Inside a branch | relflow, Visualized #11

Coordinate attention among an item's fields, the unrolled sequence, rotary positions, pre-norm layers, and a real test that the model reads SKUs and quantities as pairs.

## 12. Don't pay for padding

**YouTube title:** Don't pay for padding | relflow, Visualized #12

Padding costs the answers nothing, but it costs the hardware. Packing gathers live tokens, groups rows by length and attends without masks: timed for real.

## 13. Why softmax can't count

**YouTube title:** Why softmax can't count | relflow, Visualized #13

Learned-query cross-attention summarizes a branch, but softmax weights sum to one, so it can't count. relflow's additive mass and count lanes, tested in three real training runs.

## 14. Climbing the tree

**YouTube title:** Climbing the tree | relflow, Visualized #14

The customer model bottom up: the shapes at every hand-off and the depth-first execution plan.

## 15. Asking the tree a question

**YouTube title:** Asking the tree a question | relflow, Visualized #15

Decoders read the tree back down: heritage paths, target-shaped learned queries, and a head and loss for every datatype.

## 16. One mask, many objectives

**YouTube title:** One mask, many objectives | relflow, Visualized #16

One mechanism for targets, dropout, self-supervision and ablation: learned masks versus skipping, branch-level masks, deterministic SHA-256 sampling, and what each stage does.

## 17. Learning without labels

**YouTube title:** Learning without labels | relflow, Visualized #17

Learning without labels: pretrain by reconstructing hidden values, fine-tune with a few labels, and a real comparison against training from scratch.

## 18. Before and after the model

**YouTube title:** Before and after the model | relflow, Visualized #18

Polars preprocessors reshape records before the schema applies; postprocessors turn predictions into application output. A real model trained through both.

## 19. Watching it learn

**YouTube title:** Watching it learn | relflow, Visualized #19

A real training run, its reduction attention read out head by head, exported embeddings, and the prediction envelope.

## 20. What it costs: time and memory

**YouTube title:** What it costs: time and memory | relflow, Visualized #20

Training and inference speed and peak memory for every preset, CPU against the M1's GPU, and what longer contexts cost: measured on a laptop.

## 21. What the proofs show, and what they don't

**YouTube title:** What the proofs show, and what they don't | relflow, Visualized #21

relflow's proof catalog: what passes, what is limited, what is still missing, and a recap of the series.
