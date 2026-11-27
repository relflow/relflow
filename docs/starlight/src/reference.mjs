// @ts-check

/** @typedef {{title: string; text: string}} Note */

/** Authored reference explanations. Matching and annotation mechanics live in code/config-notes.mjs. */
export const reference = {
  representations: /** @type {Record<string, string>} */ ({
    Model: 'Typed fields and related-record context are combined to predict the declared targets.',
    Branch:
      'Each record is encoded from its typed fields. The collection provides context to its parent while preserving record boundaries.',
    Number: 'Standardized raw and signed-log values preserve numeric magnitude for the model.',
    Boolean:
      'Fixed true/false value codes and a learned value-state representation describe this fact.',
    Category: 'A learned vocabulary looks up a representation for each known label.',
    Enum: 'A learned representation comes from the labels explicitly declared in this schema.',
    Set: 'Representations of the supplied labels are combined without imposing an order.',
    Hash: 'Batch-salted fingerprints preserve identifier matches without a persistent learned ID vocabulary.',
    Period: 'Selected calendar parts become cyclical coordinates, then project into model space.',
    Text: 'A frozen pretrained text encoder supplies vectors; a task projection learns how to use them.',
    Image:
      'A frozen pretrained vision encoder supplies CLS vectors; a task projection learns how to use them.',
    Audio:
      'A frozen pretrained audio encoder supplies recording vectors; a task projection learns how to use them.',
  }),
  types: /** @type {Record<string, Note>} */ ({
    Model: {
      title: 'Model schema',
      text: 'Declare the fields, related records, and targets of a model. Defining a class allocates no network; instantiate it to create a trainable model.',
    },
    Branch: {
      title: 'Related records',
      text: 'Declare the fields in each record of a collection. Place a branch inside a model or another branch to retain that structure.',
    },
    Boolean: {
      title: 'Boolean outcome',
      text: 'Represent a true or false value. As a prediction target, this field returns the probability of true.',
    },
    Number: {
      title: 'Numeric magnitude',
      text: 'Represent a scalar measurement in its original units. relflow learns from raw and signed-log numeric representations and maintains normalization statistics.',
    },
    Quantile: {
      title: 'Numeric rank',
      text: 'Represent a number by its percentile in a learned training distribution. Choose Number when absolute distances or extrapolation matter.',
    },
    Ordinal: {
      title: 'Ordered labels and ranks',
      text: 'Represent explicitly ordered labels or integer ranks. Label order defines numeric rank; predictions return a declared label or a native integer, without a class-probability distribution.',
    },
    Category: {
      title: 'Learned labels',
      text: 'Represent one label using a vocabulary learned during training. Previously unseen labels use an unavailable representation until admitted during training.',
    },
    Enum: {
      title: 'Declared enum labels',
      text: 'Declare the allowed labels as positional arguments, such as rf.Enum("low", "normal", "high"). Other non-null labels are rejected.',
    },
    Set: {
      title: 'Unordered label membership',
      text: 'Represent an unordered collection of labels using a learned vocabulary. Use a Branch when each item has its own fields.',
    },
    Hash: {
      title: 'Identifier relationships',
      text: 'Represent identifiers without a learned vocabulary. Equal IDs preserve a matching representation across fields within an encoded batch.',
    },
    Period: {
      title: 'Calendar context',
      text: 'Encode selected calendar parts such as day_of_week or hour_of_day. Use Duration to derive elapsed time from timestamps, or Number for an already computed duration.',
    },
    Duration: {
      title: 'Elapsed time',
      text: 'Derive signed end minus start from explicit timestamp sources. Choose a fixed unit with granularity; the resulting value uses Number’s representations and numeric output. No wall clock is read.',
    },
    Text: {
      title: 'Text meaning',
      text: 'Represent strings with a frozen pretrained text encoder, then learn how those representations contribute to your task. This field does not generate text.',
    },
    Image: {
      title: 'Visual media reference',
      text: 'Decode local image paths or object-store URIs and encode them with a frozen vision backbone. Cached CLS vectors project into model space without generating images.',
    },
    Audio: {
      title: 'Audio recording reference',
      text: 'Decode and resample local audio paths or object-store URIs, pool overlapping windows with a frozen audio encoder, and project the cached vector into model space.',
    },
    Vector: {
      title: 'Fixed-width numeric values',
      text: 'Represent an existing numeric vector with exactly n_dim coordinates. Values are stored as float32 without fitting a normalizer.',
    },
    Group: {
      title: 'Shared identity representations',
      text: 'Map known identities to a fixed number of shared representations, learned through downstream objectives while visible as context. Use assignments to inspect identity routing; group IDs have no predefined business meaning.',
    },
    ArrowDataModule: {
      title: 'Training data',
      text: 'Read Arrow-compatible data sources for training and evaluation, preserving nested records. Keep training and validation sources separate.',
    },
    Writer: {
      title: 'Streaming prediction output',
      text: 'Write batch predictions to Parquet or Arrow IPC files instead of collecting the complete result in memory.',
    },
    reconstruct: {
      title: 'Reconstruction selection',
      text: 'Choose which observed values become pretraining answers. A rate samples answers, a query reads a Boolean selector, and a span groups consecutive branch rows.',
    },
    Jitter: {
      title: 'Training augmentation',
      text: 'Perturb eligible continuous inputs during training while preserving the original targets. Each field applies jitter in its own representation.',
    },
  }),
  methods: /** @type {Record<string, Note>} */ ({
    'Set.vocabulary': {
      title: 'Known set labels',
      text: 'Return an immutable tuple of unique labels in this field’s learned vocabulary. The snapshot does not change when later training admits new labels.',
    },
    'Category.vocabulary': {
      title: 'Known category labels',
      text: 'Return an immutable tuple of labels in this field’s learned vocabulary, using the field’s string-folding settings.',
    },
    'Enum.vocabulary': {
      title: 'Declared enum labels',
      text: 'Return the declared labels in class order, including labels that have not appeared in training.',
    },
    'Group.vocabulary': {
      title: 'Known grouped identities',
      text: 'Return an immutable tuple of labels in this Group field’s learned vocabulary.',
    },
    'Boolean.counts': {
      title: 'Boolean training counts',
      text: 'Return observed training exposure as a dictionary keyed by False and True, excluding the initial prior.',
    },
    'Category.counts': {
      title: 'Category training counts',
      text: 'Return observed training counts keyed by the populated vocabulary labels, excluding the initial prior.',
    },
    'Enum.counts': {
      title: 'Enum training counts',
      text: 'Return observed training counts for every declared label, including zero counts for unseen classes. The initial prior is excluded.',
    },
    'Number.normalization': {
      title: 'Numeric normalization state',
      text: 'Inspect the raw, source-unit normalizer: mean, variance, standardizing scale, count, and alpha. This is a snapshot of learned state.',
    },
    'Duration.normalization': {
      title: 'Elapsed-time normalization state',
      text: 'Inspect normalization statistics for the derived duration, in the configured fixed time unit. This is a snapshot of learned state.',
    },
    'Ordinal.normalization': {
      title: 'Rank normalization state',
      text: 'Inspect normalization statistics for numeric ranks. Declared labels use their zero-based positions; implicit integer ranks keep their values.',
    },
    'Quantile.normalization': {
      title: 'Learned numeric distribution',
      text: 'Inspect count, compression, minimum, median, and maximum of this field’s learned distribution. Minimum, median, and maximum are None before finite observations.',
    },
    'Group.assignments': {
      title: 'Learned identity assignments',
      text: 'Return each known identity’s selected group and softmax probabilities over its assignment logits. These describe input routing, rather than decoded model predictions.',
    },
    'Group.assign': {
      title: 'Assign an identity',
      text: 'Persistently assign a known or newly admitted identity to an integer group index. Vocabulary storage grows when needed; use an idle model with an unshared vocabulary.',
    },
    'Group.override': {
      title: 'Temporary assignments',
      text: 'Temporarily map identities to integer group indices within a with block. Assignments are restored on exit. Use an idle model with an unshared vocabulary; saving or rebuilding is blocked during the override.',
    },
  }),
  options: {
    target: {
      title: 'Prediction target',
      text: 'Hide and supervise this leaf or every active leaf below a child branch. Each supplied record keeps its position when its answers are predicted.',
    },
    length: (/** @type {{limit: string | number}} */ { limit }) => ({
      title: 'Collection limit',
      text: `Retain at most ${limit} records in each collection at this level. Prediction lists keep their retained lengths rather than being padded to ${limit}.`,
    }),
    overflowTail: {
      title: 'Keep the latest context',
      text: 'Keep the final records when a collection exceeds its limit. Supply histories oldest to newest; relflow does not sort them.',
    },
    overflowError: {
      title: 'Reject excess records',
      text: 'Raise an error when this collection exceeds its length limit, instead of dropping records.',
    },
    fitPreset: {
      title: 'Training defaults',
      text: 'Select the bundled fit recipe’s training defaults for this optimizer run. Explicit fit settings win; this does not change the model architecture.',
    },
    architecturePreset: {
      title: 'Architecture defaults',
      text: 'Construct a new, untrained model with the selected bundled architecture recipe. Training settings are selected separately on fit.',
    },
    embed: {
      title: 'Export a representation',
      text: 'Include a learned representation in prediction output. Exporting a representation does not itself add a training objective.',
    },
    vectorWidth: (/** @type {{width: string | number}} */ { width }) => ({
      title: 'Vector width',
      text: `Every valued vector must have exactly ${width} non-null numeric coordinates. A repeated collection of records belongs in a Branch.`,
    }),
    strip: {
      title: 'Trim string labels',
      text: 'Remove surrounding whitespace before label lookup. Vocabulary entries and written labels use the trimmed form.',
    },
    uncased: {
      title: 'Fold string labels',
      text: 'Lowercase string labels before lookup. The vocabulary and prediction labels carry that lowercase form.',
    },
    threshold: {
      title: 'Probability threshold',
      text: 'For Set, filter written membership predictions at this threshold; None keeps all populated labels. For Boolean, configure decision metrics, leaving the probability output unchanged. Neither setting changes the training loss.',
    },
    topk: {
      title: 'Additional top-k predictions',
      text: 'Request candidate lists and accuracy metrics for each k. Values must exceed one: [3] asks for the top three known labels. The best label is already returned.',
    },
    objective: {
      title: 'Reconstruction loss',
      text: 'Choose the loss for this field’s reconstruction objective. Number, Duration, and Ordinal use mae, mse, or huber; Vector, Text, Image, and Audio use l1 or l2. This option does not make a field a target by itself.',
    },
    maxLength: (/** @type {{tokens: string | number}} */ { tokens }) => ({
      title: 'Text token limit',
      text: `Limit the frozen text encoder input to ${tokens} tokens. Longer text is truncated by the tokenizer; this is not a character limit.`,
    }),
    store: {
      title: 'Persistent encoder vectors',
      text: 'Choose the persistent artifact-cache directory. Matching encoders share vectors across fields, splits, and runs; omitting store selects the platform cache directory or RELFLOW_CACHE_DIR. This runtime resource stays outside checkpoints.',
    },
    cache: {
      title: 'Shared vector cache',
      text: 'True enables automatic persistent caching with bounded RAM and device hot tiers. False disables all cache tiers. A positive integer additionally caps hot entries; it does not replace the memory budget or bound disk growth.',
    },
    memory: {
      title: 'Shared host cache budget',
      text: 'Budget host-cache payloads in MiB for this shared store owner; None selects 64 MiB. Process overhead is separate. Fields sharing a store must agree on its memory and entry-cap settings.',
    },
    tokenizerBatchSize: {
      title: 'Tokenizer batch size',
      text: 'Bound text tokenization batches separately from encoder execution batches. The default is 4096 strings.',
    },
    resident: {
      title: 'Device cache budget',
      text: 'Budget the device hot tier in MiB. Zero disables that tier; None selects one eighth of free CUDA memory up to 512 MiB, and zero on other devices. This does not bound the encoder or all device memory.',
    },
    base: {
      title: 'Media path prefix',
      text: 'Resolve relative file names against this directory or object-store URI prefix instead of the working directory.',
    },
    immutable: {
      title: 'Immutable media paths',
      text: 'Treat each resolved path as permanent for a single version of the bytes so warm cache hits make zero source storage requests.',
    },
    maxWindows: (/** @type {{count: string | number}} */ { count }) => ({
      title: 'Maximum audio windows',
      text: `Average at most ${count} overlapping windows across a long recording when clip="chunks".`,
    }),
    format: {
      title: 'Prediction file format',
      text: 'Parquet writes rank-N.parquet; ipc writes Arrow IPC files named rank-N.arrow. IPC preserves sparse nested fixed-size vector results that Parquet cannot round-trip.',
    },
    groupCount: (/** @type {{count: string | number}} */ { count }) => ({
      title: 'Shared group count',
      text: `Use ${count} shared identity representations. The count is fixed and must exceed one; it does not establish that the data has that many meaningful segments.`,
    }),
    granularity: {
      title: 'Elapsed-time unit',
      text: 'Write and model Duration values in this fixed unit. Days are 24 hours and weeks are seven days; months and years are not fixed units.',
    },
    delta: {
      title: 'Gap between records',
      text: 'Subtract the preceding timestamp in the same branch collection, in source order and before cropping. A record without a predecessor has a null duration.',
    },
    query: {
      title: 'Explicit source paths',
      text: 'Read source values through paths relative to this node’s parent record. Duration accepts a pair of paths in start, end order; ../ selects an ancestor record within branch geometry.',
    },
    endpoint: {
      title: 'Fixed timestamp endpoint',
      text: 'Use this configured timestamp as one Duration endpoint. Supply the other endpoint through the source record; the model never substitutes the current time.',
    },
    values: {
      title: 'Ordered label declaration',
      text: 'For Ordinal, this tuple declares labels in increasing rank order. The first label has rank zero; the next has rank one. Prediction returns a declared label after rounding and clamping.',
    },
    pattern: {
      title: 'Timestamp format',
      text: 'Parse timestamp strings using this explicit format. Period selects calendar parts; Duration computes elapsed time from timestamp endpoints.',
    },
    task: {
      title: 'Learning task',
      text: 'Declare a task on a leaf or branch. rf.reconstruct selects observations to hide and use as answers.',
    },
    hide: {
      title: 'Hide training inputs',
      text: 'Hide selected values during training. A branch shares each row decision across children outside its ignore list.',
    },
    ablate: {
      title: 'Training ablation',
      text: 'Drop the input and its targets during training. A leaf samples per record. A child branch samples once per parent for the entire collection.',
    },
    batchSize: (/** @type {{size: string | number}} */ { size }) => ({
      title: 'Batch size',
      text: `Process up to ${size} observations per batch. Training data loaders use this size per device; direct prediction chunks processed rows.`,
    }),
    retain: {
      title: 'Retain application fields',
      text: 'Carry selected input fields into output for joining predictions back to your application. Retaining a field does not make it a model input.',
    },
  },
  workflows: /** @type {Record<string, Note>} */ ({
    fit: {
      title: 'Model workflow',
      text: 'Train the model on the supplied data. Each fit starts a new optimizer run; use a checkpoint to resume full training state.',
    },
    predict: {
      title: 'Model workflow',
      text: 'Return an eager Polars DataFrame containing the configured predictions and representations, with nested outputs aligned to retained records.',
    },
    save: {
      title: 'Model workflow',
      text: 'Save the model schema and learned state for later loading. External encoder files and caches remain separate dependencies.',
    },
    load: {
      title: 'Model workflow',
      text: 'Load a saved relflow model with its schema and learned state. Supply the runtime dependencies required by its field types.',
    },
  }),
  contracts: /** @type {Record<string, Note>} */ ({
    inputs: {
      title: 'Known inputs',
      text: 'This information is supplied when making a prediction. A field reads its matching name or the source paths declared by its query. Nested records preserve their fields and positions.',
    },
    answers: {
      title: 'Training answer',
      text: 'These are known labels, or source values used to derive labels, in training records. The schema masks the target values from model inputs. Nested containers identify the record each answer belongs to.',
    },
    training: {
      title: 'Complete training record',
      text: 'Training observations contain both known inputs and labels. The schema decides which values are visible as context and which are hidden targets.',
    },
    predictions: {
      title: 'Prediction output',
      text: 'One row of the prediction table. Output names and repeated collections follow the model schema.',
    },
  }),
  outputs: /** @type {Record<string, Note>} */ ({
    predictions: {
      title: 'Named prediction outputs',
      text: 'The model’s decoded fields, using their schema names and nesting. model.predict returns a table; this example shows one row.',
    },
    state: {
      title: 'Value-state probabilities',
      text: 'Probabilities for valued, null, padded, masked, and other states. They sum to one and describe the kind of value, separately from the decoded content.',
    },
    valued: {
      title: 'Valued state',
      text: 'Probability that this field represents a supplied value, rather than a null, padding, masking, or other state.',
    },
    null: {
      title: 'Null state',
      text: 'Probability of a null value. Null is distinct from zero, false, or an empty label set.',
    },
    padded: {
      title: 'Padded state',
      text: 'Probability of the padding state. Branch prediction lists still follow retained records and do not add cap-padding rows.',
    },
    masked: {
      title: 'Masked state',
      text: 'Probability of the masking state, separate from whether this prediction was inferred.',
    },
    other: {
      title: 'Other state',
      text: 'Probability of a value outside this type’s ordinary valued or null states. Its interpretation is owned by the field type.',
    },
    content: {
      title: 'Decoded answer',
      text: 'The field type defines this result: a scalar for Number, Duration, or Quantile; a rank or ordered label for Ordinal; a probability of true for Boolean; numeric coordinates for Vector; a Category or Enum label readout; or independent Set memberships.',
    },
    inferred: {
      title: 'Inference flag',
      text: 'True means this field was selected for inference. It is a routing flag, not a confidence score; predict=True targets are inferred during prediction.',
    },
    value: {
      title: 'Selected value',
      text: 'A label or group index in this readout. Category and Enum select a label. Set lists the label for each returned membership prediction. Group’s decoded index is separate from its learned identity assignments.',
    },
    probability: {
      title: 'Prediction probability',
      text: 'Boolean reports the probability of true. Category and Enum labels are alternatives; Set memberships are independent and need not sum to one. A Group decoder’s probability is not a trained identity-assignment confidence.',
    },
    topk: {
      title: 'Candidate labels',
      text: 'The requested highest-probability labels, ordered by probability. The list may cover only part of the vocabulary, so its probabilities need not sum to one.',
    },
    group: {
      title: 'Group decoder readout',
      text: 'The selected zero-based index from Group’s decoder. Its head has no built-in group reconstruction objective; inspect known-label assignments with Group.assignments instead.',
    },
  }),
  fields: {
    source: (/** @type {{path: string}} */ { path }) => ({
      title: 'Queried source field',
      text: `Read ${path} from the source record. The path is relative to this node’s parent; ../ moves to an ancestor. This source name can differ from the declared model field.`,
    }),
    schema: (/** @type {{field: string}} */ { field }) => ({
      title: 'Schema field',
      text: `${field} declares a value or related collection. It reads the matching record member unless a query selects other source paths. Use predict=True to make its answers supervised targets. Matching names are highlighted across this example.`,
    }),
    prediction: (/** @type {{field: string}} */ { field }) => ({
      title: 'Prediction field',
      text: `The result for ${field}, following its position in the model schema. Repeated records retain their input order.`,
    }),
  },
  syntax: {
    uvRun: {
      title: 'Run the example',
      text: 'Run this script in a Python environment managed by uv. The --with dependency below pins the example’s tested relflow revision.',
    },
    textDependencies: {
      title: 'Text encoder dependencies',
      text: 'Install the optional text dependencies needed to load a pretrained tokenizer and encoder.',
    },
    list: {
      title: 'List-valued field',
      text: 'Keep this collection in the observation. A Branch represents repeated records; Set represents unordered labels; Vector represents fixed-width numeric coordinates.',
    },
  },
};
