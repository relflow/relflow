import type { Value } from '../../values/types';
import type { FlowCase, FlowMode } from '../../data-flow/types.ts';
import type { TreeNode } from '../../tree/types.ts';
import { parseSchema, type ModelNode } from '../../model-explainer/schema.ts';
import { scenarios } from './scenarios.ts';
import { predictions } from './predictions.ts';

/** Attach historical answers without changing the shared inputs or their record alignment. */
function labeled(input: Value, answers: Value): Value {
  if (Array.isArray(answers)) {
    if (!Array.isArray(input) || input.length !== answers.length)
      throw new Error('Data Flow training answers must align with their supplied record collection.');
    return input.map((record, index) => labeled(record, answers[index]));
  }
  if (answers !== null && typeof answers === 'object') {
    if (input === null || typeof input !== 'object' || Array.isArray(input))
      throw new Error('Data Flow nested answers need a matching observation record.');
    const result = { ...input };
    for (const [name, value] of Object.entries(answers))
      result[name] = name in input ? labeled(input[name], value) : structuredClone(value);
    return result;
  }
  return answers;
}

/** Describe the encoded contract without inventing tensor values or embedding dimensions. */
function prepared(nodes: readonly ModelNode[], mode: FlowMode): TreeNode[] {
  return nodes.map((node) => ({
    label: node.name,
    detail: node.kind === 'branch'
      ? 'Collection structure + validity'
      : node.target
        ? mode === 'train' ? `${node.type} · hidden training answer` : `${node.type} · answer omitted`
        : `${node.type} · typed input + state`,
    ...(node.children.length ? { children: prepared(node.children, mode) } : {}),
  }));
}

/** Keep the declared branches and targets visible independently of the selected mode. */
function hierarchy(nodes: readonly ModelNode[]): TreeNode[] {
  return nodes.map((node) => ({
    label: node.name,
    detail: node.kind === 'branch'
      ? 'Branch · related records'
      : `${node.type}${node.target ? ' · masked target' : ''}`,
    ...(node.children.length ? { children: hierarchy(node.children) } : {}),
  }));
}

/** Describe public output fields at their actual nested target locations. */
function readouts(nodes: readonly ModelNode[]): TreeNode[] {
  return nodes.flatMap((node): TreeNode[] => {
    if (node.kind === 'branch') {
      const children = readouts(node.children);
      return children.length ? [{ label: node.name, detail: 'Aligned record list', children }] : [];
    }
    if (!node.target) return [];
    return [{ label: node.name, children: [
      { label: 'state', detail: 'Field-state probabilities' },
      { label: 'content', detail: node.type === 'Boolean' ? 'Probability of True' : 'Numeric estimate' },
      { label: 'inferred', detail: 'Inference marker' },
    ] }];
  });
}

export const flowCases: FlowCase[] = scenarios.map((scenario) => {
  const parsed = parseSchema(scenario.schema);
  const example = predictions.examples.find((item) => item.caseId === scenario.id);
  const result = example?.blocks.find((block) => block.kind === 'prediction');
  if (!result || result.kind !== 'prediction')
    throw new Error(`${scenario.id}: Data Flow needs its canonical illustrative prediction row.`);
  const training = labeled(scenario.inputs, scenario.answers);
  const targets: TreeNode[] = scenario.targets.map((target) => ({
    label: target.path,
    detail: target.type === 'Boolean' ? 'Boolean · probability of True' : 'Number · numeric estimate',
  }));
  const resources: TreeNode[] = [
    { label: 'Prediction', detail: 'Reuse fitted field state' },
    { label: 'Training', detail: 'Apply registered field updates' },
  ];
  const model: TreeNode[] = [
    { label: '1 · Embed', detail: 'Represent known input values', children: [
      { label: 'Typed inputs', detail: 'Type-owned representations' },
      { label: 'Masked targets', detail: 'Withheld from input embedding' },
    ] },
    { label: '2 · Build context', detail: 'Combine fields and records', children: [
      { label: parsed.modelClass, detail: 'Root observation', children: hierarchy(parsed.nodes) },
    ] },
    { label: '3 · Decode', detail: 'Read context for each target', children: targets },
  ];
  const output: TreeNode[] = [
    { label: 'predictions', detail: 'Schema names and nesting', children: readouts(parsed.nodes) },
  ];
  const calls = [
    { mode: 'predict' as const, label: 'Run prediction with a fitted model', source:
      `# model is a fitted ${scenario.modelClass}.\nresults = model.predict(datamodule=data, devices=1)` },
    { mode: 'train' as const, label: 'Fit the declared model', source:
      `model = ${scenario.modelClass}(preset="small")\nmodel.fit(data, preset="small")` },
  ];
  return {
    id: scenario.id,
    label: scenario.label,
    structure: scenario.structure,
    title: scenario.title,
    description: scenario.dataset.observation,
    stages: [
      {
        id: 'load', title: 'Load data',
        summary: 'Read observations while preserving their nested records.',
        description: 'A data module reads source chunks from prepared Parquet files. The flow follows one observation within that stream. Prediction records contain known context; historical training records also contain settled answers.',
        views: [
          { kind: 'dataset', label: 'Observation contract', dataset: scenario.dataset, schema: scenario.schema },
          { kind: 'records', label: 'Known source values', mode: 'predict', value: scenario.inputs },
          { kind: 'records', label: 'Context and historical answers', mode: 'train', value: training },
        ],
        code: [{ label: 'Configure application-prepared sources', source:
          `import relflow as rf\n\ndata = rf.ArrowDataModule(\n    train="${scenario.id}/train.parquet",\n    validate="${scenario.id}/validation.parquet",\n    predict="${scenario.id}/requests.parquet",\n)` }],
        note: 'The explorer describes the complete schema, including target fields. New prediction requests omit those answers. The file paths illustrate application-prepared sources.',
      },
      {
        id: 'preprocess', title: 'Preprocess', optional: true,
        summary: 'Apply any configured transformations to source records.',
        description: 'No custom preprocessor is configured in this example: the prepared records pass through. A preprocessor can rename, join, filter, sort, or group records using an eager Polars frame before schema queries and model batching.',
        views: [
          { kind: 'text', label: 'Before', text: 'Prepared source records, with each history and nested collection already assembled.' },
          { kind: 'text', label: 'After · pass-through', text: 'The original prepared records pass through unchanged.' },
        ],
        note: 'The application chooses observation boundaries and cutoffs. For direct in-memory prediction, preprocessing runs once before processed Arrow rows are divided into model batches.',
      },
      {
        id: 'prepare', title: 'Prepare inputs',
        summary: 'Resolve field paths, encode typed values, and bind model resources.',
        description: 'Schema names and explicit queries select values inside each record. Type extensions tensorize those values and their states; branch geometry keeps collections aligned. Model resources are then bound before the forward pass.',
        views: [
          { kind: 'tree', label: 'Schematic input contract', mode: 'predict', nodes: prepared(parsed.nodes, 'predict') },
          { kind: 'tree', label: 'Schematic input contract', mode: 'train', nodes: prepared(parsed.nodes, 'train') },
          { kind: 'tree', label: 'Model-owned field state', nodes: resources },
        ],
        code: [{ label: 'The class schema configures this stage', source: scenario.schema }],
        note: 'Training applies registered field-state updates, including vocabulary and normalization learning where supported. Prediction reuses fitted state. These updates are separate from gradient-based neural weight updates.',
      },
      {
        id: 'model', title: 'Model',
        summary: 'Embed values, combine branch context, and decode the configured targets.',
        description: 'Embedders represent known input values. Branch encoders combine related fields and records through the hierarchy. Decoders use that context to estimate the masked targets at their declared locations.',
        views: [
          { kind: 'tree', label: 'Forward pass · schematic phases', nodes: model },
          { kind: 'text', label: 'Forward-pass behavior', mode: 'predict', text: 'The fitted model uses inference behavior and its learned parameters to produce estimates. No training answers are supplied.' },
          { kind: 'text', label: 'Forward-pass behavior', mode: 'train', text: 'The model estimates its targets while their observed answers remain hidden from input embedding. The next stage compares those estimates with the historical answers.' },
        ],
        code: calls,
        note: 'Class declarations describe the architecture. Prediction uses fitted parameters; training updates the participating trainable components.',
      },
      {
        id: 'write', title: 'Build prediction output', mode: 'predict',
        summary: 'Assemble typed results under their schema names and collection boundaries.',
        description: 'Type-owned writers assemble the decoded values into the public prediction format. Branch collections remain aligned lists of records. This step constructs output fields; saving a file is a separate application choice.',
        views: [
          { kind: 'tree', label: 'Before · decoded target estimates', nodes: targets },
          { kind: 'tree', label: 'After · rich prediction fields', nodes: output },
        ],
        note: 'The state field describes value states. A Boolean answer is its content.probability.',
      },
      {
        id: 'postprocess', title: 'Postprocess', mode: 'predict', optional: true,
        summary: 'Apply any configured application transformations to the prediction rows.',
        description: 'No postprocessor is configured here, so the prediction rows pass through. A postprocessor can select or reshape fields in an eager Polars frame for an application or warehouse.',
        views: [
          { kind: 'text', label: 'Before', text: 'Rich prediction rows with the names and nesting declared by the schema.' },
          { kind: 'text', label: 'After · pass-through', text: 'The same prediction rows, ready for the application to consume.' },
        ],
        note: 'Direct in-memory prediction postprocesses its combined result once. Batch export through Writer postprocesses each prediction batch before writing it to storage.',
      },
      {
        id: 'result', title: 'Returned predictions', mode: 'predict',
        summary: 'Read the selected observation’s typed and nested answers.',
        description: 'The standard prediction call returns an eager Polars DataFrame with one row per processed observation. This tree shows the existing illustrative result row for the selected record.',
        views: [{ kind: 'prediction', label: 'One illustrative prediction row', value: result.value, inputs: scenario.inputs }],
        code: [{ label: 'Inspect a returned row', source:
          'import polars as pl\n\nresults: pl.DataFrame = model.predict(datamodule=data, batch_size=128, devices=1)\nrow = results.row(0, named=True)' }],
        note: 'This is the same illustrative result row used in the Predictions examples.',
      },
      {
        id: 'loss', title: 'Compute loss', mode: 'train',
        summary: 'Compare target estimates with the corresponding historical answers.',
        description: 'Each configured reconstruction objective compares the model’s estimate with its observed answer. Boolean and numeric targets use the objectives provided by their respective types. Training reaches loss before constructing public prediction tables.',
        views: [
          { kind: 'records', label: 'Historical answers · aligned with their records', value: scenario.answers },
          { kind: 'tree', label: 'Type-owned objectives · no measured loss shown', nodes: [{
            label: 'Training objective', detail: 'Contributions from selected targets',
            children: scenario.targets.map((target) => ({ label: target.path, detail: `${target.type} reconstruction` })),
          }] },
        ],
      },
      {
        id: 'backward', title: 'Propagate gradients', mode: 'train',
        summary: 'Differentiate the loss through the participating trainable parameters.',
        description: 'Backward propagation determines gradients for parameters participating in the objective, through target decoders, branch context, and trainable input representations. Frozen components retain their parameters.',
        views: [{ kind: 'tree', label: 'Schematic gradient path', nodes: [{
          label: 'Training loss', children: [{
            label: 'Gradients', detail: 'Participating trainable parameters', children: [
              { label: 'Target decoders' }, { label: 'Branch encoders' }, { label: 'Trainable input representations' },
            ],
          }],
        }] }],
        note: 'Lightning owns the backward pass in ordinary model.fit training.',
      },
      {
        id: 'update', title: 'Update parameters', mode: 'train',
        summary: 'Let the optimizer update neural parameters, then continue with the next batch.',
        description: 'The optimizer uses accumulated gradients and its run state to update neural parameters at an optimizer step. Training continues with subsequent observations using the updated model.',
        views: [
          { kind: 'tree', label: 'Before', nodes: [{ label: 'Optimizer step', children: [
            { label: 'Gradients' }, { label: 'Optimizer state' }, { label: 'Learning rate', detail: 'Provided by the fit preset' },
          ] }] },
          { kind: 'tree', label: 'After', nodes: [{ label: 'Updated neural parameters' }, { label: 'Continue to the next batch' }] },
        ],
        code: [calls[1]],
        note: 'Gradient accumulation can span several batches. Field-state updates occurred earlier during input preparation; this step concerns neural weights. “Next batch” returns to the beginning of this illustration.',
      },
    ],
  };
});
