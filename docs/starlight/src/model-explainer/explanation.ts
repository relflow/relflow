import type { ModelNode, ModelStage } from '../data/model-explainer';
import { flatten } from './schema.ts';
import { reference } from '../reference.mjs';

export interface Explanation {
  name: string;
  path: string;
  kind: 'model' | 'field' | 'branch' | 'target';
  type: string;
  declaration: string;
  description: string;
  representation: string;
  context: string;
  learning: string;
  output: string;
  address: string;
  reference: string;
  requirement: string;
}

/** Public prediction addresses retain every repeated ancestor and the rich leaf readout. */
export function outputAddress(node: ModelNode, nodes: ModelNode[]): string {
  const branches = new Set(
    flatten(nodes)
      .filter((item) => item.kind === 'branch')
      .map((item) => item.path),
  );
  const parts = node.path.slice(1).split('/');
  let path = '';
  let index = 0;
  const address = parts
    .map((part) => {
      path += `/${part}`;
      if (!branches.has(path)) return part;
      const ordinal = ['i', 'j', 'k'][index] || `i${index}`;
      index++;
      return `${part}[${ordinal}]`;
    })
    .join('.');
  return `predictions.${address}.content${node.type === 'Boolean' ? '.probability' : ''}`;
}

/** A construction explanation, not an executed network graph or a measured prediction. */
export function explain(
  node: ModelNode | undefined,
  stage: ModelStage,
  modelClass: string,
): Explanation {
  if (!node)
    return {
      name: modelClass,
      path: '/',
      kind: 'model',
      type: 'Model',
      declaration: `${modelClass}(preset="small")`,
      description:
        'The schema describes one assembled observation. Its field types, record boundaries, and hidden answers determine the model components.',
      representation:
        'Typed fields choose how their values become representations the model can use.',
      context:
        'Record encoders combine the available fields and related context while preserving their declared structure.',
      learning:
        'Each target contributes its own objective. Fitting learns from labeled observations through shared context.',
      output: 'After fitting, each requested target is decoded at its declared record level.',
      address: 'predictions',
      reference: '/api-map/?section=schema',
      requirement: '',
    };
  const parent = node.path.slice(0, node.path.lastIndexOf('/')) || '/';
  if (node.kind === 'branch')
    return {
      name: node.name,
      path: node.path,
      kind: 'branch',
      type: node.className || 'Branch',
      declaration: `${node.name} = ${node.declaration}`,
      description: node.description,
      representation:
        'Each supplied record keeps its typed fields together. This collection is context, rather than a hand-built table of aggregate features.',
      context: `Encode records at ${node.path}, combine their repeated context, and pass a summary to ${parent === '/' ? 'the root observation' : parent}. The schema keeps record boundaries explicit.`,
      learning:
        'Downstream target objectives train the context modules. Adding a branch does not create a learning objective by itself.',
      output:
        'Targets inside this branch are written for its retained records. A context-only branch has no target readout of its own.',
      address: '',
      reference: '/concepts/branches/',
      requirement: '',
    };
  const representation = reference.representations[node.type];
  if (!representation)
    throw new Error(
      `Model explainer has no verified representation for ${node.type} at ${node.path}.`,
    );
  if (node.target && !['Boolean', 'Number'].includes(node.type))
    throw new Error(
      `Model explainer needs a verified target explanation for ${node.type} at ${node.path}.`,
    );
  return {
    name: node.name,
    path: node.path,
    kind: node.target ? 'target' : 'field',
    type: node.type,
    declaration: `${node.name} = ${node.declaration}`,
    description: node.description,
    representation: node.target
      ? 'predict=True keeps this answer value out of encoder context. Its type still defines field-owned state and the decoder contract.'
      : representation,
    context: node.target
      ? `The decoder uses available context to answer at ${parent === '/' ? 'the observation root' : `${parent}, for each retained record`}. The masked answer is never an input clue.`
      : `This value enters ${parent === '/' ? 'the root observation' : `its enclosing ${parent} record`}. Related records keep their own fields and nesting before their context is combined.`,
    learning: node.target
      ? node.type === 'Boolean'
        ? 'A binary classification objective learns this true/false outcome from the remaining context.'
        : 'A numeric regression objective learns this answer from the remaining context.'
      : 'This input has no reconstruction objective in this example. Target objectives teach the model how to use its representation.',
    output: node.target
      ? node.type === 'Boolean'
        ? 'A probability of True, with the rich leaf’s state and inferred fields retained.'
        : 'A scalar in the source units, with the rich leaf’s state and inferred fields retained.'
      : 'Used as input context. This declaration does not request a prediction for this field.',
    address: node.target ? outputAddress(node, stage.nodes) : '',
    reference: `/data-types/${node.type.toLowerCase()}/`,
    requirement: node.labelRequirement || '',
  };
}
