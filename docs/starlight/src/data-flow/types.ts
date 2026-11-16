import type { Value } from '../values/types';
import type { CaseId } from '../data/concepts/types';
import type { Dataset } from '../datasets/types';
import type { TreeNode } from '../tree/types';

export type FlowMode = 'predict' | 'train';
export type FlowStageId =
  | 'load'
  | 'preprocess'
  | 'prepare'
  | 'model'
  | 'write'
  | 'postprocess'
  | 'result'
  | 'loss'
  | 'backward'
  | 'update';

interface ViewBase {
  label: string;
  mode?: FlowMode;
}

export type FlowView = ViewBase &
  (
    | { kind: 'dataset'; dataset: Dataset; schema: string }
    | { kind: 'records'; value: Value }
    | { kind: 'tree'; nodes: TreeNode[] }
    | {
        kind: 'prediction';
        value: Record<string, Value>;
        inputs: Record<string, Value>;
      }
    | { kind: 'text'; text: string }
  );

export interface FlowCode {
  label: string;
  source: string;
  mode?: FlowMode;
}

export interface FlowStage {
  id: FlowStageId;
  title: string;
  summary: string;
  description: string;
  mode?: FlowMode;
  optional?: boolean;
  views: FlowView[];
  code?: FlowCode[];
  note?: string;
}

export interface FlowCase {
  id: CaseId;
  label: string;
  structure: string;
  title: string;
  description: string;
  stages: FlowStage[];
}

export const flowPaths: Record<FlowMode, readonly FlowStageId[]> = {
  predict: ['load', 'preprocess', 'prepare', 'model', 'write', 'postprocess', 'result'],
  train: ['load', 'preprocess', 'prepare', 'model', 'loss', 'backward', 'update'],
};

export interface FlowTransition {
  input: string;
  output: string;
}

export const flowTransitions: Record<FlowStageId, FlowTransition> = {
  load: { input: 'Prepared data sources', output: 'Nested source records' },
  preprocess: { input: 'Nested source records', output: 'Prepared observations' },
  prepare: { input: 'Prepared observations', output: 'Typed inputs + collection geometry' },
  model: { input: 'Typed inputs + collection geometry', output: 'Target estimates' },
  write: { input: 'Target estimates', output: 'Rich prediction rows' },
  postprocess: { input: 'Rich prediction rows', output: 'Rich prediction rows' },
  result: { input: 'Rich prediction rows', output: 'Polars DataFrame' },
  loss: { input: 'Target estimates + historical answers', output: 'Training objective' },
  backward: { input: 'Training objective', output: 'Parameter gradients' },
  update: { input: 'Parameter gradients', output: 'Updated model' },
};
