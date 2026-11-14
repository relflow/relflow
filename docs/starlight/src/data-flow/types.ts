import type { ExampleValue } from '../data/type-example';
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
    | { kind: 'dataset'; dataset: Dataset }
    | { kind: 'records'; value: ExampleValue }
    | { kind: 'tree'; nodes: TreeNode[] }
    | {
        kind: 'prediction';
        value: Record<string, ExampleValue>;
        inputs: Record<string, ExampleValue>;
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
