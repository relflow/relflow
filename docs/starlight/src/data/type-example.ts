import type { Value } from '../values/types';
import type { Dataset } from '../datasets/types';

export interface TypeExample {
  id: string;
  label: string;
  summary: string;
  title: string;
  description: string;
  dataset: Dataset;
  schema: string;
  inputs: Record<string, Value>;
  answers: Record<string, Value>;
  answerNote?: string;
  /** One illustrative row from model.predict(...). */
  output: Record<string, Value>;
  outputNote: string;
  takeaway: string;
}

export function prediction(content: Value): Record<string, Value> {
  return {
    state: { valued: 0.96, null: 0.01, padded: 0.01, masked: 0.01, other: 0.01 },
    content,
    inferred: true,
  };
}

