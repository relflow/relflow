import type { Dataset } from '../datasets/types';

export type ExampleValue =
  | string
  | number
  | boolean
  | null
  | ExampleValue[]
  | { [key: string]: ExampleValue };

export interface TypeExample {
  id: string;
  label: string;
  summary: string;
  title: string;
  description: string;
  dataset: Dataset;
  schema: string;
  inputs: Record<string, ExampleValue>;
  answers: Record<string, ExampleValue>;
  answerNote?: string;
  /** One illustrative row from model.predict(...). */
  output: Record<string, ExampleValue>;
  outputNote: string;
  takeaway: string;
}

export function prediction(content: ExampleValue): Record<string, ExampleValue> {
  return {
    state: { valued: 0.96, null: 0.01, padded: 0.01, masked: 0.01, other: 0.01 },
    content,
    inferred: true,
  };
}

