export type CaseId = 'tickets' | 'churn' | 'demand' | 'credit' | 'maintenance' | 'claims';

import type { ExampleValue } from '../type-example';

interface ConceptCode {
  kind?: 'code';
  label: string;
  lang: 'python' | 'yaml';
  code: string;
  note?: string;
  /** Source records use the explorer as their primary view; YAML stays inspectable. */
  records?: boolean;
}

interface ConceptPrediction {
  kind: 'prediction';
  label: string;
  value: Record<string, ExampleValue>;
  inputs?: Record<string, ExampleValue>;
  note?: string;
  code?: never;
  lang?: never;
  records?: never;
}

interface ConceptModel {
  kind: 'model';
  label: string;
  caseId: CaseId;
  note?: string;
  code?: never;
  lang?: never;
  records?: never;
}

export type ConceptBlock = ConceptCode | ConceptPrediction | ConceptModel;

export interface ConceptExample {
  caseId: CaseId;
  title: string;
  description: string;
  blocks: ConceptBlock[];
  takeaway: string;
}

export interface Concept {
  id: string;
  title: string;
  description: string;
  intro: string[];
  principles: { title: string; text: string }[];
  examples: ConceptExample[];
  takeaway: string;
  links: { label: string; href: string }[];
}
