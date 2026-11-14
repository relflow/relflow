import originalCases from '../use-cases.json' with { type: 'json' };
import type { ExampleValue } from '../type-example';
import type { CaseId } from './types';
import { observationDatasets } from '../observations.ts';

interface ScenarioDetails {
  modelClass: string;
  inputs: Record<string, ExampleValue>;
  answers: Record<string, ExampleValue>;
  /** Absolute schema addresses; targets inside branches refer to each supplied record. */
  primaryBranch: string;
  primaryTarget: string;
  targets: { path: string; type: 'Boolean' | 'Number'; label: string }[];
}

/** Shared illustrations; these values are neither training fixtures nor measured predictions. */
const details: Record<CaseId, ScenarioDetails> = {
  tickets: {
    modelClass: 'TicketModel', primaryBranch: '/messages', primaryTarget: '/escalated',
    targets: [
      { path: '/escalated', type: 'Boolean', label: 'Escalation' },
      { path: '/hours_to_resolve', type: 'Number', label: 'Hours to resolution' },
    ],
    inputs: {
      product: 'analytics', priority: 'high', messages: [
        { author: 'customer', body: 'The export stops halfway through.', minutes_since_previous: 0 },
        { author: 'agent', body: 'Which export format are you using?', minutes_since_previous: 4 },
      ],
    },
    answers: { escalated: false, hours_to_resolve: 6 },
  },
  churn: {
    modelClass: 'ChurnModel', primaryBranch: '/monthly_usage', primaryTarget: '/churn_90d',
    targets: [{ path: '/churn_90d', type: 'Boolean', label: 'Cancellation within 90 days' }],
    inputs: {
      plan: 'pro', tenure_months: 12,
      monthly_usage: [
        { month: '2026-09-01', minutes: 240, data_gb: 12, overage_fee: 0 },
        { month: '2026-10-01', minutes: 80, data_gb: 4, overage_fee: 0 },
      ],
      support_tickets: [{ channel: 'chat', resolved: true, days_open: 1 }],
    },
    answers: { churn_90d: true },
  },
  demand: {
    modelClass: 'DemandModel', primaryBranch: '/history', primaryTarget: '/horizon/units_sold',
    targets: [{ path: '/horizon/units_sold', type: 'Number', label: 'Units for each supplied future week' }],
    inputs: {
      store: 'store-4', sku: 'boots-9',
      history: [{ week: '2026-10-26', units_sold: 18, promotion: false, stockout: false }],
      horizon: [
        { week: '2026-11-02', promotion: true },
        { week: '2026-11-09', promotion: false },
      ],
    },
    answers: { horizon: [{ units_sold: 22 }, { units_sold: 17 }] },
  },
  credit: {
    modelClass: 'CreditModel', primaryBranch: '/inquiries', primaryTarget: '/default_12m',
    targets: [{ path: '/default_12m', type: 'Boolean', label: 'Default within 12 months' }],
    inputs: {
      product: 'installment', requested_amount: 5000,
      accounts: [{ account_type: 'card', balance: 1200, months_open: 24, missed_payments: 0 }],
      inquiries: [{ lender: 'lender-8', age_days: 12 }],
    },
    answers: { default_12m: false },
  },
  maintenance: {
    modelClass: 'MaintenanceModel', primaryBranch: '/sensor_readings', primaryTarget: '/failure_30d',
    targets: [{ path: '/failure_30d', type: 'Boolean', label: 'Failure within 30 days' }],
    inputs: {
      asset_type: 'pump', age_years: 4.5,
      sensor_readings: [{ hour: '2026-11-01T08:00:00', vibration: 0.18, temperature: 63, load: 0.6 }],
      work_orders: [{ type: 'inspection', parts_replaced: [], age_days: 7 }],
    },
    answers: { failure_30d: false },
  },
  claims: {
    modelClass: 'ClaimsModel', primaryBranch: '/claims', primaryTarget: '/claims/paid_amount',
    targets: [
      { path: '/claims/paid_amount', type: 'Number', label: 'Paid amount for each submitted claim' },
      { path: '/claims/is_fraudulent', type: 'Boolean', label: 'Fraud probability for each submitted claim' },
    ],
    inputs: {
      policy_type: 'auto', coverage_limit: 25000,
      claims: [{ cause: 'collision', reported_after_days: 2,
        invoices: [{ provider: 'repair-7', line_items: [{ code: 'panel', amount: 780 }, { code: 'labor', amount: 180 }] }],
      }],
    },
    answers: { claims: [{ paid_amount: 900, is_fraudulent: false }] },
  },
};

export const scenarios = originalCases.map((item) => {
  const id = item.id as CaseId;
  if (!details[id]) throw new Error(`Core Concepts has no shared scenario for ${item.id}.`);
  return { ...item, ...details[id], id, schema: item.code, dataset: observationDatasets[id] };
});
