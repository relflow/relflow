import { datasets } from './datasets.ts';
import type { CaseId } from './concepts/types';
import type { Dataset } from '../datasets/types';

/** Collection names in the canonical class schemas, keyed by child entity. */
const collections: Record<CaseId, Record<string, string>> = {
  tickets: { message: 'messages' },
  churn: { monthly_usage: 'monthly_usage', support_ticket: 'support_tickets' },
  demand: { sales_week: 'history', forecast_week: 'horizon' },
  credit: { account: 'accounts', inquiry: 'inquiries' },
  maintenance: { sensor_reading: 'sensor_readings', work_order: 'work_orders' },
  claims: { claim: 'claims', invoice: 'invoices', line_item: 'line_items' },
};

/** Reuse field meanings while showing the nested records consumed by the examples. */
export const observationDatasets = Object.fromEntries(datasets.map((dataset) => {
  const names = collections[dataset.id as CaseId];
  const observation: Dataset = {
    ...dataset,
    id: `${dataset.id}-observation`,
    entities: dataset.entities.map((entity) => ({
      ...entity,
      columns: [
        ...entity.columns.filter((column) => column.role !== 'key' && column.role !== 'reference'),
        ...dataset.relations.filter((relation) => relation.from === entity.id).map((relation) => ({
          name: names[relation.to], type: 'records[]', role: 'collection' as const,
          description: dataset.entities.find((item) => item.id === relation.to)!.description,
        })),
      ],
    })),
    relations: dataset.relations.map((relation) => ({ from: relation.from, to: relation.to, fromColumn: names[relation.to], description: relation.description })),
  };
  return [dataset.id, observation];
})) as Record<CaseId, Dataset>;

export const repeatPurchase: Dataset = {
  id: 'repeat-purchase', label: 'Repeat purchases',
  observation: 'One customer at a cutoff, with orders placed by then.',
  description: 'Account context and past orders teach whether the customer buys again in the following 30 days.',
  entities: [
    {
      id: 'customer', name: 'Customer', description: 'One decision-time customer record and its historical return label.',
      columns: [
        { name: 'customer_id', type: 'string', role: 'key', description: 'Identifier retained to match predictions to requests. This field is outside the model schema and is not embedded as input.' },
        { name: 'days_since_signup', type: 'number', description: 'Days between signup and the prediction cutoff.' },
        { name: 'orders', type: 'records[]', role: 'collection', description: 'Orders known by the cutoff, supplied oldest to newest. Each order keeps its own amount, category, and age.' },
        { name: 'returns_in_30d', type: 'boolean', role: 'target', description: 'Whether another purchase occurred in the 30 days after the cutoff. Historical training label; omitted from new prediction requests.' },
      ],
    },
    {
      id: 'order', name: 'Order', description: 'One order inside this customer’s supplied history.',
      columns: [
        { name: 'amount', type: 'number', description: 'Order value in the dataset’s currency units.' },
        { name: 'category', type: 'string', description: 'Category of the order, such as home or books.' },
        { name: 'days_ago', type: 'number', description: 'Order age in days at the cutoff. Future orders are excluded.' },
      ],
    },
  ],
  relations: [{ from: 'customer', to: 'order', fromColumn: 'orders', description: 'Each customer observation carries its own list of past orders.' }],
};
