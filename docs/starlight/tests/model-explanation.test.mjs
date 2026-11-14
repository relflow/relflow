import assert from 'node:assert/strict';
import test from 'node:test';
import { explainerByCase, explainerCases } from '../src/data/model-explainer.ts';
import { explain, outputAddress } from '../src/model-explainer/explanation.ts';
import { flatten } from '../src/model-explainer/schema.ts';

test('prediction addresses preserve root targets, repeated forecast slots, and claim targets', () => {
  const expected = {
    tickets: ['predictions.escalated.content.probability', 'predictions.hours_to_resolve.content'],
    churn: [
      'predictions.churn_90d.content.probability',
      'predictions.retained_revenue_90d.content',
    ],
    demand: [
      'predictions.horizon[i].units_sold.content',
      'predictions.horizon[i].stockout.content.probability',
    ],
    credit: ['predictions.default_12m.content.probability', 'predictions.loss_12m.content'],
    maintenance: [
      'predictions.failure_30d.content.probability',
      'predictions.repair_cost_30d.content',
    ],
    claims: [
      'predictions.claims[i].paid_amount.content',
      'predictions.claims[i].is_fraudulent.content.probability',
    ],
  };
  for (const model of explainerCases) {
    const full = model.stages.at(-1);
    const nodes = flatten(full.nodes);
    assert.deepEqual(
      nodes
        .filter((node) => node.target)
        .map((node) => explain(node, full, model.modelClass).address),
      expected[model.id],
    );
    for (const node of nodes.filter((node) => !node.target))
      assert.equal(
        explain(node, full, model.modelClass).address,
        '',
        `${model.id}${node.path}: context does not invent a prediction output`,
      );
  }
});

test('deep output addresses retain each repeated ancestor rather than flattening to slash keys', () => {
  const full = explainerByCase.claims.stages.at(-1);
  const item = flatten(full.nodes).find(
    (node) => node.path === '/claims/invoices/line_items/amount',
  );
  assert.ok(item);
  assert.equal(
    outputAddress(item, full.nodes),
    'predictions.claims[i].invoices[j].line_items[k].amount.content',
  );
});

test('new training-label requirements survive inspection and unsupported heads fail closed', () => {
  for (const model of explainerCases) {
    const full = model.stages.at(-1);
    for (const node of flatten(full.nodes).filter((node) => node.added)) {
      assert.ok(explain(node, full, model.modelClass).requirement);
    }
  }
  const full = explainerByCase.tickets.stages.at(-1);
  const field = flatten(full.nodes).find((node) => node.type === 'Category');
  assert.throws(
    () => explain({ ...field, target: true }, full, 'TicketModel'),
    /verified target explanation/,
  );
});
