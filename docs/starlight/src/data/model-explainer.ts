import { scenarios } from './concepts/scenarios.ts';
import type { CaseId } from './concepts/types';
import type { Dataset, DatasetColumn } from '../datasets/types';
import { flatten, parseSchema, projectSchema, type ModelNode } from '../model-explainer/schema.ts';

export type { ModelNode } from '../model-explainer/schema.ts';

export interface ModelStage {
  id: 'start' | 'context' | 'targets';
  label: string;
  title: string;
  description: string;
  schema: string;
  nodes: ModelNode[];
  lines: Record<string, number>;
  labelRequirement?: string;
}

export interface ModelCase {
  id: CaseId;
  modelClass: string;
  stages: ModelStage[];
  dataset: Dataset;
}

interface Lesson {
  observation: string;
  start: string;
  context: string;
  targets: string;
  labelRequirement: string;
  docstrings?: Record<string, string>;
  additional?: { path: string; type: 'Boolean' | 'Number'; description: string };
}

const lessons: Record<CaseId, Lesson> = {
  tickets: {
    observation: 'A support ticket at its triage cutoff.',
    start: 'Product and priority supply context for one hidden answer: whether this ticket escalates.',
    context: 'Add the supplied conversation. Each message keeps its author, text, and timing within its own record.',
    targets: 'Ask for resolution time as a second answer. The Boolean and numeric targets learn from the same ticket context.',
    labelRequirement: 'Training tickets need their completed resolution times as well as escalation labels. Both answers stay out of model inputs.',
  },
  churn: {
    observation: 'A subscriber at one prediction cutoff.',
    start: 'Plan and tenure supply context for cancellation within the next 90 days.',
    context: 'Add monthly usage and support history as separate collections, with their own record fields and length limits.',
    targets: 'Extend this example with retained revenue over the same 90-day window. One model can learn cancellation and a numeric business outcome.',
    labelRequirement: 'This additional example needs a new retained_revenue_90d training label: realized revenue during the following complete 90-day window, in the source currency units.',
    additional: { path: '/retained_revenue_90d', type: 'Number', description: 'Additional walkthrough label: realized revenue during the 90 days after the subscriber cutoff, in the source currency units. Supply complete historical outcome windows.' },
  },
  demand: {
    observation: 'A store-product forecast with supplied future weeks.',
    start: 'Supply the future weeks and known promotions that need forecasts. The hidden units target belongs to each supplied week.',
    context: 'Add observed sales history. Past units, promotions, and stockouts become context for the supplied forecast weeks.',
    targets: 'Extend each forecast week with a Boolean stockout answer alongside its numeric units forecast.',
    labelRequirement: 'This additional example needs a stockout label on each historical horizon week. Keep future stockout outcomes hidden; history.stockout remains observed past context.',
    additional: { path: '/horizon/stockout', type: 'Boolean', description: 'Additional walkthrough label: whether a stockout actually occurred during this supplied future week. Historical answer, omitted from prediction inputs; distinct from observed history.stockout.' },
  },
  credit: {
    observation: 'A loan application at assessment.',
    start: 'Application product and requested amount supply context for default within the following 12 months.',
    context: 'Add existing account records and recent inquiries. Their fields remain grouped with the records they describe.',
    targets: 'Extend the application with realized credit loss over the same 12-month horizon, alongside the default probability.',
    labelRequirement: 'This additional example needs a new loss_12m training label: realized credit loss during the complete following 12-month window, in the source currency units.',
    additional: { path: '/loss_12m', type: 'Number', description: 'Additional walkthrough label: realized credit loss during the 12 months after application assessment, in the source currency units. Supply complete historical outcome windows.' },
  },
  maintenance: {
    observation: 'An asset at its assessment cutoff.',
    start: 'Asset type and age supply context for failure during the next 30 days.',
    context: 'Add sensor readings and completed work orders. Calendar values, measurements, categories, and part sets retain their own meanings.',
    targets: 'Extend the assessment with repair cost over the following 30 days, alongside the failure probability.',
    labelRequirement: 'This additional example needs a new repair_cost_30d training label: realized repair cost during the complete following 30-day window, in the source currency units.',
    additional: { path: '/repair_cost_30d', type: 'Number', description: 'Additional walkthrough label: realized repair cost during the 30 days after the assessment cutoff, in the source currency units. Supply complete historical outcome windows.' },
  },
  claims: {
    observation: 'A policy review with its supplied claims.',
    docstrings: { Claim: 'One insurance claim submitted for review.' },
    start: 'Policy context and the supplied claims define one payout target for each claim. The enclosing claim records are necessary even in this first stage.',
    context: 'Add invoices and their line items beneath each claim. Provider identifiers, item codes, and billed amounts retain their nesting.',
    targets: 'Ask for a fraud probability beside each claim’s payout estimate. Both targets belong to the same supplied claim record.',
    labelRequirement: 'Training claims need settled payout amounts and adjudicated fraud outcomes for their respective claims. Neither masked answer is visible as input to the other.',
  },
};

/** Resolve authored source-column meanings along the observation's nested record paths. */
function columns(dataset: Dataset): Map<string, { column: DatasetColumn; entity: string }> {
  const result = new Map<string, { column: DatasetColumn; entity: string }>();
  const visited = new Set<string>();
  function visit(entity: string, prefix: string) {
    if (visited.has(entity)) throw new Error(`${dataset.id}: model walkthrough datasets must form a record tree.`);
    visited.add(entity);
    const definition = dataset.entities.find((item) => item.id === entity);
    if (!definition) throw new Error(`${dataset.id}: unknown record ${entity}.`);
    for (const column of definition.columns) result.set(`${prefix}/${column.name}`, { column, entity });
    for (const relation of dataset.relations.filter((item) => item.from === entity)) {
      if (!relation.fromColumn) throw new Error(`${dataset.id}: nested record ${relation.to} needs its source collection name.`);
      visit(relation.to, `${prefix}/${relation.fromColumn}`);
    }
  }
  visit(dataset.entities[0].id, '');
  return result;
}

export const explainerCases: ModelCase[] = scenarios.map((scenario) => {
  const lesson = lessons[scenario.id];
  const parsed = parseSchema(scenario.schema);
  // The observation is stable while staged projections add or remove context.
  const projection = {
    ...parsed,
    classes: parsed.classes.map((definition) => {
      const docstring = definition.kind === 'Model' ? lesson.observation : lesson.docstrings?.[definition.name];
      return docstring ? { ...definition, docstring } : definition;
    }),
  };
  const dataset: Dataset = {
    ...scenario.dataset,
    id: `${scenario.id}-model-walkthrough`,
    entities: scenario.dataset.entities.map((entity) => ({ ...entity, columns: entity.columns.map((column) => ({ ...column })) })),
    relations: scenario.dataset.relations.map((relation) => ({ ...relation })),
  };
  const source = columns(dataset);
  const full = parsed.nodes;
  for (const node of flatten(full)) {
    const meaning = source.get(node.path);
    if (!meaning) throw new Error(`${scenario.id}: schema field ${node.path} has no authored dataset description.`);
    node.description = (node.className && lesson.docstrings?.[node.className]) || meaning.column.description;
  }
  if (lesson.additional) {
    const { path, type, description } = lesson.additional;
    const parts = path.slice(1).split('/');
    const name = parts.pop()!;
    const parentPath = parts.length ? `/${parts.join('/')}` : '/';
    const parent = parentPath === '/' ? undefined : flatten(full).find((node) => node.path === parentPath && node.kind === 'branch');
    if (parentPath !== '/' && !parent) throw new Error(`${scenario.id}: additional target parent ${parentPath} does not exist.`);
    const children = parent ? parent.children : full;
    if (children.some((node) => node.name === name)) throw new Error(`${scenario.id}: additional target ${path} already exists.`);
    children.push({ path, name, kind: 'field', type, owner: parent?.className || parsed.modelClass, declaration: `rf.${type}(mask=True)`, children: [], target: true, description, added: true, labelRequirement: lesson.labelRequirement });
    const entity = parent ? source.get(parent.path)?.entity : dataset.entities[0].id;
    // A branch's placement column belongs to its parent; follow its relation to the child entity.
    const destination = parent ? dataset.relations.find((relation) => relation.from === entity && relation.fromColumn === parent.name)?.to : entity;
    const record = dataset.entities.find((item) => item.id === destination);
    if (!record) throw new Error(`${scenario.id}: additional target ${path} has no source record.`);
    record.columns.push({ name, type: type === 'Boolean' ? 'boolean' : 'number', role: 'target', description });
    dataset.description = `${dataset.description} The final walkthrough stage adds the explicitly marked outcome label.`;
  }

  function select(stage: ModelStage['id'], nodes: ModelNode[]): ModelNode[] {
    return nodes.flatMap((node) => {
      if (node.kind === 'field') {
        if (node.target && node.path !== scenario.primaryTarget && stage !== 'targets') return [];
        return [{ ...node, children: [] }];
      }
      if (stage === 'start' && !scenario.primaryTarget.startsWith(`${node.path}/`)) return [];
      return [{ ...node, children: select(stage, node.children) }];
    });
  }

  const stages: ModelStage[] = (['start', 'context', 'targets'] as const).map((id, index) => {
    const nodes = select(id, full);
    return {
      id,
      label: ['One answer', 'Add context', 'Add an answer'][index],
      title: ['Start with one prediction', 'Keep the related records', 'Learn two answers together'][index],
      description: lesson[id],
      nodes,
      ...projectSchema(projection, nodes),
      ...(id === 'targets' ? { labelRequirement: lesson.labelRequirement } : {}),
    };
  });
  return { id: scenario.id, modelClass: scenario.modelClass, stages, dataset };
});

export const explainerByCase = Object.fromEntries(explainerCases.map((item) => [item.id, item])) as Record<CaseId, ModelCase>;
