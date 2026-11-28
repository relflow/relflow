import type { Concept } from '../concepts/types';
import { scenarios } from '../concepts/scenarios';

const examples: Record<(typeof scenarios)[number]['id'], Omit<Concept['examples'][number], 'caseId'>> = {
  tickets: {
    title: 'Inspect targets, then expand the conversation cap',
    description: 'Select nodes from an existing TicketModel and update the message branch through its absolute schema address. The selector sees the resolved schema, including the two inherited targets.',
    blocks: [{
      label: 'Select before making a persistent edit', lang: 'python',
      code: `# model is an existing TicketModel; no model loop is active.
targets = model.select(rf.where("reconstruct"))
target_paths = [str(node.address) for node in targets]

messages = rf.where("address") == rf.Address("messages")
model.update(messages, length=96, overflow="tail")`,
      note: 'The target addresses are /escalated and /hours_to_resolve. A larger cap only supplies additional context when the source provides more messages; it does not recover rows already removed upstream.',
    }],
    takeaway: 'Use select for a read-only inspection and update for a lasting configuration change. Reevaluate the new context policy even when learned weights remain compatible.',
  },
  churn: {
    title: 'Add a second subscriber-level target',
    description: 'Extend one fitted ChurnModel at its root to predict downgrading as well as cancellation. Existing compatible state can be reused, while the new head starts without learned task-specific behavior.',
    blocks: [{
      label: 'Add a target under the anonymous root', lang: 'python',
      code: `# model is an existing ChurnModel; no model loop is active.
root = rf.where("address") == rf.Address()
model.extend(root, downgraded_90d=rf.Boolean(predict=True))`,
      note: 'The next training data must supply downgraded_90d answers for the intended 90-day definition, alongside the existing inputs and churn_90d answers. Train the new target before using its predictions.',
    }],
    takeaway: 'An added output is a new learning task. Retaining compatible parts of the old model does not train its new prediction head.',
  },
  demand: {
    title: 'Compare shorter history without changing forecast slots',
    description: 'Temporarily change the fitted DemandModel’s history limit while keeping the supplied horizon and its targets intact. Predict on the same held-out request table inside and outside the context.',
    blocks: [{
      label: 'Override history policy for one complete prediction operation', lang: 'python',
      code: `# model is a fitted DemandModel; inputs is an Arrow request table.
history = rf.where("address") == rf.Address("history")

with model.override(history, length=52, overflow="tail"):
    shorter_history = model.predict(inputs, batch_size=256)

ordinary_history = model.predict(inputs, batch_size=256)`,
      note: 'The original 104-row cap returns on exit, including exceptional exit. The shared illustration has one historical week; this comparison needs a request set with longer histories to exercise the different limits.',
    }],
    takeaway: 'Override restores schema attributes. It is not a checkpoint rollback: compatible learned-state changes made inside a context are not generally undone.',
  },
  credit: {
    title: 'Remove the inquiry source from the model',
    description: 'Delete the entire inquiries subtree from an existing CreditModel while retaining the application fields, account context, and default target. This changes the information available to prediction.',
    blocks: [{
      label: 'Delete a selected branch and its children', lang: 'python',
      code: `# model is an existing CreditModel; no model loop is active.
inquiries = rf.where("address") == rf.Address("inquiries")
selected = model.select(inquiries)
model.delete(inquiries)`,
      note: 'Deletion removes /inquiries and its lender and age_days children. Prepare data and any application contracts for the changed schema, then evaluate or adapt the model using the remaining information.',
    }],
    takeaway: 'Deleting an input can change predictions immediately. A valid rebuild and retained compatible parameters do not establish equivalent predictive behavior.',
  },
  maintenance: {
    title: 'Restart only the failure prediction node',
    description: 'Reinitialize the failure_30d runtime node for a fresh training experiment while leaving its field declaration and the other model nodes in place.',
    blocks: [{
      label: 'Keep the schema and reset selected learned state', lang: 'python',
      code: `# model is an existing MaintenanceModel; no model loop is active.
failure = rf.where("address") == rf.Address("failure_30d")
model.reset(failure)`,
      note: 'The selected node’s runtime state is discarded. Train it again with failure_30d answers before relying on its output. Resetting a branch includes its descendants only when descendants=True is supplied.',
    }],
    takeaway: 'reset is deliberate reinitialization, not a harmless metadata edit. Save a checkpoint first if you need to recover the old learned state.',
  },
  claims: {
    title: 'Pause provider identity inside invoice context',
    description: 'Deactivate only the provider Hash field in an existing ClaimsModel. Invoice line items, their billed amounts, and both per-claim targets remain active.',
    blocks: [{
      label: 'Select the provider leaf by its absolute path', lang: 'python',
      code: `# model is an existing ClaimsModel; no model loop is active.
provider = rf.where("address") == rf.Address("claims", "invoices", "provider")
selected = model.select(provider)
model.update(provider, active=False)`,
      note: 'This persistent identity ablation leaves invoice records and line items intact. Each supplied claim still requests paid_amount and is_fraudulent predictions, now without provider identity. Restoring active=True is another schema edit, not a restoration of a saved checkpoint.',
    }],
    takeaway: 'Select the leaf whose input you want to pause: active is a leaf option, not a branch option. Measure how removing provider identity changes behavior before using the edited artifact.',
  },
};

export const mutations: Concept = {
  id: 'mutations',
  title: 'Mutations',
  description: 'Select, update, extend, delete, reset, and temporarily override a model schema while rebuilding its runtime graph.',
  intro: [
    'relflow empowers enterprise teams to create centralized, pretrained models and then for individual teams to create versions which are suited for their own data and predictions tasks. The key mechanism for this, mutations, allows each team to add data or targets, or even to remove the reliance on previously trained-to data entirely; all of this is possible while relflow models retain all compatible learned states.',
    'The following examples start with an instance of the selected case’s model. Make edits between complete operations: mutation is prohibited during active forward, inference, or Lightning Trainer loops. Each block is an independent example, rather than a sequence of edits to one shared instance.',
  ],
  principles: [
    { title: 'Select exact schema nodes', text: 'rf.Address() denotes the anonymous root; rf.Address("claims", "invoices") denotes /claims/invoices. Combine rf.where predicates or inspect model.select(...) before a broad edit. Updates are strict by default, so every selected node must support each supplied option.' },
    { title: 'Retain compatible state, then check behavior', text: 'Graph rebuilds retain compatible learned state; new or incompatible components need initialization. An edit can still change context, routing, or outputs. New prediction targets and reset nodes require training, and state retention alone does not promise unchanged predictions.' },
    { title: 'Distinguish a temporary edit from a checkpoint', text: 'Override restores the previous schema attributes on exit. It does not snapshot every weight, vocabulary, or normalizer. Use a saved checkpoint when the requirement is to recover an earlier trained artifact.' },
    { title: 'Start optimization on the updated parameters', text: 'After an edit, prepare data for the updated schema and begin a new fit when adaptation is needed. An old custom optimizer can reference replaced modules. Mutations also clear compilation and cached output contracts; rebuild the deployment artifact you intend to use.' },
  ],
  examples: scenarios.map(({ id }) => ({ caseId: id, ...examples[id] })),
  takeaway: 'Schema changes are explicit model-development operations. Inspect what matched, understand which state was kept or replaced, and evaluate the resulting model as a changed artifact.',
  links: [
    { label: 'Checkpoint the model before an edit', href: '/advanced/checkpoints-and-state/' },
    { label: 'Adapt reconstruction objectives', href: '/advanced/masking-and-pretraining/' },
    { label: 'Mutation API methods', href: '/api-map/#api-mutation' },
    { label: 'Schema address helper', href: '/api-map/#api-rf-address' },
    { label: 'Define a model', href: '/concepts/models/' },
    { label: 'Training and continuation', href: '/advanced/training/' },
    { label: 'Recorded mutation experiments', href: '/evidence/?family=Mutations' },
  ],
};
