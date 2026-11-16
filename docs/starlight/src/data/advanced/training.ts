import { scenarios } from '../concepts/scenarios';
import type { CaseId, Concept } from '../concepts/types';

const learning: Record<CaseId, { title: string; description: string; fit: string; inspect: string; stateNote: string; takeaway: string }> = {
  tickets: {
    title: 'Fit the ticket tasks and keep the learned field state',
    description: 'Constructing TicketModel selects the architecture. Fitting teaches its escalation and resolution objectives from the available conversation, while learning field state from training data.',
    fit: 'model.fit(data, preset="small")',
    inspect: `products = rf.Category.vocabulary(model, "/product")
duration_statistics = rf.Number.normalization(model, "/hours_to_resolve")`,
    stateNote: 'The product vocabulary and the hidden duration target’s numeric statistics are learned state. The resolution statistics use historical labels without embedding those labels as context.',
    takeaway: 'The Text encoder remains frozen; relflow learns how to use its representations with the other ticket fields. Keep the text dependencies and encoder assets available when loading the saved model.',
  },
  churn: {
    title: 'Choose architecture and fit defaults separately',
    description: 'The constructor’s small preset chooses the network. Selecting a fit preset separately chooses an optimizer and training policy without changing the subscriber schema or its labels.',
    fit: 'model.fit(data, preset="small")',
    inspect: `plans = rf.Category.vocabulary(model, "/plan")
usage_statistics = rf.Number.normalization(model, "/monthly_usage/minutes")`,
    stateNote: 'Validation and test reuse the training vocabulary and numeric statistics. They do not admit evaluation plan labels or refit normalization to the evaluation population.',
    takeaway: 'The bundled fit recipe uses eight epochs and EMA, with early stopping and rollback disabled unless you override them. Test cancellation risk on held-out subscriber decisions whose outcome windows are complete.',
  },
  demand: {
    title: 'Learn separate numeric state for history and horizon targets',
    description: 'Historical sales provide context, while units on each supplied horizon week provide supervision. Fitting updates one network across these observation rows and target positions.',
    fit: 'model.fit(data, preset="small")',
    inspect: `history_statistics = rf.Number.normalization(model, "/history/units_sold")
target_statistics = rf.Number.normalization(model, "/horizon/units_sold")`,
    stateNote: 'The two units_sold fields have different addresses and their own statistics. A shared leaf name does not merge their training state.',
    takeaway: 'A test row still contains the forecast dates and planned promotions that were known at its forecast origin. Supply observed units only as hidden labels; score the horizons that matter to the planning decision.',
  },
  credit: {
    title: 'Evaluate a trained risk model on held-out applications',
    description: 'The default label supplies supervision; account and inquiry histories supply context. Fitting can select a model with validation data, and test measures its behavior on a separate application population.',
    fit: 'model.fit(data, preset="small")',
    inspect: `account_types = rf.Category.vocabulary(model, "/accounts/account_type")
amount_statistics = rf.Number.normalization(model, "/requested_amount")`,
    stateNote: 'Inspect the training categories and requested-amount scale. A saved model restores this state; constructing CreditModel again would create a new network and fresh field state.',
    takeaway: 'Default fit monitoring uses validation loss and restores the selected best model when rollback is enabled. Evaluate the score and the application policy separately on the intended population; validation selection is not a test result.',
  },
  maintenance: {
    title: 'Train from asset snapshots, then reuse the learned encodings',
    description: 'Learning updates the network and field-owned resources from training snapshots. Validation and test evaluate failure risk using the state learned from those snapshots.',
    fit: 'model.fit(data, preset="small")',
    inspect: `vibration_statistics = rf.Number.normalization(model, "/sensor_readings/vibration")
known_parts = rf.Set.vocabulary(model, "/work_orders/parts_replaced")`,
    stateNote: 'Numeric normalization and the parts vocabulary travel with the weights. An empty observed parts set remains different from a missing work-order collection.',
    takeaway: 'A second ordinary fit starts a new optimizer run from the current model state. To continue optimizer, scheduler, and Trainer progress, supply a full Lightning checkpoint through ckpt_path.',
  },
  claims: {
    title: 'Train and evaluate payout and fraud at claim depth',
    description: 'Each supplied claim has two hidden objectives. Both train with policy and invoice context, while retaining their own answer types and evaluation meaning.',
    fit: 'model.fit(data, preset="small")',
    inspect: `causes = rf.Category.vocabulary(model, "/claims/cause")
payout_statistics = rf.Number.normalization(model, "/claims/paid_amount")`,
    stateNote: 'The payout statistics summarize the claim-level training labels. Hash provider identifiers do not build a persistent provider vocabulary.',
    takeaway: 'Read payout errors in the label’s currency units and evaluate fraud probabilities separately. One joint validation loss does not replace checking both outcomes on held-out policy reviews.',
  },
};

export const training: Concept = {
  id: 'training',
  title: 'Training and learned state',
  description: 'Distinguish construction, fitting, evaluation, model artifacts, and full training-state resume.',
  intro: [
    'A schema constructs an untrained network. Fitting changes its parameters and field-owned state using historical observations and hidden answers. Prediction uses that fitted state on new observations.',
    'The examples assume the selected class definition and application-prepared labeled splits. They illustrate API calls, not measured results. The saved-model step follows fitting; constructing a schema alone does not produce a useful trained predictor.',
  ],
  principles: [
    {
      title: 'Learning includes more than weights',
      text: 'Field-owned state includes learned vocabularies and numeric normalizers. Training updates these resources as well as neural parameters. Validation, test, and prediction reuse their fitted state instead of learning an evaluation vocabulary or distribution.',
    },
    {
      title: 'Fitting and evaluation have different jobs',
      text: 'fit learns and can use validation metrics for model selection. validate and test evaluate their corresponding splits without training the model. Reserve test for the population and decisions you intend to report, with labels unavailable as inputs.',
    },
    {
      title: 'Architecture and fit presets are independent',
      text: 'The constructor’s preset chooses architecture defaults. fit(preset=...) independently selects optional training defaults. Explicit settings win. The bundled fit recipes use eight epochs and EMA with early stopping and rollback off; ordinary fit defaults enable early stopping and rollback using validation loss.',
    },
    {
      title: 'Saving a model is different from resuming a run',
      text: 'model.save and Model.load preserve the schema, parameters, field state, and provenance. A second ordinary fit starts a new optimizer run. Continuing optimizer, scheduler, and Trainer progress requires a full Lightning checkpoint passed to fit with ckpt_path; a model.save artifact does not contain that full state.',
    },
    {
      title: 'Some resources stay outside the artifact',
      text: 'Recreate data sources and processors when loading a model, and make required external text-encoder assets available. Compilation is transient. A schema mutation can retain compatible learned state, but a newly added target still needs its own training answers and evaluation.',
    },
  ],
  examples: scenarios.map((scenario) => ({
    caseId: scenario.id,
    title: learning[scenario.id].title,
    description: learning[scenario.id].description,
    blocks: [
      {
        label: 'Construct, fit, and evaluate on prepared splits',
        lang: 'python',
        code: `import relflow as rf

model = ${scenario.modelClass}(preset="small")
data = rf.ArrowDataModule(
    train="${scenario.id}/train.parquet",
    validate="${scenario.id}/validation.parquet",
    test="${scenario.id}/test.parquet",
)
${learning[scenario.id].fit}
validation_metrics = model.validate(data)
test_metrics = model.test(data)`,
        note: 'Use the selected class definition. fit already evaluates validation data while training; the separate validate call shows how to evaluate fitted state explicitly. Returned metrics are not filled in by this illustration.',
      },
      {
        label: 'Inspect fitted field state',
        lang: 'python',
        code: learning[scenario.id].inspect,
        note: learning[scenario.id].stateNote,
      },
      {
        label: 'Save the fitted model and score new records after loading',
        lang: 'python',
        code: `model.save("artifacts/${scenario.id}.rf")
trained = rf.Model.load("artifacts/${scenario.id}.rf")
requests = rf.ArrowDataModule(
    predict="${scenario.id}/requests.parquet",
)
result = trained.predict(
    datamodule=requests,
    batch_size=128,
    devices=1,
)`,
        note: 'Prediction records omit every target answer. A fresh data module belongs to the loaded model; result is an eager Polars frame. This loads model state for prediction and does not resume the earlier optimizer run.',
      },
      ...(scenario.id === 'maintenance' ? [{
        label: 'Resume a run from an existing full Lightning checkpoint',
        lang: 'python' as const,
        code: `checkpoint = "checkpoints/maintenance/resume.ckpt"
resumed = rf.Model.load(checkpoint)
resume_data = rf.ArrowDataModule(
    train="maintenance/train.parquet",
    validate="maintenance/validation.parquet",
)
resumed.fit(
    resume_data,
    max_epochs=30,
    ckpt_path=checkpoint,
)`,
        note: 'The original run must have supplied this full Lightning checkpoint; model.save does not create optimizer or Trainer state. max_epochs is the run’s total epoch ceiling, including progress already recorded in the checkpoint.',
      }] : []),
    ],
    takeaway: learning[scenario.id].takeaway,
  })),
  takeaway: 'Keep architecture configuration, fitted model state, and optimizer-run state distinct. Validate the model you will actually use, and preserve its data boundaries and resource requirements when loading it elsewhere.',
  links: [
    { label: 'Resume checkpoints and version learned state', href: '/advanced/checkpoints-and-state/' },
    { label: 'Scale training and streaming', href: '/advanced/distributed-data/' },
    { label: 'Manage an online vocabulary', href: '/advanced/online-vocabulary/' },
    { label: 'Inspect training, persistence, and mutation APIs', href: '/api-map/' },
    { label: 'Read prediction types and nested output', href: '/concepts/runtime/' },
    { label: 'Review measured protocols and their limits', href: '/evidence/' },
  ],
};
