import { predictions } from './predictions';
import { scenarios } from './scenarios';
import type { Concept } from './types';

export const runtime: Concept = {
  id: 'runtime',
  title: 'Runtime',
  description: 'Fit with a preset, evaluate held-out splits, and predict new records.',
  intro: [],
  principles: [
    {
      title: 'Presets choose defaults; the schema defines the task',
      text: 'Construction applies architecture defaults to the declared fields and branches. The fit preset independently supplies the training recipe. Neither choice adds targets or changes which answers are hidden from the inputs.',
    },
    {
      title: 'Keep validation and test separate',
      text: 'Use validation metrics when comparing presets or other model choices. Keep the test split for the final evaluation. These calls evaluate the fitted model without training it or learning vocabularies and normalization from evaluation records.',
    },
    {
      title: 'Learned state travels with the model',
      text: 'Fitting updates neural parameters and field-owned state such as vocabularies and numeric normalizers. Validation, test, and prediction reuse that fitted state. Constructing the class again creates a new, untrained model.',
    },
    ...predictions.principles,
  ],
  examples: scenarios.map((scenario): Concept['examples'][number] => {
    const illustration = predictions.examples.find((example) => example.caseId === scenario.id);
    if (!illustration)
      throw new Error(`Runtime needs the shared prediction illustration for ${scenario.id}.`);
    return {
      caseId: scenario.id,
      title: `Fit, evaluate, and predict for ${scenario.label.toLowerCase()}`,
      description: `Use the ${scenario.modelClass} schema with prepared historical splits and a separate source of new request records.`,
      blocks: [
        {
          label: 'Run the preset workflow',
          lang: 'python',
          code: `import relflow as rf

# Use the ${scenario.modelClass} class schema shown below.
if __name__ == "__main__":
    data = rf.ArrowDataModule(
        train="${scenario.id}/train.parquet",
        validate="${scenario.id}/validation.parquet",
        test="${scenario.id}/test.parquet",
        predict="${scenario.id}/requests.parquet",
    )

    model = ${scenario.modelClass}(preset="small")
    model.fit(data, preset="small")

    validation_metrics = model.validate(data)
    test_metrics = model.test(data)
    results = model.predict(data, devices=1)`,
          note: 'The paths name application-prepared files; each row keeps its related records together. The model methods bind the data module automatically. devices=1 collects prediction rows into one Polars DataFrame.',
        },
        ...illustration.blocks.filter((block) => block.kind === 'prediction'),
      ],
      takeaway: illustration.takeaway,
    };
  }),
  takeaway: 'Choose a preset, supply meaningful data splits, and use the same fitted model through evaluation and prediction. Read each answer through its declared type and preserve its record structure.',
  links: [
    { label: 'Choose what the model learns', href: '/concepts/targets/' },
    { label: 'Read from files, queries, or custom sources', href: '/advanced/data-modules/' },
    { label: 'Understand training and learned state', href: '/advanced/training/' },
    { label: 'Save models and resume training runs', href: '/advanced/checkpoints-and-state/' },
    { label: 'Stream predictions or serve a model', href: '/advanced/inference-and-deployment/' },
    { label: 'Inspect the runtime API', href: '/api-map/?section=training' },
  ],
};
