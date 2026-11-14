// @ts-check

/**
 * @typedef {{id: string; schema: string; lines: Record<string, number>}} Stage
 * @typedef {{stages: string[]; paths: string[]}} ModelLine
 */

/**
 * Map staged schemas onto one highlighted source block without duplicating its DOM.
 * Each smaller stage must retain the complete schema's source order and text.
 * @param {Stage[]} stages
 * @returns {ModelLine[]}
 */
export function codePlan(stages) {
  if (!stages.length) throw new Error('Model explainer needs at least one staged schema.');
  const identifiers = new Set();
  for (const stage of stages) {
    if (!/^[a-z][a-z0-9-]*$/.test(stage.id) || identifiers.has(stage.id)) {
      throw new Error(
        `Model stage ${stage.id}: stage identifiers must be unique lowercase tokens.`,
      );
    }
    if (typeof stage.schema !== 'string') {
      throw new Error(`Model stage ${stage.id}: schema must contain source text.`);
    }
    identifiers.add(stage.id);
  }
  const complete = stages[stages.length - 1];
  const source = complete.schema.split('\n');
  const plan = source.map(() => ({
    stages: /** @type {string[]} */ ([]),
    paths: /** @type {string[]} */ ([]),
  }));
  for (const stage of stages) {
    const lines = stage.schema.split('\n');
    let offset = 0;
    const positions = lines.map((line, index) => {
      while (offset < source.length && source[offset] !== line) offset++;
      if (offset === source.length) {
        throw new Error(
          `Model stage ${stage.id}: source line ${index + 1} is not an ordered subset of the complete schema. Preserve declaration text and source order.`,
        );
      }
      plan[offset].stages.push(stage.id);
      return offset++;
    });
    for (const [path, line] of Object.entries(stage.lines)) {
      if (
        !path.startsWith('/') ||
        /\s/.test(path) ||
        !Number.isInteger(line) ||
        line < 1 ||
        line > lines.length
      ) {
        throw new Error(
          `Model stage ${stage.id}: ${path} must name a schema address and a source line from 1 to ${lines.length}; received ${line}.`,
        );
      }
      const completeLine = complete.lines[path];
      if (completeLine === undefined || positions[line - 1] !== completeLine - 1) {
        throw new Error(
          `Model stage ${stage.id}: ${path} does not retain its declaration in the complete schema. Rebuild the stage's line map.`,
        );
      }
    }
  }
  for (const [path, line] of Object.entries(complete.lines)) plan[line - 1].paths.push(path);
  return plan;
}

/** @param {unknown} value @returns {value is ModelLine} */
function modelLine(value) {
  return (
    !!value &&
    typeof value === 'object' &&
    'stages' in value &&
    Array.isArray(value.stages) &&
    value.stages.length > 0 &&
    value.stages.every((stage) => typeof stage === 'string' && /^[a-z][a-z0-9-]*$/.test(stage)) &&
    new Set(value.stages).size === value.stages.length &&
    'paths' in value &&
    Array.isArray(value.paths) &&
    value.paths.every(
      (path) => typeof path === 'string' && path.startsWith('/') && !/\s/.test(path),
    ) &&
    new Set(value.paths).size === value.paths.length
  );
}

/**
 * Attach the authored stage plan after ordinary syntax highlighting and code notes.
 * @returns {import('@expressive-code/core').ExpressiveCodePlugin}
 */
export function modelCode() {
  /** @type {WeakMap<import('@expressive-code/core').ExpressiveCodeBlock, ModelLine[]>} */
  const plans = new WeakMap();
  return {
    name: 'relflow model declaration map',
    hooks: {
      annotateCode({ codeBlock }) {
        const metadata = codeBlock.metaOptions.getString('model-lines');
        if (metadata === undefined) return;
        /** @type {unknown} */
        let plan;
        try {
          plan = JSON.parse(metadata);
        } catch (error) {
          throw new Error('Model source line metadata must be valid JSON.', { cause: error });
        }
        if (
          !['python', 'py'].includes(codeBlock.language) ||
          !Array.isArray(plan) ||
          plan.length !== codeBlock.getLines().length ||
          !plan.every(modelLine)
        ) {
          throw new Error(
            'Model source metadata requires one valid stage/path entry per Python source line. Generate it with codePlan.',
          );
        }
        plans.set(codeBlock, plan);
      },
      postprocessRenderedLine({ codeBlock, lineIndex, renderData }) {
        const line = plans.get(codeBlock)?.[lineIndex];
        if (!line) return;
        renderData.lineAst.properties['data-model-stages'] = line.stages.join(' ');
        if (line.paths.length)
          renderData.lineAst.properties['data-model-paths'] = line.paths.join(' ');
      },
    },
  };
}
