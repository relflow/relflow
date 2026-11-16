// @ts-check
import { pythonContext } from './python-context.mjs';
import { reference } from '../reference.mjs';

/**
 * @typedef {ReturnType<typeof pythonContext>} PythonContext
 * @typedef {RegExpExecArray} Match
 * @typedef {{ pattern: RegExp, capture?: number, field?: (match: Match) => string | undefined,
 *   note: import('../reference.mjs').Note | ((match: Match, context?: PythonContext, offset?: number) => import('../reference.mjs').Note),
 *   accept?: (match: Match, context: PythonContext, offset: number) => boolean }} Rule
 */

/** @type {Rule[]} */
const pythonRules = [
  ...Object.entries(reference.methods).map(([name, note]) => ({
    pattern: new RegExp(`\\brf\\.${name.replaceAll('.', '\\.')}\\b(?=\\s*\\()`, 'g'),
    note,
  })),
  ...Object.entries(reference.types).map(([name, note]) => ({
    pattern: new RegExp(`\\brf\\.${name}\\b(?!\\.[A-Za-z_]\\w*\\s*\\()`, 'g'),
    note,
  })),
  {
    pattern: /\bmask\s*=\s*True\b/g,
    note: reference.options.target,
  },
  {
    pattern: /\blength\s*=\s*(\d+)\b/g,
    note: (match) => reference.options.length({ limit: match[1] }),
  },
  {
    pattern: /\boverflow\s*=\s*["']tail["']/g,
    note: reference.options.overflowTail,
  },
  {
    pattern: /\boverflow\s*=\s*["']error["']/g,
    note: reference.options.overflowError,
  },
  {
    pattern: /\bpreset\s*=\s*["'](?:small|medium)["']/g,
    accept: (_match, context, offset) =>
      context.isConstructor(context.callAt(offset)) ||
      (context.callAt(offset).endsWith('.fit') &&
        context.isModelReceiver(context.callAt(offset).slice(0, -4))),
    note: (_match, context, offset) =>
      context?.callAt(offset ?? 0).endsWith('.fit')
        ? reference.options.fitPreset
        : reference.options.architecturePreset,
  },
  {
    pattern: /\bembed\s*=\s*True\b/g,
    note: reference.options.embed,
  },
  {
    pattern: /\bn_dim\s*=\s*(\d+)\b/g,
    note: (match) => reference.options.vectorWidth({ width: match[1] }),
  },
  {
    pattern: /\bstrip\s*=\s*True\b/g,
    note: reference.options.strip,
  },
  {
    pattern: /\bcased\s*=\s*False\b/g,
    note: reference.options.uncased,
  },
  {
    pattern: /\bthreshold\s*=\s*(?:\[[^\]]*\]|None\b|0?\.\d+\b|[01]\b)/g,
    note: reference.options.threshold,
  },
  {
    pattern: /\btopk\s*=\s*\[[^\]]*\]/g,
    note: reference.options.topk,
  },
  {
    pattern: /\bobjective\s*=\s*["'](?:mae|mse|huber|l1|l2)["']/g,
    note: reference.options.objective,
  },
  {
    pattern: /\bmax_length\s*=\s*(\d+)\b/g,
    note: (match) => reference.options.maxLength({ tokens: match[1] }),
  },
  {
    pattern: /\bstore\s*=\s*["'][^"']+["']/g,
    accept: (_match, context, offset) => context.callAt(offset) === 'rf.Text',
    note: reference.options.store,
  },
  {
    pattern: /\bcache\s*=\s*(?:True|False|\d+)\b/g,
    accept: (_match, context, offset) => context.callAt(offset) === 'rf.Text',
    note: reference.options.cache,
  },
  {
    pattern: /\bmemory\s*=\s*(?:None|\d+)\b/g,
    accept: (_match, context, offset) => context.callAt(offset) === 'rf.Text',
    note: reference.options.memory,
  },
  {
    pattern: /\btokenizer_batch_size\s*=\s*\d+\b/g,
    accept: (_match, context, offset) => context.callAt(offset) === 'rf.Text',
    note: reference.options.tokenizerBatchSize,
  },
  {
    pattern: /\bresident\s*=\s*(?:None|\d+)\b/g,
    accept: (_match, context, offset) => context.callAt(offset) === 'rf.Text',
    note: reference.options.resident,
  },
  {
    pattern: /\bformat\s*=\s*["'](?:parquet|ipc)["']/g,
    accept: (_match, context, offset) => context.callAt(offset) === 'rf.Writer',
    note: reference.options.format,
  },
  {
    pattern: /\bn_groups\s*=\s*(\d+)\b/g,
    note: (match) => reference.options.groupCount({ count: match[1] }),
  },
  {
    pattern: /\bgranularity\s*=\s*["'](?:seconds|minutes|hours|days|weeks)["']/g,
    note: reference.options.granularity,
  },
  {
    pattern: /\bquery(?=\s*=)/g,
    note: reference.options.query,
  },
  ...[
    /\bquery\s*=\s*["']([^"']+)["']/g,
    /\bquery\s*=\s*\(\s*["']([^"']+)["']/g,
    /\bquery\s*=\s*\(\s*["'][^"']+["']\s*,\s*["']([^"']+)["']/g,
  ].map((pattern) => ({
    pattern,
    capture: 1,
    /** @param {Match} match */
    field: (match) =>
      match[1]
        .replace(/^(?:\.\.\/)+/, '')
        .split('.')
        .at(-1),
    /** @param {Match} match */
    note: (match) => reference.fields.source({ path: match[1] }),
  })),
  {
    pattern: /\b(?:start|end)\s*=\s*["'][^"']+["']/g,
    note: reference.options.endpoint,
  },
  {
    pattern: /\bvalues\s*=\s*\([^)]*\)/g,
    note: reference.options.values,
  },
  {
    pattern: /\bpattern\s*=\s*["'][^"']+["']/g,
    note: reference.options.pattern,
  },
  {
    pattern: /\breconstruct\s*=\s*True\b/g,
    note: reference.options.reconstruct,
  },
  {
    pattern: /\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)(\.(?:fit|predict|save|load))(?=\s*\()/g,
    capture: 2,
    accept: (match, context) => context.isModelReceiver(match[1]),
    note: (match) => reference.workflows[match[2].slice(1)],
  },
  {
    pattern: /\bbatch_size\s*=\s*(\d+)\b/g,
    note: (match) => reference.options.batchSize({ size: match[1] }),
  },
  {
    pattern: /\bretain\s*=\s*\([^)]*\)/g,
    note: reference.options.retain,
  },
];

/** @type {Rule[]} */
const shellRules = [
  {
    pattern: /\buv run\b/g,
    note: reference.syntax.uvRun,
  },
  {
    pattern: /relflow\[text\]/g,
    note: reference.syntax.textDependencies,
  },
];

/** @type {Rule[]} */
const jsonRules = [
  {
    pattern: /"[^"]+"\s*:\s*\[/g,
    note: reference.syntax.list,
  },
];

/**
 * Shared, authored explanations applied automatically to every code example.
 * @returns {import('@expressive-code/core').ExpressiveCodePlugin}
 */
export function configNotes() {
  return {
    name: 'relflow configuration notes',
    hooks: {
      annotateCode({ codeBlock }) {
        const contract = codeBlock.metaOptions.getString('contract') || '';
        const lines = codeBlock.getLines();
        const python = ['python', 'py'].includes(codeBlock.language)
          ? pythonContext(lines.map((line) => line.text).join('\n'))
          : undefined;
        let rules = ['python', 'py'].includes(codeBlock.language)
          ? pythonRules
          : ['bash', 'sh', 'shell'].includes(codeBlock.language)
            ? shellRules
            : ['json', 'jsonc'].includes(codeBlock.language)
              ? jsonRules
              : [];
        if (contract === 'schema' && ['python', 'py'].includes(codeBlock.language)) {
          rules = [
            ...rules,
            {
              pattern: /^\s*([A-Za-z_]\w*)\s*=/g,
              capture: 1,
              field: (match) => match[1],
              /** @param {Match} match */
              note: (match) => reference.fields.schema({ field: match[1] }),
            },
          ];
        } else if (
          reference.contracts[contract] &&
          ['json', 'jsonc', 'yaml', 'yml'].includes(codeBlock.language)
        ) {
          const note = reference.contracts[contract];
          const yaml = ['yaml', 'yml'].includes(codeBlock.language);
          /** @param {Match} match */
          const key = (match) => {
            const raw = match[1];
            if (!yaml) return JSON.parse(`"${raw}"`);
            if (raw.startsWith('"')) return JSON.parse(raw);
            return raw.startsWith("'") ? raw.slice(1, -1).replaceAll("''", "'") : raw;
          };
          rules = [
            {
              pattern: yaml
                ? /^\s*(?:-\s+)?("(?:[^"\\]|\\.)*"|'(?:[^']|'')*'|[A-Za-z_][\w.-]*)(?=\s*:)/g
                : /"([^"\\]*(?:\\.[^"\\]*)*)"(?=\s*:)/g,
              ...(yaml ? { capture: 1 } : {}),
              ...(contract === 'predictions'
                ? {
                    note: (match) =>
                      reference.outputs[key(match)] ??
                      reference.fields.prediction({ field: key(match) }),
                  }
                : { field: key, note }),
            },
          ];
        }
        let lineOffset = 0;
        for (const line of lines) {
          for (const rule of rules) {
            for (const match of line.text.matchAll(rule.pattern)) {
              const offset = lineOffset + match.index + match[0].search(/\S/);
              if (
                python &&
                (!python.isCode(offset) || (rule.accept && !rule.accept(match, python, offset)))
              )
                continue;
              const { title, text } =
                typeof rule.note === 'function' ? rule.note(match, python, offset) : rule.note;
              const value = rule.capture ? match[rule.capture] : match[0];
              const start = match.index + (rule.capture ? match[0].lastIndexOf(value) : 0);
              line.addAnnotation({
                name: 'Configuration note',
                renderPhase: 'latest',
                inlineRange: { columnStart: start, columnEnd: start + value.length },
                render({ nodesToTransform }) {
                  // Nested syntax can split a range; preserve each segment's position.
                  return nodesToTransform.map((node) => ({
                    type: 'element',
                    tagName: 'span',
                    properties: {
                      className: ['rf-code-note'],
                      'data-note-title': title,
                      'data-note-text': text,
                      'data-contract-field': rule.field?.(match),
                      title: `${title}: ${text}`,
                    },
                    children:
                      node.type === 'root'
                        ? node.children.filter(
                            (child) => child.type !== 'doctype' && child.type !== 'mdxjsEsm',
                          )
                        : [node],
                  }));
                },
              });
            }
          }
          lineOffset += line.text.length + 1;
        }
      },
    },
  };
}
