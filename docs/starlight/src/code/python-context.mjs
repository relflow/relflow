// @ts-check

/**
 * Exclude Python strings and comments while retaining their source positions.
 * This is a lexical filter for authored snippets, not a Python type resolver.
 *
 * @param {string} source
 */
export function pythonContext(source) {
  const code = source.split('');
  let index = 0;
  while (index < source.length) {
    const start = index;
    if (source[index] === '#') {
      while (index < source.length && source[index] !== '\n') index++;
    } else if (source[index] === '"' || source[index] === "'") {
      const quote = source[index].repeat(source.startsWith(source[index].repeat(3), index) ? 3 : 1);
      index += quote.length;
      while (index < source.length) {
        if (source[index] === '\\') index += 2;
        else if (source.startsWith(quote, index)) {
          index += quote.length;
          break;
        } else index++;
      }
    } else {
      index++;
      continue;
    }
    for (let cursor = start; cursor < Math.min(index, source.length); cursor++) {
      if (source[cursor] !== '\n') code[cursor] = ' ';
    }
  }
  const masked = code.join('');
  const models = new Set(['model']);
  const constructors = new Set(['rf.Model']);
  // Focused snippets conventionally call their already-constructed instance
  // `model`. Other names need a visible declaration or model-loading assignment.
  for (const match of masked.matchAll(/\bclass\s+(\w+)\s*\(\s*(rf\.Model|\w+)\b/g)) {
    if (constructors.has(match[2])) constructors.add(match[1]);
  }
  for (const match of masked.matchAll(/\b(\w+)\s*:\s*rf\.Model\b/g)) models.add(match[1]);
  for (const match of masked.matchAll(/\b(\w+)(?:\s*:\s*[\w.]+)?\s*=\s*([\w.]+)\s*\(/g)) {
    const [, target, callable] = match;
    if (callable === 'rf.Model.load' || constructors.has(callable)) models.add(target);
    if (target === 'model' && /^[A-Z]\w*$/.test(callable)) constructors.add(callable);
  }

  /** @type {{start: number; end: number; name: string}[]} */
  const calls = [];
  /** @type {{start: number; end: number; name: string}[]} */
  const stack = [];
  let previous = '';
  for (const token of masked.matchAll(/[A-Za-z_]\w*(?:[ \t]*\.[ \t]*[A-Za-z_]\w*)*|[^\s]/g)) {
    if (token[0] === '(') {
      const name = /^[A-Za-z_]\w*(?:\s*\.\s*[A-Za-z_]\w*)*$/.test(previous)
        ? previous.replaceAll(/\s/g, '')
        : '';
      const call = { start: token.index, end: masked.length, name };
      calls.push(call);
      stack.push(call);
    } else if (token[0] === ')') {
      const call = stack.pop();
      if (call) call.end = token.index;
    }
    previous = token[0];
  }

  return {
    /** @param {number} offset */
    isCode(offset) {
      return masked[offset] !== ' ' && masked[offset] !== undefined;
    },
    /** @param {number} offset */
    callAt(offset) {
      return calls.findLast((call) => call.start < offset && call.end > offset)?.name || '';
    },
    /** @param {string} name */
    isModelReceiver(name) {
      return models.has(name) || constructors.has(name);
    },
    /** @param {string} name */
    isConstructor(name) {
      return constructors.has(name);
    },
  };
}
