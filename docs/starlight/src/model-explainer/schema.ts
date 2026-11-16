import { pythonContext } from '../code/python-context.mjs';

export interface ModelNode {
  path: string;
  name: string;
  kind: 'field' | 'branch';
  type: string;
  owner: string;
  className?: string;
  docstring?: string;
  declaration: string;
  children: ModelNode[];
  target: boolean;
  description: string;
  added?: boolean;
  labelRequirement?: string;
}

interface FieldDeclaration {
  name: string;
  declaration: string;
}

interface ClassDeclaration {
  name: string;
  kind: 'Model' | 'Branch';
  docstring: string;
  fields: FieldDeclaration[];
}

export interface ParsedSchema {
  modelClass: string;
  classes: ClassDeclaration[];
  nodes: ModelNode[];
}

/** Keep grouped field declarations together while preserving their original text and line. */
function statements(source: string): [number, string][] {
  const context = pythonContext(source);
  const result: [number, string][] = [];
  const stack: string[] = [];
  const closes: Record<string, string> = { ')': '(', ']': '[', '}': '{' };
  let offset = 0;
  let start = 0;
  let pending: string[] = [];
  for (const [index, line] of source.split('\n').entries()) {
    if (!pending.length) start = index;
    pending.push(line);
    for (let cursor = 0; cursor < line.length; cursor++) {
      const token = line[cursor];
      if (!context.isCode(offset + cursor)) continue;
      if ('([{'.includes(token)) stack.push(token);
      else if (token in closes && stack.pop() !== closes[token])
        throw new Error(`Schema has mismatched grouping at line ${index + 1}.`);
    }
    offset += line.length + 1;
    if (!stack.length) {
      result.push([start, pending.join('\n')]);
      pending = [];
    } else if (!/^    [A-Za-z_]\w* = /.test(pending[0]))
      throw new Error(`Schema only supports grouped field declarations at line ${start + 1}.`);
  }
  if (pending.length) throw new Error(`Schema field at line ${start + 1} has unclosed grouping.`);
  return result;
}

/** Parse only the explicit class declarations used by the authored showcase. */
export function parseSchema(
  source: string,
  options: { leaves?: readonly string[] } = {},
): ParsedSchema {
  const optionsLeaves = new Set(options.leaves);
  const classes: ClassDeclaration[] = [];
  let current: ClassDeclaration | undefined;
  let guarded = false;
  let constructed = '';
  let imported = false;
  for (const [index, line] of statements(source)) {
    if (!line.trim()) continue;
    // Literal labels and configuration names remain visible in the original schema.
    if (!classes.length && /^[A-Z][A-Z0-9_]* = /.test(line)) continue;
    if (line === 'import relflow as rf' && !classes.length && !imported) {
      imported = true;
      continue;
    }
    const external = /^from [\w.]+ import ([\w, ]+)$/.exec(line);
    if (
      external &&
      !classes.length &&
      external[1].split(',').every((name) => options.leaves?.includes(name.trim()))
    )
      continue;
    const declaration = /^class ([A-Za-z_]\w*)\(rf\.(Model|Branch)\):$/.exec(line);
    if (declaration && !guarded) {
      current = {
        name: declaration[1],
        kind: declaration[2] as 'Model' | 'Branch',
        docstring: '',
        fields: [],
      };
      if (classes.some((item) => item.name === current!.name))
        throw new Error(`Schema class ${current.name} is declared twice.`);
      classes.push(current);
      continue;
    }
    if (/^if __name__ == ["']__main__["']:$/.test(line) && !guarded) {
      guarded = true;
      current = undefined;
      continue;
    }
    const construction = /^    model = ([A-Za-z_]\w*)\(preset=["']small["']\)$/.exec(line);
    if (guarded && construction && !constructed) {
      constructed = construction[1];
      continue;
    }
    const docstring = /^    """([^\n]*?)"""$/.exec(line);
    if (current && docstring && !current.docstring && !current.fields.length) {
      current.docstring = docstring[1];
      continue;
    }
    const field = /^    ([A-Za-z_]\w*) = ([\s\S]+)$/.exec(line);
    if (current && field) {
      if (current.fields.some((item) => item.name === field[1]))
        throw new Error(`Schema field ${current.name}.${field[1]} is declared twice.`);
      current.fields.push({ name: field[1], declaration: field[2] });
      continue;
    }
    throw new Error(
      `Model explainer cannot describe schema line ${index + 1}: ${line}. Use explicit Model/Branch classes and field declarations.`,
    );
  }
  const roots = classes.filter((item) => item.kind === 'Model');
  if (!imported || roots.length !== 1 || (guarded && constructed !== roots[0]?.name))
    throw new Error(
      'Schema needs one Model class; an optional main guard must construct that class with the small preset.',
    );
  for (const item of classes) {
    if (!item.docstring || !item.fields.length)
      throw new Error(
        `Schema class ${item.name} needs its record docstring and explicit children.`,
      );
  }
  const definitions = new Map(classes.map((item) => [item.name, item]));

  function fields(owner: ClassDeclaration, prefix: string, ancestors: Set<string>): ModelNode[] {
    if (ancestors.has(owner.name))
      throw new Error(`Model explainer cannot expand recursive branch class ${owner.name}.`);
    const parents = new Set([...ancestors, owner.name]);
    return owner.fields.map(({ name, declaration }) => {
      const call = /^(rf\.)?([A-Za-z_]\w*)(?:\(([\s\S]*)\))?$/.exec(declaration);
      if (!call)
        throw new Error(
          `Schema field ${owner.name}.${name} must use a direct field or declared branch constructor.`,
        );
      const [, namespace, type, options = ''] = call;
      const context = pythonContext(options);
      for (const call of options.matchAll(/(?<![\w.])([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*)\s*\(/g)) {
        if (context.isCode(call.index) && !call[1].startsWith('rf.'))
          throw new Error(
            `Schema field ${owner.name}.${name} has nested configuration calls outside this explainer's literal schema contract.`,
          );
      }
      const path = `${prefix}/${name}`;
      if (namespace || (!definitions.has(type) && optionsLeaves.has(type))) {
        if (['Model', 'Branch', 'Struct'].includes(type))
          throw new Error(`Schema field ${path} needs a declared Branch class.`);
        const target = [...options.matchAll(/\bmask\s*=\s*True\b/g)].some((match) =>
          context.isCode(match.index),
        );
        return {
          path,
          name,
          kind: 'field' as const,
          type,
          owner: owner.name,
          declaration,
          children: [],
          target,
          description: '',
        };
      }
      const branch = definitions.get(type);
      if (!branch || branch.kind !== 'Branch')
        throw new Error(`Schema field ${path} references unknown branch class ${type}.`);
      return {
        path,
        name,
        kind: 'branch' as const,
        type: 'Branch',
        owner: owner.name,
        className: type,
        docstring: branch.docstring,
        declaration,
        children: fields(branch, path, parents),
        target: false,
        description: branch.docstring,
      };
    });
  }

  return { modelClass: roots[0].name, classes, nodes: fields(roots[0], '', new Set()) };
}

export function flatten(nodes: ModelNode[]): ModelNode[] {
  return nodes.flatMap((node) => [node, ...flatten(node.children)]);
}

/** Project a teaching stage without changing class order, declarations, or node placement. */
export function projectSchema(
  parsed: ParsedSchema,
  nodes: ModelNode[],
): { schema: string; lines: Record<string, number> } {
  const owners = new Map<string, ModelNode[]>([[parsed.modelClass, nodes]]);
  for (const node of flatten(nodes)) {
    if (node.kind !== 'branch') continue;
    const previous = owners.get(node.className!);
    if (
      previous &&
      JSON.stringify(previous.map(({ name, declaration }) => [name, declaration])) !==
        JSON.stringify(node.children.map(({ name, declaration }) => [name, declaration]))
    )
      throw new Error(
        `Branch class ${node.className} has incompatible teaching projections at different paths.`,
      );
    owners.set(node.className!, node.children);
  }
  const lines: Record<string, number> = {};
  const blocks: string[] = [];
  let firstLine = 4;
  for (const definition of parsed.classes) {
    const fields = owners.get(definition.name);
    if (!fields) continue;
    if (!fields.length)
      throw new Error(`Teaching schema class ${definition.name} has no remaining children.`);
    if (definition.kind === 'Model') lines['/'] = firstLine;
    const body = [
      `class ${definition.name}(rf.${definition.kind}):`,
      `    """${definition.docstring}"""`,
      '',
      ...fields.map((field) => `    ${field.name} = ${field.declaration}`),
    ];
    for (const node of flatten(nodes).filter((node) => node.owner === definition.name)) {
      const position = fields.findIndex((field) => field.name === node.name);
      if (position < 0) throw new Error(`Teaching schema lost the declaration for ${node.path}.`);
      lines[node.path] = firstLine + 3 + position;
    }
    blocks.push(body.join('\n'));
    firstLine += body.length + 2;
  }
  const guard = `if __name__ == "__main__":\n    model = ${parsed.modelClass}(preset="small")`;
  return { schema: ['import relflow as rf', ...blocks, guard].join('\n\n\n'), lines };
}
