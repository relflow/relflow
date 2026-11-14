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

/** Parse only the explicit class declarations used by the authored showcase. */
export function parseSchema(source: string): ParsedSchema {
  const classes: ClassDeclaration[] = [];
  let current: ClassDeclaration | undefined;
  let guarded = false;
  let constructed = '';
  let imported = false;
  for (const [index, line] of source.split('\n').entries()) {
    if (!line.trim()) continue;
    if (line === 'import relflow as rf' && !classes.length && !imported) {
      imported = true;
      continue;
    }
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
    const field = /^    ([A-Za-z_]\w*) = (.+)$/.exec(line);
    if (current && field) {
      if (current.fields.some((item) => item.name === field[1]))
        throw new Error(`Schema field ${current.name}.${field[1]} is declared twice.`);
      current.fields.push({ name: field[1], declaration: field[2] });
      continue;
    }
    throw new Error(
      `Model explainer cannot describe schema line ${index + 1}: ${line}. Use explicit Model/Branch classes and one-line field declarations.`,
    );
  }
  const roots = classes.filter((item) => item.kind === 'Model');
  if (!imported || roots.length !== 1 || constructed !== roots[0]?.name)
    throw new Error(
      'Model explainer needs one Model class and its guarded small-preset constructor.',
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
      const call = /^(rf\.)?([A-Za-z_]\w*)(?:\((.*)\))?$/.exec(declaration);
      if (!call)
        throw new Error(
          `Schema field ${owner.name}.${name} must use a direct field or declared branch constructor.`,
        );
      const [, namespace, type, options = ''] = call;
      const context = pythonContext(options);
      for (let index = 0; index < options.length; index++) {
        if (['(', ')'].includes(options[index]) && context.isCode(index))
          throw new Error(
            `Schema field ${owner.name}.${name} has nested configuration calls outside this explainer's literal schema contract.`,
          );
      }
      const path = `${prefix}/${name}`;
      if (namespace) {
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
