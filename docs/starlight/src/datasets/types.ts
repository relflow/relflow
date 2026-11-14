/** Authored data contracts describe source records, independently of example values. */
export interface DatasetColumn {
  name: string;
  type: string;
  description: string;
  role?: 'key' | 'reference' | 'target' | 'training' | 'collection' | 'record';
}

export interface DatasetEntity {
  id: string;
  name: string;
  description: string;
  columns: DatasetColumn[];
}

export interface DatasetRelation {
  from: string;
  to: string;
  fromColumn?: string;
  toColumn?: string;
  cardinality?: 'many' | 'one';
  description: string;
}

export interface Dataset {
  id: string;
  label: string;
  observation: string;
  description: string;
  entities: DatasetEntity[];
  relations: DatasetRelation[];
}

/** Place every entity once, including disconnected tables and shared children. */
export function layout(dataset: Dataset): DatasetEntity[][] {
  const entities = new Map(dataset.entities.map((entity) => [entity.id, entity]));
  if (!entities.size || entities.size !== dataset.entities.length)
    throw new Error(`${dataset.id}: provide at least one entity and unique entity IDs.`);
  for (const entity of dataset.entities) {
    if (new Set(entity.columns.map((column) => column.name)).size !== entity.columns.length)
      throw new Error(`${dataset.id}/${entity.id}: column names must be unique.`);
  }
  for (const relation of dataset.relations) {
    const parent = entities.get(relation.from);
    const child = entities.get(relation.to);
    if (!parent || !child)
      throw new Error(
        `${dataset.id}: relationship ${relation.from} → ${relation.to} references an unknown entity.`,
      );
    if (
      (relation.fromColumn &&
        !parent.columns.some((column) => column.name === relation.fromColumn)) ||
      (relation.toColumn && !child.columns.some((column) => column.name === relation.toColumn))
    )
      throw new Error(
        `${dataset.id}: relationship ${relation.from} → ${relation.to} references an unknown column.`,
      );
  }
  const incoming = new Set(dataset.relations.map((relation) => relation.to));
  const queue = dataset.entities
    .filter((entity) => !incoming.has(entity.id))
    .map((entity) => ({ entity, depth: 0 }));
  const visited = new Set<string>();
  const levels: DatasetEntity[][] = [];
  while (visited.size < entities.size) {
    if (!queue.length)
      queue.push({ entity: dataset.entities.find((entity) => !visited.has(entity.id))!, depth: 0 });
    const { entity, depth } = queue.shift()!;
    if (visited.has(entity.id)) continue;
    visited.add(entity.id);
    (levels[depth] ||= []).push(entity);
    for (const relation of dataset.relations.filter((item) => item.from === entity.id)) {
      if (!visited.has(relation.to))
        queue.push({ entity: entities.get(relation.to)!, depth: depth + 1 });
    }
  }
  return levels;
}
