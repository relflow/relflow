import { parseFragment, type DefaultTreeAdapterTypes } from 'parse5';

const sections = [
  ['input-values', 'Input values'],
  ['standardization-and-processor-pipelines', 'Standardization and processor pipelines'],
  ['how-is-this-leaf-trained', 'How is this leaf trained?'],
  ['leaf-specific-parameters', 'Leaf-specific parameters'],
  ['errors-raised', 'Errors raised'],
  ['relevant-evidence', 'Relevant evidence'],
] as const;

/** Validate the rendered Markdown contract without rewriting prose or code. */
export function checkSections(html: string, page: string): void {
  const found: [string | undefined, string][] = [];
  function text(node: DefaultTreeAdapterTypes.Node): string {
    if (node.nodeName === '#text' && 'value' in node) return node.value;
    return 'childNodes' in node ? node.childNodes.map(text).join('') : '';
  }
  function visit(node: DefaultTreeAdapterTypes.Node): void {
    if ('tagName' in node && node.tagName === 'h2')
      found.push([node.attrs.find((attribute) => attribute.name === 'id')?.value, text(node)]);
    if ('childNodes' in node) node.childNodes.forEach(visit);
  }
  visit(parseFragment(html));
  if (
    found.length !== sections.length ||
    found.some(([id, title], index) => id !== sections[index][0] || title !== sections[index][1])
  ) {
    throw new Error(
      `Leaf type page ${page} must provide ${sections.map(([id, title]) => `${title} (#${id})`).join(', ')} in that order; received ${found.map(([id, title]) => `${title} (#${id ?? 'missing'})`).join(', ') || 'no sections'}. Use ordinary level-two Markdown headings for this contract.`,
    );
  }
}
