import { parseFragment, type DefaultTreeAdapterTypes } from 'parse5';

const sections = [
  ['inputs', 'Inputs'],
  ['predictions', 'Predictions'],
  ['options', 'Options'],
  ['request-methods', 'Request Methods'],
  ['footguns', 'Footguns'],
  ['related-types', 'Related types'],
  ['evidence', 'Evidence'],
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
      `Data type page ${page} must provide ${sections.map(([id, title]) => `${title} (#${id})`).join(', ')} in that order after its examples; received ${found.map(([id, title]) => `${title} (#${id ?? 'missing'})`).join(', ') || 'no sections'}. Use ordinary level-two Markdown headings for this contract.`,
    );
  }
}
