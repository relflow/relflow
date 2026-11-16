// @ts-check
import { parse } from 'parse5';

const excludedTags = new Set([
  'code',
  'pre',
  'script',
  'style',
  'textarea',
  'template',
  'svg',
  'math',
  'form',
  'button',
  'input',
  'select',
  'option',
  'label',
  'kbd',
  'samp',
]);
const excludedClasses = new Set([
  'rf-wordmark',
  'expressive-code',
  'json-record',
  'value-tree',
  'concept-records',
  'df-records',
  'records-pane',
  'training-record',
  'raw-record',
  'df-code',
]);
const excludedAttributes = new Set([
  'data-no-brand',
  'data-value-tree',
  'data-json-record',
  'data-tree',
  'data-tree-source',
]);
const mention =
  /(?<![\p{L}\p{N}_./\\@+\-\[\]])relflow(?![\p{L}\p{N}_/\\@+\-\[\]<>=~]|\.[\p{L}\p{N}_/])/gu;

/**
 * Decorate prose without changing authored text, attributes, or recorded data.
 * Source-offset insertions preserve all bytes outside the added spans.
 * @param {string} html
 */
export function brandHTML(html) {
  if (!html.includes('relflow')) return html;
  const document = parse(html, { sourceCodeLocationInfo: true, scriptingEnabled: false });
  /** @type {{start: number; end: number}[]} */
  const edits = [];

  /**
   * @param {import('parse5').DefaultTreeAdapterTypes.Node} node
   * @param {boolean} prose
   */
  function visit(node, prose = false) {
    if ('tagName' in node) {
      if (excludedTags.has(node.tagName)) return;
      for (const attribute of node.attrs) {
        if (excludedAttributes.has(attribute.name)) return;
        if (
          attribute.name === 'class' &&
          attribute.value.split(/\s+/).some((name) => excludedClasses.has(name))
        )
          return;
        if (
          attribute.name === 'data-flow-view-kind' &&
          ['records', 'prediction'].includes(attribute.value)
        )
          return;
      }
      prose ||= node.tagName === 'main';
    }
    if (prose && node.nodeName === '#text' && node.sourceCodeLocation) {
      const location = node.sourceCodeLocation;
      const source = html.slice(location.startOffset, location.endOffset);
      for (const match of source.matchAll(mention))
        edits.push({
          start: location.startOffset + match.index,
          end: location.startOffset + match.index + match[0].length,
        });
    }
    if ('childNodes' in node) node.childNodes.forEach((child) => visit(child, prose));
  }

  visit(document);
  for (const edit of edits.sort((left, right) => right.start - left.start))
    html =
      html.slice(0, edit.start) +
      '<span class="rf-wordmark">' +
      html.slice(edit.start, edit.end) +
      '</span>' +
      html.slice(edit.end);
  return html;
}
