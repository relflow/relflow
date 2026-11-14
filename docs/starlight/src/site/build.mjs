// @ts-check
import { readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { parse } from 'parse5';
import { sitePath } from './urls.mjs';

/**
 * Scope HTML URL attributes while preserving every other source byte, including code.
 * @param {string} html
 * @param {string} base
 */
export function scopeHTML(html, base) {
  if (base === '/') return html;
  const document = parse(html, { sourceCodeLocationInfo: true, scriptingEnabled: false });
  /** @type {{start: number; end: number; text: string}[]} */
  const edits = [];
  /** @param {import('parse5').DefaultTreeAdapterTypes.Node} node */
  function visit(node) {
    if ('attrs' in node) {
      for (const attribute of node.attrs) {
        if (!['href', 'src', 'poster', 'action'].includes(attribute.name)) continue;
        const scoped = sitePath(attribute.value, base);
        if (scoped === attribute.value) continue;
        const location = node.sourceCodeLocation?.attrs?.[attribute.name];
        if (!location)
          throw new Error(`Hosted URL ${attribute.value} has no HTML source location.`);
        edits.push({
          start: location.startOffset,
          end: location.endOffset,
          text: `${attribute.name}="${scoped.replaceAll('&', '&amp;').replaceAll('"', '&quot;')}"`,
        });
      }
    }
    if ('childNodes' in node) node.childNodes.forEach(visit);
    if ('content' in node) visit(node.content);
  }
  visit(document);
  for (const edit of edits.sort((a, b) => b.start - a.start))
    html = html.slice(0, edit.start) + edit.text + html.slice(edit.end);
  return html;
}

/** @param {string} base @returns {import('astro').AstroIntegration} */
export function hostedURLs(base) {
  return {
    name: 'relflow version URLs',
    hooks: {
      'astro:build:done': ({ dir }) => {
        if (base === '/') return;
        /** @param {string} directory */
        function scope(directory) {
          for (const entry of readdirSync(directory, { withFileTypes: true })) {
            const path = join(directory, entry.name);
            if (entry.isDirectory()) scope(path);
            else if (entry.name.endsWith('.html')) {
              const original = readFileSync(path, 'utf8');
              const scoped = scopeHTML(original, base);
              if (scoped !== original) writeFileSync(path, scoped);
            }
          }
        }
        scope(fileURLToPath(dir));
      },
    },
  };
}
