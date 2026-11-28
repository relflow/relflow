import { palettes, type ColorScheme, type Palette } from '../data/palettes';
import { codeThemes } from '../styles/code-themes.mjs';
import { sitePath } from '../site/urls.mjs';

type Mode = 'light' | 'dark';
const body = document.body;
const search = document.querySelector<HTMLInputElement>('#palette-search')!;
const family = document.querySelector<HTMLSelectElement>('#palette-family')!;
const shortlistFilter = document.querySelector<HTMLButtonElement>('#shortlist-filter')!;
const cards = [...document.querySelectorAll<HTMLElement>('[data-palette-card]')];
const dialog = document.querySelector<HTMLDialogElement>('#palette-preview')!;
const frame = document.querySelector<HTMLIFrameElement>('#preview-frame')!;
const paletteSelect = document.querySelector<HTMLSelectElement>('#preview-palette')!;
const pageSelect = document.querySelector<HTMLSelectElement>('#preview-page')!;
const previewStatus = document.querySelector<HTMLElement>('#preview-status')!;
const storageKey = 'relflow-docs-palette-shortlist-v1';
let mode: Mode = body.dataset.mode === 'dark' ? 'dark' : 'light';
let current = palettes[0];
const base = import.meta.env.BASE_URL;
let currentPage = new URL(
  sitePath(pageSelect.value || '/data-types/number/', base),
  location.origin,
);
let hoveredPalette = current;
let lastOpener: HTMLElement | null = null;
let loadGeneration = 0;
let pending: AbortController | undefined;
const tokenStyles = new WeakMap<HTMLElement, { dark: string; light: string }>();
const shortlist = new Set<string>();
try {
  const saved: unknown = JSON.parse(localStorage.getItem(storageKey) || '[]');
  if (Array.isArray(saved))
    for (const id of saved)
      if (typeof id === 'string' && palettes.some((palette) => palette.id === id))
        shortlist.add(id);
} catch {
  /* The gallery remains usable when browser storage is unavailable. */
}

function paletteById(id: string | undefined): Palette | undefined {
  return palettes.find((palette) => palette.id === id);
}

function activeTooltip(palette: Palette) {
  hoveredPalette = palette;
  const scheme = palette[mode];
  body.style.setProperty('--active-tooltip', scheme.tooltip);
  body.style.setProperty('--active-text', scheme.text);
  body.style.setProperty('--active-border', scheme.border);
}

function visiblePalettes(): Palette[] {
  const query = search.value.trim().toLowerCase();
  const onlyShortlisted = shortlistFilter.getAttribute('aria-pressed') === 'true';
  return palettes.filter(
    (palette) =>
      (!family.value || family.value === 'all' || palette.family === family.value) &&
      (!query ||
        `${palette.name} ${palette.family} ${palette.description}`.toLowerCase().includes(query)) &&
      (!onlyShortlisted || shortlist.has(palette.id)),
  );
}

function filter() {
  const visible = new Set(visiblePalettes().map((palette) => palette.id));
  for (const card of cards) card.hidden = !visible.has(card.dataset.paletteCard!);
  document.querySelector('#palette-count')!.textContent =
    `Showing ${visible.size} of ${palettes.length} accents`;
  document.querySelector('#shortlist-count')!.textContent = String(shortlist.size);
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-shortlist]')) {
    const palette = paletteById(button.dataset.shortlist)!;
    const selected = shortlist.has(palette.id);
    button.setAttribute('aria-pressed', String(selected));
    button.setAttribute(
      'aria-label',
      `${selected ? 'Remove' : 'Add'} ${palette.name} ${selected ? 'from' : 'to'} shortlist`,
    );
    button.title = selected ? 'Remove from shortlist' : 'Add to shortlist';
    const label = button.querySelector('[data-shortlist-label]');
    if (label) label.textContent = selected ? 'Shortlisted' : 'Shortlist';
  }
  const empty = document.querySelector<HTMLElement>('#palette-empty');
  if (empty) empty.hidden = visible.size !== 0;
}

function updateHash() {
  const params = new URLSearchParams({
    palette: current.id,
    mode,
    page: currentPage.pathname + currentPage.search + currentPage.hash,
  });
  history.replaceState(null, '', `${location.pathname}${location.search}#${params}`);
}

function swatches(scheme: ColorScheme) {
  const container = document.querySelector('#preview-swatches')!;
  container.replaceChildren();
  for (const role of ['accent', 'keyword'] as const) {
    const name = role === 'accent' ? 'Links / UI' : 'Code';
    const swatch = document.createElement('span');
    swatch.className = 'preview-swatch';
    swatch.style.setProperty('--swatch', scheme[role]);
    swatch.title = `${name}: ${scheme[role]}`;
    swatch.setAttribute('role', 'img');
    swatch.setAttribute('aria-label', `${name}: ${scheme[role]}`);
    const color = document.createElement('i');
    color.setAttribute('aria-hidden', 'true');
    color.style.backgroundColor = scheme[role];
    const label = document.createElement('span');
    label.textContent = name;
    const hex = document.createElement('code');
    hex.textContent = scheme[role];
    swatch.append(color, label, hex);
    container.append(swatch);
  }
}

function details() {
  paletteSelect.value = current.id;
  document.querySelector('#preview-tokens')!.textContent = JSON.stringify(current, null, 2);
  swatches(current[mode]);
  for (const button of document.querySelectorAll<HTMLButtonElement>('[data-mode-select]'))
    button.setAttribute('aria-pressed', String(button.dataset.modeSelect === mode));
}

const tokenRoles: Record<string, keyof ColorScheme> = Object.fromEntries(
  codeThemes.flatMap((theme) => [
    [theme.colors['editor.foreground'].toUpperCase(), 'text'],
    ...theme.tokenColors.map((token) => [
      token.settings.foreground.toUpperCase(),
      token.scope.includes('comment')
        ? 'comment'
        : token.scope.includes('string')
          ? 'string'
          : token.scope.includes('constant.numeric')
            ? 'number'
            : 'keyword',
    ]),
  ]),
);

function applyFrameTheme() {
  const doc = frame.contentDocument;
  if (!doc?.documentElement.dataset.paletteRequest) return;
  const scheme = current[mode];
  doc.documentElement.dataset.theme = mode;
  doc.documentElement.style.colorScheme = mode;
  const variables: Record<string, string> = {
    '--sl-color-white': scheme.ink,
    '--sl-color-gray-1': scheme.ink,
    '--sl-color-gray-2': scheme.text,
    '--sl-color-gray-3': scheme.muted,
    '--sl-color-gray-4': scheme.muted,
    '--sl-color-gray-5': scheme.border,
    '--sl-color-gray-6': scheme.surface,
    '--sl-color-gray-7': scheme.code,
    '--sl-color-black': scheme.bg,
    '--sl-color-bg': scheme.bg,
    '--sl-color-bg-nav': scheme.sidebar,
    '--sl-color-bg-sidebar': scheme.sidebar,
    '--sl-color-bg-inline-code': scheme.surface,
    '--sl-color-bg-accent': scheme.accent,
    '--sl-color-text': scheme.text,
    '--sl-color-text-accent': scheme.accent,
    '--sl-color-text-invert': scheme.bg,
    '--sl-color-accent': scheme.accent,
    '--sl-color-accent-high': scheme.accent,
    '--sl-color-accent-low': scheme.accentSoft,
    '--sl-color-hairline': scheme.border,
    '--sl-color-hairline-light': scheme.border,
    '--sl-color-hairline-shade': scheme.border,
    '--rf-ink': scheme.ink,
    '--rf-muted': scheme.muted,
    '--rf-border': scheme.border,
    '--rf-surface': scheme.surface,
    '--rf-card': scheme.code,
    '--rf-code': scheme.code,
  };
  for (const [name, value] of Object.entries(variables))
    doc.documentElement.style.setProperty(name, value);
  let style = doc.querySelector<HTMLStyleElement>('#palette-preview-style');
  if (!style) {
    style = doc.createElement('style');
    style.id = 'palette-preview-style';
    doc.head.append(style);
  }
  style.textContent = `
    .expressive-code { --ec-codeFg:${scheme.text}!important;--ec-codeBg:${scheme.code}!important;--ec-brdCol:${scheme.border}!important;--ec-focusBrd:${scheme.accent}!important;--ec-frm-inlBtnFg:${scheme.text}!important;--ec-frm-inlBtnBg:${scheme.surface}!important;--ec-frm-tooltipSuccessBg:${scheme.accent}!important;--ec-frm-tooltipSuccessFg:${scheme.bg}!important; }
    .starlight-aside--note,.starlight-aside--tip { --sl-color-asides-text-accent:${scheme.accent};--sl-color-asides-border:${scheme.accent};background:${scheme.accentSoft}!important;color:${scheme.text}!important; }
    [data-palette-sample-note] { padding:1rem;border-inline-start:4px solid ${scheme.accent};margin-block:1.25rem; }
    [data-palette-sample-note] .starlight-aside__title { color:${scheme.accent};font-weight:650;margin:0 0 .25rem; }
    [data-palette-sample-note] .starlight-aside__content p { margin:0; }
    .rf-code-tooltip { background:${scheme.tooltip}!important;color:${scheme.text}!important;border-color:${scheme.border}!important; }
  `;
  for (const span of doc.querySelectorAll<HTMLElement>('.expressive-code span[style*="--0:"]')) {
    let original = tokenStyles.get(span);
    if (!original) {
      original = {
        dark: span.style.getPropertyValue('--0').trim().toUpperCase(),
        light: span.style.getPropertyValue('--1').trim().toUpperCase(),
      };
      tokenStyles.set(span, original);
    }
    span.style.setProperty('--0', scheme[tokenRoles[original.dark] || 'text']);
    span.style.setProperty('--1', scheme[tokenRoles[original.light] || 'text']);
  }
}

function prepareDocument(html: string, url: URL, generation: number): string {
  const doc = new DOMParser().parseFromString(html, 'text/html');
  doc.documentElement.dataset.paletteRequest = String(generation);
  for (const picker of doc.querySelectorAll('starlight-theme-select')) picker.remove();
  const base = doc.querySelector('base') || doc.createElement('base');
  const target = new URL(url);
  target.hash = '';
  base.href = target.href;
  doc.head.prepend(base);
  const content = doc.querySelector('.sl-markdown-content');
  if (content) {
    const note = doc.createElement('aside');
    note.className = 'starlight-aside starlight-aside--note';
    note.dataset.paletteSampleNote = '';
    note.setAttribute('aria-label', 'Note');
    note.innerHTML =
      '<p class="starlight-aside__title" aria-hidden="true">Note</p><div class="starlight-aside__content"><p>Target values stay out of the model’s inputs. Use <code>predict=True</code> for a supervised answer.</p></div>';
    const firstHeading = content.querySelector('.sl-heading-wrapper');
    if (firstHeading) firstHeading.before(note);
    else content.prepend(note);
  }
  return `<!doctype html>\n${doc.documentElement.outerHTML}`;
}

function scrollHash(hash: string) {
  const win = frame.contentWindow;
  if (!win || !hash) return;
  win.location.hash = hash;
  win.dispatchEvent(new HashChangeEvent('hashchange'));
  try {
    frame.contentDocument
      ?.getElementById(decodeURIComponent(hash.slice(1)))
      ?.scrollIntoView({ block: 'start' });
  } catch {
    /* Ignore malformed fragment escapes. */
  }
}

function syncPageSelect(url: URL) {
  const value = url.pathname + url.search;
  for (const option of pageSelect.querySelectorAll('option[data-custom-preview]')) option.remove();
  if (![...pageSelect.options].some((option) => option.value === value)) {
    const option = new Option(url.pathname, value);
    option.dataset.customPreview = '';
    pageSelect.append(option);
  }
  pageSelect.value = value;
}

async function loadPreview(url: URL) {
  if (url.origin !== location.origin || url.pathname.startsWith(sitePath('/palettes', base)))
    return;
  currentPage = new URL(url);
  syncPageSelect(currentPage);
  updateHash();
  pending?.abort();
  pending = new AbortController();
  const generation = ++loadGeneration;
  previewStatus.textContent = `Loading ${current.name} preview…`;
  frame.setAttribute('aria-busy', 'true');
  frame.style.visibility = 'hidden';
  try {
    const response = await fetch(currentPage.href, { signal: pending.signal });
    if (!response.ok) throw new Error(`HTTP ${response.status}`);
    const html = await response.text();
    if (generation !== loadGeneration) return;
    frame.onload = () => {
      const doc = frame.contentDocument;
      if (
        generation !== loadGeneration ||
        doc?.documentElement.dataset.paletteRequest !== String(generation)
      )
        return;
      applyFrameTheme();
      frame.style.visibility = 'visible';
      frame.setAttribute('aria-busy', 'false');
      previewStatus.textContent = `${current.name} · ${mode} · ${currentPage.pathname}`;
      scrollHash(currentPage.hash);
      doc.addEventListener('click', (event) => {
        const node = event.target as Node | null;
        const element = node?.nodeType === 1 ? (node as Element) : node?.parentElement;
        const anchor = element?.closest<HTMLAnchorElement>('a[href]');
        if (
          !anchor ||
          event.defaultPrevented ||
          event.button !== 0 ||
          event.ctrlKey ||
          event.metaKey ||
          event.shiftKey ||
          event.altKey
        )
          return;
        const target = new URL(anchor.href, currentPage);
        event.preventDefault();
        if (
          anchor.hasAttribute('download') ||
          target.origin !== location.origin ||
          /\.[a-z0-9]{1,8}$/i.test(target.pathname)
        ) {
          window.open(target.href, '_blank', 'noopener,noreferrer');
          return;
        }
        if (
          target.pathname === currentPage.pathname &&
          target.search === currentPage.search &&
          target.hash
        ) {
          currentPage = target;
          updateHash();
          scrollHash(target.hash);
        } else void loadPreview(target);
      });
    };
    frame.srcdoc = prepareDocument(html, currentPage, generation);
  } catch (error) {
    if (
      generation !== loadGeneration ||
      (error instanceof DOMException && error.name === 'AbortError')
    )
      return;
    frame.style.visibility = 'visible';
    frame.setAttribute('aria-busy', 'false');
    previewStatus.textContent = `Preview could not load: ${error instanceof Error ? error.message : 'unknown error'}. Choose a page to retry.`;
  }
}

function setMode(next: Mode) {
  mode = next;
  body.dataset.mode = next;
  activeTooltip(hoveredPalette);
  details();
  if (dialog.open) {
    applyFrameTheme();
    updateHash();
    previewStatus.textContent = `${current.name} · ${mode} · ${currentPage.pathname}`;
  }
}

function setPalette(palette: Palette) {
  current = palette;
  activeTooltip(palette);
  details();
  if (dialog.open) {
    applyFrameTheme();
    updateHash();
    previewStatus.textContent = `${current.name} · ${mode} · ${currentPage.pathname}`;
  }
}

function openPreview(palette: Palette, opener?: HTMLElement) {
  document.dispatchEvent(new Event('code-notes:close'));
  current = palette;
  lastOpener = opener || null;
  details();
  activeTooltip(palette);
  if (!dialog.open) dialog.showModal();
  void loadPreview(currentPage);
}

async function copy(palette: Palette, button: HTMLButtonElement) {
  try {
    await navigator.clipboard.writeText(JSON.stringify(palette, null, 2));
    const label = button.querySelector('[data-copy-label]');
    const original = label?.textContent;
    const originalMarkup = button.innerHTML;
    const originalTitle = button.getAttribute('title');
    if (label) label.textContent = 'Copied';
    else button.textContent = 'Copied';
    button.dataset.copied = '';
    button.title = 'Palette JSON copied';
    setTimeout(() => {
      delete button.dataset.copied;
      if (label && original !== undefined) label.textContent = original;
      else button.innerHTML = originalMarkup;
      if (originalTitle === null) button.removeAttribute('title');
      else button.title = originalTitle;
    }, 1600);
  } catch {
    button.title = 'Copy unavailable; select the palette JSON in the preview.';
    if (dialog.open)
      previewStatus.textContent =
        'Clipboard access is unavailable. Select the palette JSON below to copy it.';
  }
}

search.addEventListener('input', filter);
family.addEventListener('change', filter);
shortlistFilter.addEventListener('click', () => {
  shortlistFilter.setAttribute(
    'aria-pressed',
    String(shortlistFilter.getAttribute('aria-pressed') !== 'true'),
  );
  filter();
});
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-shortlist]'))
  button.addEventListener('click', () => {
    const id = button.dataset.shortlist!;
    if (shortlist.has(id)) shortlist.delete(id);
    else shortlist.add(id);
    try {
      localStorage.setItem(storageKey, JSON.stringify([...shortlist]));
    } catch {
      /* Shortlisting still works for this visit. */
    }
    filter();
  });
for (const card of cards) {
  const palette = paletteById(card.dataset.paletteCard)!;
  card.addEventListener('pointerenter', () => activeTooltip(palette));
  card.addEventListener('focusin', () => activeTooltip(palette));
}
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-preview]'))
  button.addEventListener('click', () => openPreview(paletteById(button.dataset.preview)!, button));
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-copy]'))
  button.addEventListener('click', () => void copy(paletteById(button.dataset.copy)!, button));
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-mode-select]'))
  button.addEventListener('click', () =>
    setMode(button.dataset.modeSelect === 'dark' ? 'dark' : 'light'),
  );
paletteSelect.addEventListener('change', () => {
  const palette = paletteById(paletteSelect.value);
  if (palette) setPalette(palette);
});
pageSelect.addEventListener(
  'change',
  () => void loadPreview(new URL(sitePath(pageSelect.value, base), location.origin)),
);
for (const button of document.querySelectorAll<HTMLButtonElement>('[data-preview-step]'))
  button.addEventListener('click', () => {
    const visible = visiblePalettes();
    const pool = visible.length ? visible : palettes;
    const index = pool.findIndex((palette) => palette.id === current.id);
    const direction = Number(button.dataset.previewStep);
    const next =
      index < 0
        ? direction > 0
          ? 0
          : pool.length - 1
        : (index + direction + pool.length) % pool.length;
    setPalette(pool[next]);
  });
document.querySelector('#close-preview')!.addEventListener('click', () => dialog.close());
document
  .querySelector<HTMLButtonElement>('#copy-current')!
  .addEventListener(
    'click',
    (event) => void copy(current, event.currentTarget as HTMLButtonElement),
  );
dialog.addEventListener('close', () => {
  pending?.abort();
  loadGeneration++;
  history.replaceState(null, '', `${location.pathname}${location.search}`);
  lastOpener?.focus({ preventScroll: true });
});
function restoreHash() {
  const params = new URLSearchParams(location.hash.slice(1));
  const palette = paletteById(params.get('palette') || undefined);
  if (!palette) return;
  setMode(params.get('mode') === 'dark' ? 'dark' : 'light');
  const page = params.get('page');
  if (page) {
    try {
      const url = new URL(sitePath(page, base), location.origin);
      if (url.origin === location.origin && !url.pathname.startsWith(sitePath('/palettes', base)))
        currentPage = url;
    } catch {
      /* An invalid shared page does not prevent opening the selected palette. */
    }
  }
  openPreview(palette);
}
window.addEventListener('hashchange', restoreHash);
filter();
setMode(mode);
restoreHash();
