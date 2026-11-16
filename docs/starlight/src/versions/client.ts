import { catalog, type VersionMenu } from './catalog';

declare global {
  interface Window {
    ReadTheDocsEventData?: { data(): unknown };
  }
}

const root = document.querySelector<HTMLElement>('[data-version-switcher]');
if (root) {
  const trigger = root.querySelector<HTMLButtonElement>('[data-version-launch]')!;
  const label = root.querySelector<HTMLElement>('[data-version-current]')!;
  const dialog = root.querySelector<HTMLDialogElement>('dialog')!;
  const search = root.querySelector<HTMLInputElement>('[data-version-search]')!;
  const list = root.querySelector<HTMLElement>('[data-version-list]')!;
  const empty = root.querySelector<HTMLElement>('[data-version-empty]')!;
  const template = root.querySelector<HTMLTemplateElement>('[data-version-row]')!;

  function filter() {
    const query = search.value.trim().toLocaleLowerCase();
    let matches = 0;
    for (const link of list.querySelectorAll<HTMLAnchorElement>('a')) {
      link.hidden = !link.textContent!.toLocaleLowerCase().includes(query);
      if (!link.hidden) matches++;
    }
    empty.hidden = matches > 0;
  }

  function render(menu: VersionMenu, hosted: boolean) {
    const current = menu.versions.find((version) => version.id === menu.current);
    if (!current) return;
    const rows = document.createDocumentFragment();
    for (const version of menu.versions) {
      const row = template.content.firstElementChild!.cloneNode(true) as HTMLAnchorElement;
      row.href = version.url;
      row.querySelector('strong')!.textContent = version.label;
      row.querySelector('small')!.textContent = version.detail;
      const badge = row.querySelector<HTMLElement>('[data-version-badge]')!;
      badge.textContent = version.badge;
      badge.hidden = !version.badge;
      badge.classList.toggle('rf-versions-stable', version.badge === 'Stable');
      if (version.id === menu.current) row.setAttribute('aria-current', 'page');
      rows.append(row);
    }
    list.replaceChildren(rows);
    label.textContent = current.label;
    trigger.setAttribute('aria-label', `Documentation version: ${current.label}`);
    root!.querySelector<HTMLElement>('[data-version-provider]')!.textContent = hosted
      ? 'Hosted on Read the Docs'
      : 'Local preview · illustrative versions';
    filter();
    root!.hidden = false;
    if (hosted) document.documentElement.dataset.versionSwitcherReady = '';
  }

  trigger.addEventListener('click', () => {
    search.value = '';
    filter();
    dialog.showModal();
    search.focus();
  });
  search.addEventListener('input', filter);
  dialog.addEventListener('click', (event) => {
    const bounds = dialog.getBoundingClientRect();
    if (
      event.target === dialog &&
      (event.clientX < bounds.left ||
        event.clientX > bounds.right ||
        event.clientY < bounds.top ||
        event.clientY > bounds.bottom)
    )
      dialog.close();
  });
  dialog.addEventListener('keydown', (event) => {
    if (!(event.target instanceof Element)) return;
    const links = [...list.querySelectorAll<HTMLAnchorElement>('a:not([hidden])')];
    if (!links.length) return;
    const index = links.indexOf(event.target as HTMLAnchorElement);
    let next: number;
    if (event.key === 'ArrowDown') next = (index + 1) % links.length;
    else if (event.key === 'ArrowUp')
      next = index < 0 ? links.length - 1 : (index - 1 + links.length) % links.length;
    else if (index >= 0 && event.key === 'Home') next = 0;
    else if (index >= 0 && event.key === 'End') next = links.length - 1;
    else return;
    event.preventDefault();
    links[next].focus();
  });

  if (root.dataset.versionSample) {
    const sample = JSON.parse(root.dataset.versionSample) as VersionMenu;
    const selected = new URL(location.href).searchParams.get('version');
    if (sample.versions.some((version) => version.id === selected)) sample.current = selected!;
    render(sample, false);
  }

  function receive(source: { data(): unknown }) {
    let data: unknown;
    try {
      data = source.data();
    } catch {
      // Older Addons responses may not support the opted-in API version.
      return;
    }
    const menu = catalog(data, location.href);
    if (menu) render(menu, true);
  }
  document.addEventListener('readthedocs-addons-data-ready', (event) => {
    const source = (event as CustomEvent<{ data(): unknown }>).detail;
    if (source && typeof source.data === 'function') receive(source);
  });
  if (window.ReadTheDocsEventData) receive(window.ReadTheDocsEventData);
}
