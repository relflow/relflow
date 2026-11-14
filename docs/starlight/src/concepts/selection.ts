const preferenceKey = 'relflow:use-case';
const base = import.meta.env.BASE_URL.replace(/\/$/, '');

for (const root of document.querySelectorAll<HTMLElement>('[data-use-cases]')) {
  const buttons = [...root.querySelectorAll<HTMLButtonElement>('[data-case-select]')];
  const examples = [...root.querySelectorAll<HTMLElement>('[data-case]')];
  const status = root.querySelector<HTMLElement>('[data-case-status]')!;
  let selected = 0;

  function hashTarget() {
    try {
      return document.getElementById(decodeURIComponent(location.hash.slice(1)));
    } catch {
      return null;
    }
  }

  function storedCase() {
    try {
      return localStorage.getItem(preferenceKey);
    } catch {
      return null;
    }
  }

  function remember(id: string) {
    try {
      localStorage.setItem(preferenceKey, id);
    } catch {
      /* URL-based selection still works when browser storage is unavailable. */
    }
  }

  function linkExamples(id: string) {
    for (const link of document.querySelectorAll<HTMLAnchorElement>('a[href]')) {
      if (link.hasAttribute('data-case-permalink')) continue;
      const url = new URL(link.href, location.href);
      if (
        url.origin !== location.origin ||
        !(
          url.pathname.startsWith(`${base}/concepts/`) ||
          url.pathname.startsWith(`${base}/advanced/`) ||
          url.pathname === `${base}/` ||
          url.pathname === `${base}/datasets/`
        )
      )
        continue;
      url.searchParams.set('case', id);
      link.href = url.pathname + url.search + url.hash;
    }
  }

  function select(index: number, announce = true, updateURL = true) {
    if (index !== selected) {
      document.dispatchEvent(new Event('code-notes:close'));
      if (examples[selected].contains(document.activeElement))
        buttons[index].focus({ preventScroll: true });
    }
    const target = hashTarget();
    const previous = examples[selected];
    selected = index;
    const id = buttons[index].dataset.caseSelect!;
    examples.forEach((example, position) => {
      const active = position === index;
      example.inert = !active;
      example.setAttribute('aria-hidden', String(!active));
      buttons[position].setAttribute('aria-pressed', String(active));
    });
    remember(id);
    linkExamples(id);
    if (updateURL) {
      const url = new URL(location.href);
      url.searchParams.set('case', id);
      if (target && previous.contains(target) && !examples[index].contains(target)) url.hash = '';
      history.replaceState(null, '', url);
    }
    if (announce)
      status.textContent = `${buttons[index].querySelector('.case-label')!.textContent} example selected.`;
  }

  function restore() {
    const target = hashTarget();
    const fromHash = examples.findIndex((example) => target && example.contains(target));
    const fromURL = buttons.findIndex(
      (button) => button.dataset.caseSelect === new URL(location.href).searchParams.get('case'),
    );
    const fromStorage = buttons.findIndex((button) => button.dataset.caseSelect === storedCase());
    select(
      fromHash >= 0 ? fromHash : fromURL >= 0 ? fromURL : fromStorage >= 0 ? fromStorage : 0,
      false,
      false,
    );
  }

  buttons.forEach((button, index) => {
    button.addEventListener('click', () => select(index));
    button.addEventListener('keydown', (event) => {
      if (event.altKey || event.ctrlKey || event.metaKey) return;
      const destinations: Record<string, number> = {
        ArrowLeft: (index + buttons.length - 1) % buttons.length,
        ArrowRight: (index + 1) % buttons.length,
        Home: 0,
        End: buttons.length - 1,
      };
      if (!(event.key in destinations)) return;
      event.preventDefault();
      select(destinations[event.key]);
      buttons[destinations[event.key]].focus({ preventScroll: true });
    });
  });

  restore();
  root.dataset.ready = '';
  window.addEventListener('popstate', restore);
  window.addEventListener('hashchange', restore);
}
