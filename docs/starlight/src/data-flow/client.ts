import { flowPaths, type FlowMode, type FlowStageId } from './types';

for (const root of document.querySelectorAll<HTMLElement>('[data-data-flow]')) {
  if (root.hasAttribute('data-flow-ready')) continue;
  const cases = [...root.querySelectorAll<HTMLElement>('[data-case]')];
  const modeButtons = [...root.querySelectorAll<HTMLButtonElement>('[data-flow-mode-select]')];
  const status = root.querySelector<HTMLElement>('[data-flow-status]')!;
  let selected = selectedCase();
  let mode: FlowMode = 'predict';
  let stage: FlowStageId = 'load';

  function selectedCase(): string {
    return root.querySelector<HTMLButtonElement>('[data-case-select][aria-pressed="true"]')!.dataset
      .caseSelect!;
  }

  function activeStage(): HTMLDetailsElement {
    return document.getElementById(`flow-${selected}-${stage}`) as HTMLDetailsElement;
  }

  function writeURL(action: 'push' | 'replace') {
    const url = new URL(location.href);
    url.searchParams.set('case', selected);
    url.searchParams.set('mode', mode);
    url.hash = `flow-${selected}-${stage}`;
    if (url.href === location.href) return;
    if (action === 'push') history.pushState(null, '', url);
    else history.replaceState(null, '', url);
  }

  function render(announce = false) {
    const active = activeStage();
    const focused = document.activeElement;
    if (focused instanceof HTMLElement && root.contains(focused)) {
      const oldStage = focused.closest('details[data-flow-stage]');
      const oldCase = focused.closest<HTMLElement>('[data-case]');
      const oldView = focused.closest<HTMLElement>('[data-flow-view-mode]');
      if (
        (oldStage && oldStage !== active) ||
        (oldCase && oldCase.dataset.case !== selected) ||
        (oldView && oldView.dataset.flowViewMode !== mode)
      )
        active.querySelector<HTMLElement>('summary')!.focus();
    }
    root.dataset.flowMode = mode;
    root.dataset.flowStage = stage;
    root.dataset.flowCase = selected;
    for (const button of modeButtons)
      button.setAttribute('aria-pressed', String(button.dataset.flowModeSelect === mode));
    for (const item of cases) {
      for (const path of item.querySelectorAll<HTMLElement>('[data-flow-path]'))
        path.hidden = path.dataset.flowPath !== mode;
      for (const view of item.querySelectorAll<HTMLElement>('[data-flow-view-mode]'))
        view.hidden = view.dataset.flowViewMode !== mode;
      for (const code of item.querySelectorAll<HTMLElement>('[data-flow-code-modes]'))
        code.hidden = !code.dataset.flowCodeModes!.split(' ').includes(mode);
      for (const detail of item.querySelectorAll<HTMLDetailsElement>('details[data-flow-stage]')) {
        const isActive = detail === active;
        detail.open = isActive;
        detail.toggleAttribute('data-flow-active', isActive);
        const position = flowPaths[mode].indexOf(detail.dataset.flowStage as FlowStageId);
        if (position < 0) continue;
        const previous = detail.querySelector<HTMLButtonElement>('[data-flow-previous]')!;
        const next = detail.querySelector<HTMLButtonElement>('[data-flow-next]')!;
        previous.disabled = position === 0;
        previous.setAttribute(
          'aria-controls',
          `flow-${item.dataset.case}-${flowPaths[mode][Math.max(0, position - 1)]}`,
        );
        const following = flowPaths[mode][position + 1] || 'load';
        next.setAttribute('aria-controls', `flow-${item.dataset.case}-${following}`);
      }
    }
    if (announce)
      status.textContent = `${mode === 'train' ? 'Training' : 'Prediction'} · Step ${flowPaths[mode].indexOf(stage) + 1} of ${flowPaths[mode].length}: ${active.querySelector('summary strong')!.textContent}.`;
  }

  function select(next: FlowStageId, focus = false) {
    if (!flowPaths[mode].includes(next)) return;
    document.dispatchEvent(new Event('code-notes:close'));
    stage = next;
    root.dataset.flowInteracted = '';
    render(true);
    writeURL('push');
    if (focus) activeStage().querySelector<HTMLElement>('summary')!.focus();
  }

  function restore() {
    selected = selectedCase();
    mode = new URL(location.href).searchParams.get('mode') === 'train' ? 'train' : 'predict';
    let fragment = '';
    try {
      fragment = decodeURIComponent(location.hash.slice(1));
    } catch {
      /* Keep a usable default for malformed fragments. */
    }
    const prefix = `flow-${selected}-`;
    const requested = fragment.startsWith(prefix)
      ? (fragment.slice(prefix.length) as FlowStageId)
      : undefined;
    if (requested && !flowPaths[mode].includes(requested)) {
      const other = mode === 'train' ? 'predict' : 'train';
      if (flowPaths[other].includes(requested)) mode = other;
    }
    stage = requested && flowPaths[mode].includes(requested) ? requested : 'load';
    render();
  }

  root.addEventListener('click', (event) => {
    const button =
      event.target instanceof Element ? event.target.closest<HTMLButtonElement>('button') : null;
    if (!button || button.disabled) return;
    const requested = button.dataset.flowModeSelect;
    if (requested === 'train' || requested === 'predict') {
      if (requested === mode) return;
      const position = flowPaths[mode].indexOf(stage);
      mode = requested;
      stage = flowPaths[mode][position];
      root.dataset.flowInteracted = '';
      document.dispatchEvent(new Event('code-notes:close'));
      render(true);
      writeURL('push');
    } else if (button.hasAttribute('data-flow-next') || button.hasAttribute('data-flow-previous')) {
      const position = flowPaths[mode].indexOf(stage);
      const offset = button.hasAttribute('data-flow-next') ? 1 : -1;
      select(flowPaths[mode][position + offset] || 'load', true);
    }
  });

  root.addEventListener(
    'toggle',
    (event) => {
      const detail = event.target;
      if (!(detail instanceof HTMLDetailsElement) || !detail.hasAttribute('data-flow-stage'))
        return;
      if (detail.closest<HTMLElement>('[data-case]')!.dataset.case !== selected) return;
      const next = detail.dataset.flowStage as FlowStageId;
      if (!flowPaths[mode].includes(next)) return;
      if (detail.open && next !== stage) select(next);
    },
    true,
  );

  new MutationObserver((records) => {
    if (
      !records.some(
        (record) =>
          record.target instanceof HTMLElement && record.target.hasAttribute('data-case-select'),
      )
    )
      return;
    const next = selectedCase();
    if (next === selected) return;
    selected = next;
    root.dataset.flowInteracted = '';
    render(true);
    writeURL('replace');
  }).observe(root, { subtree: true, attributes: true, attributeFilter: ['aria-pressed'] });

  window.addEventListener('popstate', restore);
  window.addEventListener('hashchange', restore);
  restore();
  for (const button of modeButtons) button.disabled = false;
  for (const controls of root.querySelectorAll<HTMLElement>('[data-flow-stage-controls]'))
    controls.hidden = false;
  root.dataset.flowReady = '';
}
