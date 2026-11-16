import { flowPaths, type FlowMode, type FlowStageId } from './types';

for (const root of document.querySelectorAll<HTMLElement>('[data-data-flow]')) {
  if (root.hasAttribute('data-flow-ready')) continue;
  const cases = [...root.querySelectorAll<HTMLElement>('[data-case]')];
  const panels = cases.map((element) => ({
    element,
    stages: [...element.querySelectorAll<HTMLDetailsElement>('details[data-flow-stage]')],
    links: [...element.querySelectorAll<HTMLAnchorElement>('[data-flow-select]')],
    views: [...element.querySelectorAll<HTMLElement>('[data-flow-view-mode]')],
    code: [...element.querySelectorAll<HTMLElement>('[data-flow-code-modes]')],
  }));
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
    let moveFocus = false;
    if (focused instanceof HTMLElement && root.contains(focused)) {
      const oldStage = focused.closest('details[data-flow-stage]');
      const oldCase = focused.closest<HTMLElement>('[data-case]');
      const oldView = focused.closest<HTMLElement>('[data-flow-view-mode]');
      if (
        (oldStage && oldStage !== active) ||
        (oldCase && oldCase.dataset.case !== selected) ||
        (oldView && oldView.dataset.flowViewMode !== mode)
      )
        moveFocus = true;
    }
    root.dataset.flowMode = mode;
    root.dataset.flowStage = stage;
    root.dataset.flowCase = selected;
    for (const button of modeButtons)
      button.setAttribute('aria-pressed', String(button.dataset.flowModeSelect === mode));
    const path = flowPaths[mode];
    const currentPosition = path.indexOf(stage);
    for (const panel of panels) {
      for (const link of panel.links) {
        const position = path.indexOf(link.dataset.flowSelect as FlowStageId);
        const isActive = panel.element.dataset.case === selected && position === currentPosition;
        link.parentElement!.hidden = position < 0;
        link.parentElement!.dataset.flowPosition =
          position < currentPosition ? 'complete' : isActive ? 'current' : 'upcoming';
        if (isActive) link.setAttribute('aria-current', 'step');
        else link.removeAttribute('aria-current');
        link.tabIndex = isActive ? 0 : -1;
      }
      for (const view of panel.views) view.hidden = view.dataset.flowViewMode !== mode;
      for (const code of panel.code)
        code.hidden = !code.dataset.flowCodeModes!.split(' ').includes(mode);
      for (const detail of panel.stages) {
        const isActive = detail === active;
        detail.open = isActive;
        detail.hidden = !isActive;
        detail.toggleAttribute('data-flow-active', isActive);
        detail.querySelector('summary')!.tabIndex = -1;
        if (!isActive) continue;
        const previous = detail.querySelector<HTMLButtonElement>('[data-flow-previous]')!;
        const next = detail.querySelector<HTMLButtonElement>('[data-flow-next]')!;
        previous.disabled = currentPosition === 0;
        previous.setAttribute(
          'aria-controls',
          `flow-${selected}-${path[Math.max(0, currentPosition - 1)]}`,
        );
        const following = path[currentPosition + 1] || 'load';
        const nextStage = panel.stages.find((item) => item.dataset.flowStage === following)!;
        next.setAttribute('aria-controls', nextStage.id);
        next.setAttribute(
          'aria-label',
          currentPosition === path.length - 1
            ? mode === 'train'
              ? 'Next batch'
              : 'Back to data'
            : `Continue to ${nextStage.querySelector('summary strong')!.textContent}`,
        );
      }
    }
    if (moveFocus) active.querySelector<HTMLElement>('summary')!.focus({ preventScroll: true });
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
    if (focus) activeStage().querySelector<HTMLElement>('summary')!.focus({ preventScroll: true });
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
    if (!(event.target instanceof Element)) return;
    const link = event.target.closest<HTMLAnchorElement>('[data-flow-select]');
    if (link && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) {
      event.preventDefault();
      select(link.dataset.flowSelect as FlowStageId);
      return;
    }
    // Stages share one workspace. Native disclosures inside the active stage
    // retain their own interaction; only the stage heading stays open.
    const summary = event.target.closest('summary');
    if (summary?.parentElement?.matches('details[data-flow-stage]')) {
      event.preventDefault();
      return;
    }
    const button = event.target.closest<HTMLButtonElement>('button');
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

  root.addEventListener('keydown', (event) => {
    if (!(event.target instanceof Element) || event.altKey || event.ctrlKey || event.metaKey)
      return;
    const link = event.target.closest<HTMLAnchorElement>('[data-flow-select]');
    if (!link) return;
    const path = flowPaths[mode];
    const position = path.indexOf(stage);
    const destinations: Record<string, number> = {
      ArrowUp: Math.max(0, position - 1),
      ArrowDown: Math.min(path.length - 1, position + 1),
      Home: 0,
      End: path.length - 1,
    };
    if (!(event.key in destinations)) return;
    event.preventDefault();
    const next = path[destinations[event.key]];
    select(next);
    root
      .querySelector<HTMLAnchorElement>(`[data-case="${selected}"] [data-flow-select="${next}"]`)!
      .focus({ preventScroll: true });
  });

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
