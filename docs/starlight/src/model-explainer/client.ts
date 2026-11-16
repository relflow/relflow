import type { Explanation } from './explanation';
import { sitePath } from '../site/urls.mjs';

interface ModelData {
  stages: { id: string; paths: string[] }[];
  explanations: Record<string, Explanation>;
}

for (const root of document.querySelectorAll<HTMLElement>('[data-model-explainer]')) {
  const data: ModelData = JSON.parse(root.querySelector('[data-model-data]')!.textContent!);
  const steps = [...root.querySelectorAll<HTMLButtonElement>('[data-model-step]')];
  const declarations = [...root.querySelectorAll<HTMLButtonElement>('[data-model-select]')];
  const lines = [...root.querySelectorAll<HTMLElement>('.ec-line[data-model-paths]')];
  const sourceLines = [
    ...root.querySelectorAll<HTMLElement>('.me-source .ec-line[data-model-stages]'),
  ].map((line) => {
    const text = line.querySelector<HTMLElement>('.code')!.textContent!;
    return { stages: line.dataset.modelStages!.split(' '), text: text === '\n' ? '' : text };
  });
  const status = root.querySelector<HTMLElement>('[data-model-status]')!;
  const mobile = window.matchMedia('(max-width: 600px)');
  let stage = data.stages[0];
  let selected = '/';

  function select(path: string, announce = true) {
    selected = stage.paths.includes(path) ? path : '/';
    const detail = data.explanations[selected];
    root.dataset.modelFocus = detail.kind;
    for (const button of declarations)
      button.setAttribute('aria-pressed', String(button.dataset.modelSelect === selected));
    for (const line of lines)
      line.toggleAttribute(
        'data-model-active',
        line.dataset.modelPaths!.split(' ').includes(selected),
      );
    for (const name of [
      'name',
      'path',
      'declaration',
      'description',
      'representation',
      'context',
      'learning',
      'output',
      'address',
      'requirement',
    ] as const) {
      root.querySelector<HTMLElement>(`[data-model-${name}]`)!.textContent = detail[name];
    }
    root.querySelector<HTMLElement>('[data-model-kind]')!.textContent =
      detail.kind === 'target'
        ? `${detail.type} answer`
        : detail.kind === 'model'
          ? 'Model'
          : detail.kind === 'branch'
            ? 'Record collection'
            : `${detail.type} input`;
    root.querySelector<HTMLElement>('[data-model-address-wrap]')!.hidden = !detail.address;
    root.querySelector<HTMLElement>('[data-model-requirement]')!.hidden = !detail.requirement;
    root.querySelector<HTMLAnchorElement>('[data-model-reference]')!.href = sitePath(
      detail.reference,
      import.meta.env.BASE_URL,
    );
    if (announce)
      status.textContent = `${detail.path === '/' ? detail.name : detail.path} selected. ${detail.kind === 'target' ? 'This answer is hidden from input context.' : detail.kind === 'branch' ? 'Inspect how these records contribute context.' : 'Inspect its representation and learning role.'}`;
  }

  function change(id: string, announce = true) {
    const next = data.stages.find((item) => item.id === id);
    if (!next) return;
    document.dispatchEvent(new Event('code-notes:close'));
    stage = next;
    root.dataset.modelStage = stage.id;
    for (const step of steps)
      step.setAttribute('aria-pressed', String(step.dataset.modelStep === stage.id));
    const copy = root.querySelector<HTMLButtonElement>('.me-source .copy button');
    if (copy)
      copy.dataset.code = sourceLines
        .filter((line) => line.stages.includes(stage.id))
        .map((line) => line.text)
        .join('\n');
    const label = steps
      .find((step) => step.dataset.modelStep === stage.id)!
      .querySelector('strong')!.textContent!;
    root.querySelector<HTMLElement>('[data-model-copy-hint]')!.textContent =
      `· ${label.toLowerCase()}`;
    select(selected, false);
    if (announce)
      status.textContent = `${label} selected. The schema, available context, and target objectives now match this step.`;
  }

  root.addEventListener('click', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    if (target?.closest('[data-model-back]')) {
      const button = declarations.find((item) => item.dataset.modelSelect === selected)!;
      button.focus({ preventScroll: true });
      button.scrollIntoView({ block: 'center' });
      return;
    }
    const step = target?.closest<HTMLElement>('[data-model-step]');
    if (step) {
      change(step.dataset.modelStep!);
      return;
    }
    const node = target?.closest<HTMLElement>('[data-model-select]');
    const line = target?.closest<HTMLElement>('.ec-line[data-model-paths]');
    if (node) {
      select(node.dataset.modelSelect!);
      if (mobile.matches) {
        root.querySelector<HTMLElement>('[data-model-name]')!.focus({ preventScroll: true });
        root.querySelector<HTMLElement>('.me-insight')!.scrollIntoView({ block: 'start' });
      }
    } else if (line) select(line.dataset.modelPaths!.split(' ')[0]);
  });
  root.addEventListener('focusin', (event) => {
    const target = event.target instanceof Element ? event.target : null;
    const line = target?.closest<HTMLElement>('.ec-line[data-model-paths]');
    if (line) select(line.dataset.modelPaths!.split(' ')[0], false);
  });
  root.addEventListener('keydown', (event) => {
    const target =
      event.target instanceof Element
        ? event.target.closest<HTMLButtonElement>('[data-model-step]')
        : null;
    if (!target || event.altKey || event.ctrlKey || event.metaKey) return;
    const index = steps.indexOf(target);
    const destinations: Record<string, number> = {
      ArrowLeft: (index + steps.length - 1) % steps.length,
      ArrowRight: (index + 1) % steps.length,
      Home: 0,
      End: steps.length - 1,
    };
    if (!(event.key in destinations)) return;
    event.preventDefault();
    const next = steps[destinations[event.key]];
    change(next.dataset.modelStep!);
    next.focus({ preventScroll: true });
  });
  for (const button of [...steps, ...declarations]) {
    button.disabled = false;
    button.setAttribute('aria-controls', `${root.id}-selected`);
  }
  change('start', false);
  root.dataset.modelReady = '';
}
