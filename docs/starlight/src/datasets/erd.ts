import { tooltip } from '../ui/tooltip';

const diagrams = [...document.querySelectorAll<HTMLElement>('[data-erd]')];

/** Measure the HTML cards so relationships stay attached across responsive layouts. */
function connect(figure: HTMLElement) {
  const canvas = figure.querySelector<HTMLElement>('.erd-canvas')!;
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const vertical = matchMedia('(max-width: 600px)').matches;
  const scroll = figure.querySelector<HTMLElement>('.erd-scroll')!;
  const overflow = bounds.width > scroll.clientWidth + 2;
  const svg = figure.querySelector<SVGSVGElement>('.erd-edges')!;
  const geometry = [...svg.querySelectorAll<SVGGElement>('[data-edge-from]')].map((edge) => {
    const from = canvas.querySelector<HTMLElement>(
      `[data-entity="${CSS.escape(edge.dataset.edgeFrom!)}"]`,
    )!;
    const to = canvas.querySelector<HTMLElement>(
      `[data-entity="${CSS.escape(edge.dataset.edgeTo!)}"]`,
    )!;
    const source = edge.dataset.edgeFromColumn
      ? from.querySelector<HTMLElement>(
          `[data-column="${CSS.escape(edge.dataset.edgeFromColumn)}"]`,
        )!
      : from.querySelector<HTMLElement>('header')!;
    const destination = edge.dataset.edgeToColumn
      ? to.querySelector<HTMLElement>(`[data-column="${CSS.escape(edge.dataset.edgeToColumn)}"]`)!
      : to.querySelector<HTMLElement>('header')!;
    return {
      edge,
      parent: source.getBoundingClientRect(),
      child: destination.getBoundingClientRect(),
    };
  });
  // Collect every anchor before changing SVG attributes or readiness styles.
  figure.querySelector<HTMLElement>('[data-erd-scroll-hint]')!.hidden = !overflow;
  svg.setAttribute('viewBox', `0 0 ${bounds.width} ${bounds.height}`);
  for (const { edge, parent, child } of geometry) {
    const direction = vertical || child.left >= parent.right ? 1 : -1;
    const x1 = (vertical || direction < 0 ? parent.left : parent.right) - bounds.left;
    const y1 = parent.top + parent.height / 2 - bounds.top;
    const x2 = (vertical || direction > 0 ? child.left : child.right) - bounds.left;
    const y2 = child.top + child.height / 2 - bounds.top;
    const elbow = vertical ? Math.max(7, Math.min(x1, x2) - 12) : (x1 + x2) / 2;
    edge
      .querySelector('.erd-edge-line')!
      .setAttribute('d', `M ${x1} ${y1} H ${elbow} V ${y2} H ${x2}`);
    const bar = x1 + (vertical ? -5 : direction * 7);
    edge.querySelector('.erd-edge-one')!.setAttribute('d', `M ${bar} ${y1 - 5} V ${y1 + 5}`);
    // A circle and crow's foot mark zero-or-many child records.
    const foot = x2 - direction * 9;
    edge
      .querySelector('.erd-edge-many')!
      .setAttribute(
        'd',
        edge.dataset.edgeCardinality === 'one'
          ? `M ${x2 - direction * 7} ${y2 - 5} V ${y2 + 5}`
          : `M ${x2} ${y2 - 5} L ${foot} ${y2} L ${x2} ${y2 + 5} M ${foot} ${y2} H ${x2} M ${x2 - direction * 24} ${y2} a 3.5 3.5 0 1 0 7 0 a 3.5 3.5 0 1 0 -7 0`,
      );
  }
  figure.dataset.erdReady = '';
}

const enhanced = new WeakSet<HTMLElement>();

function enhance(root: HTMLElement) {
  if (enhanced.has(root)) return;
  enhanced.add(root);
  const popup = root.querySelector<HTMLElement>('[data-erd-tooltip]')!;
  const name = popup.querySelector<HTMLElement>('[data-note-name]')!;
  const meta = popup.querySelector<HTMLElement>('[data-note-meta]')!;
  const description = popup.querySelector<HTMLElement>('[data-note-description]')!;
  function highlight(button?: HTMLElement) {
    for (const figure of root.querySelectorAll<HTMLElement>('[data-erd]')) {
      const entity = button?.closest<HTMLElement>('[data-entity]');
      for (const edge of figure.querySelectorAll<SVGGElement>('[data-edge-from]')) {
        const linked =
          entity &&
          ((edge.dataset.edgeFrom === entity.dataset.entity &&
            edge.dataset.edgeFromColumn === button?.dataset.column) ||
            (edge.dataset.edgeTo === entity.dataset.entity &&
              edge.dataset.edgeToColumn === button?.dataset.column));
        edge.toggleAttribute('data-edge-active', Boolean(linked));
      }
    }
  }

  tooltip({
    popup,
    root,
    selector: '.erd-column',
    placement: 'start',
    activate: true,
    clip: (button) => button.closest('.erd-scroll'),
    render: (button) => {
      name.textContent = button.dataset.columnName!;
      meta.textContent = `${button.dataset.columnType} · ${button.dataset.columnRole}`;
      description.textContent = button.dataset.columnNote!;
    },
    change: highlight,
  });
}

if (diagrams.length) {
  const observed = new Set<HTMLElement>();
  const pending = new Set<HTMLElement>();
  let frame: number | undefined;

  function schedule(figure: HTMLElement) {
    if (!observed.has(figure)) return;
    pending.add(figure);
    if (frame !== undefined) return;
    frame = requestAnimationFrame(() => {
      frame = undefined;
      for (const figure of pending) if (observed.has(figure)) connect(figure);
      pending.clear();
    });
  }

  const observer = new ResizeObserver((entries) => {
    // Canvas and viewport can report the same change; measure a figure once per frame.
    for (const entry of entries) schedule(entry.target.closest<HTMLElement>('[data-erd]')!);
  });

  const visibility = new IntersectionObserver(
    (entries) => {
      for (const entry of entries) {
        const figure = entry.target as HTMLElement;
        const canvas = figure.querySelector<HTMLElement>('.erd-canvas')!;
        const viewport = figure.querySelector<HTMLElement>('.erd-scroll')!;
        if (!entry.isIntersecting) {
          if (observed.delete(figure)) {
            observer.unobserve(canvas);
            observer.unobserve(viewport);
            pending.delete(figure);
          }
          continue;
        }
        enhance(figure.closest<HTMLElement>('[data-data-explorer]')!);
        if (!observed.has(figure)) {
          observed.add(figure);
          observer.observe(canvas);
          observer.observe(viewport);
        }
        schedule(figure);
      }
    },
    { rootMargin: '256px 0px' },
  );

  // Intersection changes also cover hidden tabs, native details, and case switches.
  diagrams.forEach((figure) => visibility.observe(figure));
  document.fonts.ready.then(() => observed.forEach(schedule));
}
