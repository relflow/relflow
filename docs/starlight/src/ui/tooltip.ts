interface Tooltip {
  close(): boolean;
  contains(target: Node): boolean;
  reposition(target?: Node): void;
}

interface Options {
  popup: HTMLElement;
  root: HTMLElement;
  selector: string;
  render(trigger: HTMLElement): void;
  clip?(trigger: HTMLElement): Element | null;
  placement?: 'center' | 'start' | 'sidebar';
  delay?: number;
  activate?: boolean;
  lockPinned?: boolean;
  change?(trigger?: HTMLElement): void;
}

// Only open notes and pending intents participate in global coordination.
const tooltips = new Set<Tooltip>();
let pointer = { x: 0, y: 0 };
let dismissedAt: typeof pointer | undefined;

function hoverIntent(event: PointerEvent) {
  if (dismissedAt && (event.clientX !== dismissedAt.x || event.clientY !== dismissedAt.y)) {
    dismissedAt = undefined;
  }
  return !dismissedAt;
}

document.addEventListener(
  'pointermove',
  (event) => {
    pointer = { x: event.clientX, y: event.clientY };
  },
  true,
);

function closeAll(except?: Tooltip) {
  let dismissed = false;
  for (const tooltip of tooltips) if (tooltip !== except) dismissed = tooltip.close() || dismissed;
  return dismissed;
}

document.addEventListener('pointerdown', (event) => {
  if (!(event.target instanceof Node)) return;
  for (const tooltip of tooltips) if (!tooltip.contains(event.target)) tooltip.close();
});
document.addEventListener(
  'keydown',
  (event) => {
    if (event.key !== 'Escape') return;
    const previous = dismissedAt;
    // Removing an overlay can expose a different trigger under a stationary pointer.
    dismissedAt = { ...pointer };
    if (!closeAll()) {
      dismissedAt = previous;
      return;
    }
    event.preventDefault();
    event.stopPropagation();
  },
  true,
);
document.addEventListener(
  'scroll',
  (event) => {
    for (const tooltip of tooltips)
      tooltip.reposition(event.target instanceof Node ? event.target : undefined);
  },
  true,
);
// Selectors dismiss every note, including notes waiting for hover/focus timers.
document.addEventListener('code-notes:close', () => closeAll());
window.addEventListener('resize', () => closeAll(), { passive: true });
window.addEventListener('pagehide', () => closeAll());

/** One lifecycle for notes; each caller supplies its own content and trigger semantics. */
export function tooltip(options: Options) {
  const {
    popup,
    root,
    selector,
    render,
    clip,
    placement = 'center',
    delay = 120,
    activate = false,
    lockPinned = false,
    change,
  } = options;
  const events = new AbortController();
  const { signal } = events;
  let active: HTMLElement | undefined;
  let description: string | null = null;
  let hovered: HTMLElement | undefined;
  let pinned = false;
  let overPopup = false;
  let showTimer: ReturnType<typeof setTimeout> | undefined;
  let hideTimer: ReturnType<typeof setTimeout> | undefined;
  let focusFrame: number | undefined;
  if (typeof popup.showPopover === 'function') popup.setAttribute('popover', 'manual');

  function cancel() {
    clearTimeout(showTimer);
    clearTimeout(hideTimer);
    if (focusFrame !== undefined) cancelAnimationFrame(focusFrame);
    showTimer = hideTimer = undefined;
    focusFrame = undefined;
  }

  function close() {
    const dismissed = !!active || showTimer !== undefined || focusFrame !== undefined;
    cancel();
    if (active) {
      if (description === null) active.removeAttribute('aria-describedby');
      else active.setAttribute('aria-describedby', description);
      if (active.hasAttribute('aria-expanded')) active.setAttribute('aria-expanded', 'false');
      active.removeAttribute('data-note-open');
      change?.();
    }
    if (popup.hasAttribute('popover') && popup.matches(':popover-open')) popup.hidePopover();
    popup.hidden = true;
    active = undefined;
    pinned = overPopup = false;
    tooltips.delete(instance);
    return dismissed;
  }

  function position() {
    if (!active) return;
    const pane = clip?.(active)?.getBoundingClientRect();
    const rect = active.getBoundingClientRect();
    const left = Math.max(0, rect.left, pane?.left ?? 0);
    const right = Math.min(innerWidth, rect.right, pane?.right ?? innerWidth);
    const top = Math.max(0, rect.top, pane?.top ?? 0);
    const bottom = Math.min(innerHeight, rect.bottom, pane?.bottom ?? innerHeight);
    if (
      !active.getClientRects().length ||
      active.closest('[inert], [aria-hidden="true"]') ||
      getComputedStyle(active).visibility !== 'visible' ||
      left >= right ||
      top >= bottom
    ) {
      close();
      return;
    }
    const { width, height } = popup.getBoundingClientRect();
    const gutter = 12;
    const alongside =
      placement === 'sidebar' &&
      pane &&
      matchMedia('(min-width: 50em)').matches &&
      pane.right + width + gutter * 2 <= innerWidth;
    const below = bottom + 8 + height <= innerHeight - gutter || top < height + gutter + 8;
    const x = alongside
      ? pane.right + gutter
      : placement === 'center'
        ? (left + right - width) / 2
        : left;
    const y = alongside ? (top + bottom - height) / 2 : below ? bottom + 8 : top - height - 8;
    popup.style.left = `${Math.round(Math.max(gutter, Math.min(x, innerWidth - width - gutter)))}px`;
    popup.style.top = `${Math.round(Math.max(gutter, Math.min(y, innerHeight - height - gutter)))}px`;
    popup.dataset.side = below ? 'below' : 'above';
  }

  function show(trigger: HTMLElement, pin = false, hover = false) {
    if (hover && lockPinned && pinned && active !== trigger) return;
    cancel();
    if (active !== trigger) {
      close();
      closeAll(instance);
      active = trigger;
      description = trigger.getAttribute('aria-describedby');
      trigger.removeAttribute('title');
      render(trigger);
    }
    tooltips.add(instance);
    pinned = pin || pinned;
    popup.hidden = false;
    if (popup.hasAttribute('popover') && !popup.matches(':popover-open')) popup.showPopover();
    trigger.setAttribute('aria-describedby', popup.id);
    if (trigger.hasAttribute('aria-expanded')) trigger.setAttribute('aria-expanded', 'true');
    trigger.setAttribute('data-note-open', '');
    position();
    if (active) change?.(active);
  }

  function delayClose() {
    clearTimeout(showTimer);
    showTimer = undefined;
    clearTimeout(hideTimer);
    hideTimer = setTimeout(() => {
      hideTimer = undefined;
      if (
        pinned ||
        overPopup ||
        (active && (hovered === active || document.activeElement === active))
      )
        return;
      close();
    }, 180);
  }

  function toggle(trigger: HTMLElement) {
    if (active === trigger && pinned) close();
    else show(trigger, true);
  }

  function scheduleHover(trigger: HTMLElement) {
    hovered = trigger;
    cancel();
    tooltips.add(instance);
    showTimer = setTimeout(() => {
      showTimer = undefined;
      show(trigger, false, true);
    }, delay);
  }

  function triggerAt(target: EventTarget | null) {
    const trigger = target instanceof Element ? target.closest<HTMLElement>(selector) : null;
    return trigger && root.contains(trigger) ? trigger : undefined;
  }
  // Delegation keeps hidden examples free of per-token and per-column handlers.
  root.addEventListener(
    'pointerover',
    (event) => {
      const trigger = triggerAt(event.target);
      if (
        !trigger ||
        event.pointerType === 'touch' ||
        (event.relatedTarget instanceof Node && trigger.contains(event.relatedTarget)) ||
        !hoverIntent(event)
      )
        return;
      scheduleHover(trigger);
    },
    { signal },
  );
  root.addEventListener(
    'pointermove',
    (event) => {
      if (!dismissedAt || event.pointerType === 'touch' || !hoverIntent(event)) return;
      const trigger = triggerAt(event.target);
      if (trigger) scheduleHover(trigger);
    },
    { signal },
  );
  root.addEventListener(
    'pointerout',
    (event) => {
      const trigger = triggerAt(event.target);
      if (
        !trigger ||
        event.pointerType === 'touch' ||
        (event.relatedTarget instanceof Node && trigger.contains(event.relatedTarget))
      )
        return;
      if (hovered === trigger) hovered = undefined;
      delayClose();
    },
    { signal },
  );
  root.addEventListener(
    'focusin',
    (event) => {
      const trigger = triggerAt(event.target);
      if (!trigger) return;
      cancel();
      tooltips.add(instance);
      // Focus can scroll a clipped trigger into view before the next frame.
      focusFrame = requestAnimationFrame(() => {
        focusFrame = undefined;
        if (document.activeElement === trigger) show(trigger);
      });
    },
    { signal },
  );
  root.addEventListener(
    'focusout',
    (event) => {
      const trigger = triggerAt(event.target);
      if (!trigger) return;
      if (focusFrame !== undefined) cancelAnimationFrame(focusFrame);
      focusFrame = undefined;
      if (active === trigger) pinned = false;
      delayClose();
    },
    { signal },
  );
  if (activate) {
    root.addEventListener(
      'click',
      (event) => {
        const trigger = triggerAt(event.target);
        if (trigger) toggle(trigger);
      },
      { signal },
    );
    root.addEventListener(
      'keydown',
      (event) => {
        if (event.key !== 'Enter' && event.key !== ' ') return;
        const trigger = triggerAt(event.target);
        if (!trigger || trigger instanceof HTMLButtonElement) return;
        event.preventDefault();
        toggle(trigger);
      },
      { signal },
    );
  }
  popup.addEventListener(
    'pointerenter',
    () => {
      overPopup = true;
      clearTimeout(hideTimer);
    },
    { signal },
  );
  popup.addEventListener(
    'pointerleave',
    () => {
      overPopup = false;
      delayClose();
    },
    { signal },
  );
  const instance: Tooltip = {
    close,
    contains: (target) => !!active && (active.contains(target) || popup.contains(target)),
    reposition: (target) => {
      if (!target || !popup.contains(target)) position();
    },
  };
  return {
    close,
    reposition: () => position(),
    destroy: () => {
      close();
      events.abort();
    },
  };
}
