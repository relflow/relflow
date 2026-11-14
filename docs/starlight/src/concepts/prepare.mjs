// @ts-check

// This inline bootstrap executes while its showcase is being parsed. Keep it
// self-contained so selection geometry never waits for a deferred module.
(() => {
  const script = document.currentScript;
  if (!(script instanceof HTMLScriptElement) || !script.parentElement) return;
  const root = script.parentElement;
  /** @type {{id: string; panel: string}[]} Authored by ExampleSelection.astro. */
  const choices = JSON.parse(script.dataset.choices || '[]');
  if (!choices.length) return;
  const persistent = script.hasAttribute('data-persistent');
  let hash = '';
  /** @type {string | null} */
  let saved = null;
  try {
    hash = decodeURIComponent(location.hash.slice(1));
  } catch {
    /* A malformed fragment does not prevent query or saved selection. */
  }
  if (persistent) {
    try {
      saved = localStorage.getItem('relflow:use-case');
    } catch {
      /* Explicit links still select a case when storage is unavailable. */
    }
  }
  const requested = persistent ? new URL(location.href).searchParams.get('case') : null;
  const fromHash = choices
    .filter((choice) => hash === choice.panel || hash.startsWith(`${choice.panel}-`))
    .sort((left, right) => right.panel.length - left.panel.length)[0];
  const selected =
    fromHash ||
    choices.find((choice) => choice.id === requested) ||
    choices.find((choice) => choice.id === saved) ||
    choices[0];
  root.dataset.initialPanel = selected.panel;
  root.dataset.prepared = '';
  const style = document.createElement('style');
  style.textContent = `[data-initial-panel="${CSS.escape(selected.panel)}"]:not([data-ready]) :is([data-case], [data-type-example]):not(#${CSS.escape(selected.panel)}) { display: none; }`;
  document.head.append(style);
})();
