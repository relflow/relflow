import { tooltip } from '../ui/tooltip';

const notes = [...document.querySelectorAll<HTMLElement>('.rf-code-note[data-note-text]')];

if (notes.length && !document.getElementById('rf-code-tooltip')) {
  const popup = document.createElement('div');
  popup.id = 'rf-code-tooltip';
  popup.className = 'rf-code-tooltip';
  popup.setAttribute('role', 'tooltip');
  popup.hidden = true;
  const heading = document.createElement('strong');
  heading.className = 'rf-code-tooltip__title';
  const explanation = document.createElement('p');
  explanation.className = 'rf-code-tooltip__text';
  popup.append(heading, explanation);
  document.body.append(popup);
  for (const note of notes) {
    note.tabIndex = 0;
    note.setAttribute('role', 'button');
    note.setAttribute(
      'aria-label',
      `${note.textContent}: ${note.dataset.noteTitle || 'explanation'}`,
    );
    note.setAttribute('aria-controls', popup.id);
    note.setAttribute('aria-expanded', 'false');
  }
  tooltip({
    popup,
    root: document.body,
    selector: '.rf-code-note[data-note-text]',
    delay: 0,
    activate: true,
    lockPinned: true,
    clip: (trigger) => trigger.closest('pre'),
    render: (trigger) => {
      heading.textContent = trigger.dataset.noteTitle || trigger.textContent;
      explanation.textContent = trigger.dataset.noteText!;
    },
  });
}
