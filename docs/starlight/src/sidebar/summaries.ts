import { tooltip } from '../ui/tooltip';

const popup = document.querySelector<HTMLElement>('#rf-sidebar-tooltip');
const links = [
  ...document.querySelectorAll<HTMLAnchorElement>('.sidebar-content a[data-sidebar-summary]'),
];

if (popup && links.length) {
  const title = popup.querySelector<HTMLElement>('[data-summary-title]')!;
  const text = popup.querySelector<HTMLElement>('[data-summary-text]')!;
  const sidebar = document.querySelector<HTMLElement>('#starlight__sidebar')!;
  const note = tooltip({
    popup,
    root: sidebar,
    selector: 'a[data-sidebar-summary]',
    placement: 'sidebar',
    delay: 160,
    clip: () => sidebar,
    render: (link) => {
      title.textContent = link.dataset.sidebarLabel!;
      text.textContent = link.dataset.sidebarSummary!;
    },
  });
  sidebar.addEventListener(
    'toggle',
    (event) => {
      if ((event as ToggleEvent).newState === 'closed') note.close();
      else note.reposition();
    },
    true,
  );
}
