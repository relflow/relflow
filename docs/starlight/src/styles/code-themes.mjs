const colors = {
  dark: {
    text: '#e6e7ee',
    bg: '#1c1f24',
    surface: '#202329',
    border: '#36393f',
    keyword: '#68b78a',
    string: '#9fd5b4',
    number: '#e8bd97',
    comment: '#a1a6b8',
  },
  light: {
    text: '#303644',
    bg: '#f8f8f8',
    surface: '#f4f4f4',
    border: '#dddddd',
    keyword: '#185a3c',
    string: '#286b52',
    number: '#955127',
    comment: '#70778b',
  },
};

export const codeThemes = Object.entries(colors).map(([type, color]) => ({
  name: `relflow-${type}`,
  type,
  colors: {
    'editor.background': color.bg,
    'editor.foreground': color.text,
    'editorGroupHeader.tabsBackground': color.surface,
    'tab.activeBackground': color.bg,
    'tab.activeForeground': color.text,
    'tab.activeBorderTop': color.keyword,
    'panel.border': color.border,
  },
  tokenColors: [
    { scope: ['comment'], settings: { foreground: color.comment } },
    {
      scope: ['keyword', 'storage', 'entity.name', 'support.function', 'support.type'],
      settings: { foreground: color.keyword },
    },
    { scope: ['string'], settings: { foreground: color.string } },
    { scope: ['constant.numeric', 'constant.language'], settings: { foreground: color.number } },
  ],
}));
