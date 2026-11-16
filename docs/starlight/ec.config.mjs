import { codeThemes } from './src/styles/code-themes.mjs';
import { configNotes } from './src/code/config-notes.mjs';
import { modelCode } from './src/model-explainer/code.mjs';

export default {
  plugins: [configNotes(), modelCode()],
  themes: codeThemes,
  // Keep explicit palette colors identical in code blocks and live previews.
  minSyntaxHighlightingColorContrast: 0,
  defaultProps: { frame: 'none' },
  styleOverrides: {
    borderRadius: '0.35rem',
    codeFontSize: '0.84rem',
    codeLineHeight: '1.65',
  },
};
