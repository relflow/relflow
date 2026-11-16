export interface ColorScheme {
  bg: string;
  surface: string;
  sidebar: string;
  code: string;
  ink: string;
  text: string;
  muted: string;
  border: string;
  accent: string;
  accentSoft: string;
  keyword: string;
  string: string;
  number: string;
  comment: string;
  tooltip: string;
}

export const paletteFamilies = ['Slate', 'Blue', 'Teal', 'Green', 'Olive & Gold', 'Orange & Rust', 'Rose & Red', 'Purple'] as const;
export type PaletteFamily = typeof paletteFamilies[number];

export interface Palette {
  id: string;
  name: string;
  family: PaletteFamily;
  description: string;
  light: ColorScheme;
  dark: ColorScheme;
}

function channels(color: string): number[] {
  return [1, 3, 5].map((position) => Number.parseInt(color.slice(position, position + 2), 16));
}

function mix(first: string, second: string, weight: number): string {
  const right = channels(second);
  return `#${channels(first).map((value, index) => Math.round(value * (1 - weight) + right[index] * weight).toString(16).padStart(2, '0')).join('')}`;
}

/** WCAG relative-luminance contrast for opaque sRGB colors. */
export function contrast(first: string, second: string): number {
  function luminance(color: string): number {
    const linear = channels(color).map((channel) => {
      const value = channel / 255;
      return value <= 0.04045 ? value / 12.92 : ((value + 0.055) / 1.055) ** 2.4;
    });
    return linear[0] * 0.2126 + linear[1] * 0.7152 + linear[2] * 0.0722;
  }
  const values = [luminance(first), luminance(second)].sort((a, b) => a - b);
  return (values[1] + 0.05) / (values[0] + 0.05);
}

/** All options share neutral surfaces, text, and semantic syntax values. */
const neutral: { light: ColorScheme; dark: ColorScheme } = {
  "light": {
    "bg": "#ffffff",
    "surface": "#f4f4f4",
    "sidebar": "#ffffff",
    "code": "#f8f8f8",
    "ink": "#202430",
    "text": "#303644",
    "muted": "#646b7c",
    "border": "#dddddd",
    "accent": "#3d48a8",
    "accentSoft": "#f0f0f0",
    "keyword": "#4654bb",
    "string": "#286b52",
    "number": "#955127",
    "comment": "#70778b",
    "tooltip": "#ffffff"
  },
  "dark": {
    "bg": "#16181e",
    "surface": "#202329",
    "sidebar": "#16181e",
    "code": "#1c1f24",
    "ink": "#f4f4f8",
    "text": "#e6e7ee",
    "muted": "#b3b5c3",
    "border": "#36393f",
    "accent": "#b9c2ff",
    "accentSoft": "#26292f",
    "keyword": "#b0baff",
    "string": "#9fd5b4",
    "number": "#e8bd97",
    "comment": "#a1a6b8",
    "tooltip": "#16181e"
  }
};

function readable(color: string, mode: 'light' | 'dark'): string {
  const base = neutral[mode];
  const backgrounds = [base.bg, base.surface, base.sidebar, base.code, base.accentSoft, base.tooltip];
  const passes = (candidate: string) => backgrounds.every((background) => contrast(candidate, background) >= 4.6);
  if (passes(color)) return color;
  const anchor = mode === 'dark' ? '#ffffff' : '#080b0c';
  let lower = 0;
  let upper = 1;
  for (let step = 0; step < 20; step++) {
    const weight = (lower + upper) / 2;
    if (passes(mix(color, anchor, weight))) upper = weight;
    else lower = weight;
  }
  return mix(color, anchor, upper);
}

interface Accent {
  id: string;
  name: string;
  family: PaletteFamily;
  description: string;
  light: string;
  dark: string;
  lightKeyword?: string;
  darkKeyword?: string;
}

const accents: Accent[] = [
  {"id": "current-neutral", "name": "Restrained Indigo", "family": "Purple", "description": "A restrained indigo accent.", "light": "#3d48a8", "dark": "#b9c2ff", "lightKeyword": "#4654bb", "darkKeyword": "#b0baff"},
  {"id": "current-blue", "name": "Soft Blue", "family": "Blue", "description": "A soft blue accent.", "light": "#2c568f", "dark": "#a6c9eb", "lightKeyword": "#335d9d", "darkKeyword": "#accef4"},
  {"id": "graphite", "name": "Graphite", "family": "Slate", "description": "A neutral graphite accent.", "light": "#44484e", "dark": "#aeb8c6"},
  {"id": "chalk", "name": "Chalk", "family": "Slate", "description": "A balanced gray accent.", "light": "#5d5e63", "dark": "#bcbdbf"},
  {"id": "ink-paper", "name": "Ink", "family": "Slate", "description": "A deep blue-black accent.", "light": "#27384b", "dark": "#8da5c1"},
  {"id": "warm-slate", "name": "Warm Slate", "family": "Slate", "description": "A subdued violet-gray accent.", "light": "#6a5868", "dark": "#bda5ba"},
  {"id": "storm", "name": "Storm", "family": "Slate", "description": "A muted steel-blue accent.", "light": "#45657a", "dark": "#8bb0c7"},
  {"id": "sandstone", "name": "Sandstone", "family": "Slate", "description": "A soft brown-gray accent.", "light": "#7c614d", "dark": "#c6a88e"},
  {"id": "atlantic", "name": "Atlantic", "family": "Blue", "description": "A clear ocean-blue accent.", "light": "#235b8c", "dark": "#6faee6"},
  {"id": "glacier", "name": "Glacier", "family": "Blue", "description": "A cool cyan-blue accent.", "light": "#276c91", "dark": "#69bfeb"},
  {"id": "deep-sea", "name": "Deep Sea", "family": "Blue", "description": "A concentrated navy accent.", "light": "#173c76", "dark": "#6f98de"},
  {"id": "cobalt", "name": "Cobalt", "family": "Blue", "description": "A vivid cobalt-blue accent.", "light": "#1e55c6", "dark": "#719fff"},
  {"id": "ultramarine", "name": "Ultramarine", "family": "Blue", "description": "A saturated blue-violet accent.", "light": "#4439be", "dark": "#8d8aff"},
  {"id": "basalt", "name": "Basalt", "family": "Teal", "description": "A deep mineral-teal accent.", "light": "#004f4d", "dark": "#6cb5ad"},
  {"id": "juniper", "name": "Juniper", "family": "Teal", "description": "A muted evergreen-teal accent.", "light": "#26635b", "dark": "#7bb7a6"},
  {"id": "petrol", "name": "Petrol", "family": "Teal", "description": "A dense blue-teal accent.", "light": "#15576c", "dark": "#5fb8cf"},
  {"id": "lagoon", "name": "Lagoon", "family": "Teal", "description": "A clear turquoise accent.", "light": "#087c78", "dark": "#4bc5c0"},
  {"id": "electric-teal", "name": "Electric Teal", "family": "Teal", "description": "A sharper, brighter teal accent.", "light": "#008e83", "dark": "#2dc9b5"},
  {"id": "sea-glass", "name": "Sea Glass", "family": "Teal", "description": "A soft aqua-green accent.", "light": "#3c756c", "dark": "#8acfc0"},
  {"id": "pine-needle", "name": "Pine Needle", "family": "Green", "description": "A dark pine-green accent.", "light": "#185a3c", "dark": "#68b78a"},
  {"id": "moss", "name": "Moss", "family": "Green", "description": "A subdued woodland-green accent.", "light": "#52612b", "dark": "#a4b967"},
  {"id": "sage", "name": "Sage", "family": "Green", "description": "A quiet gray-green accent.", "light": "#536a57", "dark": "#9cbaa1"},
  {"id": "fern", "name": "Fern", "family": "Green", "description": "A fresh leaf-green accent.", "light": "#34742c", "dark": "#7ec36b"},
  {"id": "signal-green", "name": "Signal Green", "family": "Green", "description": "A strong, saturated green accent.", "light": "#087e4b", "dark": "#46c78b"},
  {"id": "emerald", "name": "Emerald", "family": "Green", "description": "A clear jewel-green accent.", "light": "#126b4f", "dark": "#5ad4a1"},
  {"id": "olive", "name": "Olive Grove", "family": "Olive & Gold", "description": "An earthy olive accent.", "light": "#666022", "dark": "#bbb263"},
  {"id": "saffron", "name": "Saffron", "family": "Olive & Gold", "description": "A warm golden-yellow accent.", "light": "#94701b", "dark": "#e0b84e"},
  {"id": "ochre", "name": "Ochre", "family": "Olive & Gold", "description": "A muted yellow-earth accent.", "light": "#8b6b2e", "dark": "#c9a45d"},
  {"id": "midnight-gold", "name": "Midnight Gold", "family": "Olive & Gold", "description": "A deeper amber-gold accent.", "light": "#9d7421", "dark": "#d5ad57"},
  {"id": "chartreuse", "name": "Chartreuse", "family": "Olive & Gold", "description": "A bright yellow-green accent.", "light": "#64751d", "dark": "#bdcd58"},
  {"id": "brass", "name": "Brass", "family": "Olive & Gold", "description": "A restrained antique-gold accent.", "light": "#7d6a3b", "dark": "#bfa96f"},
  {"id": "terracotta", "name": "Terracotta", "family": "Orange & Rust", "description": "A warm clay-red accent.", "light": "#a64d35", "dark": "#e58d70"},
  {"id": "ember", "name": "Ember", "family": "Orange & Rust", "description": "A strong burnt-orange accent.", "light": "#b45419", "dark": "#f49b52"},
  {"id": "cedar", "name": "Cedar", "family": "Orange & Rust", "description": "A reddish-brown accent.", "light": "#8f4838", "dark": "#cf8e78"},
  {"id": "copper", "name": "Copper", "family": "Orange & Rust", "description": "A vivid reddish-copper accent.", "light": "#a95935", "dark": "#e6a26d"},
  {"id": "apricot", "name": "Apricot", "family": "Orange & Rust", "description": "A softer peach-orange accent.", "light": "#a45d3d", "dark": "#ecb088"},
  {"id": "burnt-sienna", "name": "Burnt Sienna", "family": "Orange & Rust", "description": "A concentrated red-earth accent.", "light": "#a24628", "dark": "#e27953"},
  {"id": "rosewood", "name": "Rosewood", "family": "Rose & Red", "description": "A subdued dusty-rose accent.", "light": "#9a5660", "dark": "#d79da7"},
  {"id": "mulberry", "name": "Mulberry", "family": "Rose & Red", "description": "A saturated berry-pink accent.", "light": "#a03068", "dark": "#ea7bab"},
  {"id": "crimson", "name": "Crimson", "family": "Rose & Red", "description": "A strong crimson-red accent.", "light": "#b12b42", "dark": "#f17c8d"},
  {"id": "coral", "name": "Coral", "family": "Rose & Red", "description": "A clear red-coral accent.", "light": "#b64d4d", "dark": "#fa9391"},
  {"id": "garnet", "name": "Garnet", "family": "Rose & Red", "description": "A deeper wine-red accent.", "light": "#8c304b", "dark": "#d87898"},
  {"id": "raspberry", "name": "Raspberry", "family": "Rose & Red", "description": "A vivid pink-red accent.", "light": "#aa2861", "dark": "#f17fb4"},
  {"id": "current-indigo", "name": "Royal Indigo", "family": "Purple", "description": "A stronger royal-indigo accent.", "light": "#4a3b9c", "dark": "#9f91f5"},
  {"id": "iris", "name": "Iris", "family": "Purple", "description": "A clear lavender-violet accent.", "light": "#6751a5", "dark": "#ad94ed"},
  {"id": "plum", "name": "Plum", "family": "Purple", "description": "A muted plum-purple accent.", "light": "#80516f", "dark": "#d098bf"},
  {"id": "orchid", "name": "Orchid", "family": "Purple", "description": "A brighter purple-pink accent.", "light": "#974c91", "dark": "#db88d4"},
  {"id": "aubergine", "name": "Aubergine", "family": "Purple", "description": "A deep eggplant-purple accent.", "light": "#634267", "dark": "#b18bb8"}
];

export const palettes: Palette[] = accents.map((accent) => ({
  id: accent.id,
  name: accent.name,
  family: accent.family,
  description: accent.description,
  light: { ...neutral.light, accent: readable(accent.light, 'light'), keyword: readable(accent.lightKeyword ?? accent.light, 'light') },
  dark: { ...neutral.dark, accent: readable(accent.dark, 'dark'), keyword: readable(accent.darkKeyword ?? accent.dark, 'dark') },
}));
