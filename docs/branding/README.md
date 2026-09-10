# RelFlow branding

The logo uses one softly lit canopy shared by the `r` and `f` stems.

Sources and exports are grouped in `logos/`, `banners/`, and `previews/`.
`logos/logo.svg`, `banners/banner.svg`, and `previews/preview.svg` switch between
the warm ivory light palette and Forest dark palette using `prefers-color-scheme`.
Each contains one vector drawing, with light colors as the fallback. The banner
lettering is outlined, so viewers do not need the original font installed.

The `.light.svg` and `.dark.svg` files have fixed colors. The repository README
selects these with GitHub's supported [`<picture>` markup](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github),
so the banner follows the reader's GitHub appearance setting.

PNGs use fixed light or dark colors and are rendered at 144 ppi:

- `logos/logo.light.png` and `logos/logo.dark.png`: 390 × 390 pixels.
- `banners/banner.light.png` and `banners/banner.dark.png`: 2240 × 600 pixels.
- `previews/preview.light.png` and `previews/preview.dark.png`: 1280 × 640 pixels.

The previews center the banner artwork with extra space on a solid background
for sharing.

To regenerate all nine SVGs and six PNGs, install Typst, then run from the
repository root:

```sh
python3 docs/branding/render.py
```

The geometry and palettes live in `logos/logo.typ`; the banner layout lives in
`banners/banner.typ`. `previews/preview.typ` reuses the banner artwork in a roomier
layout. The renderer derives the adaptive SVG's dark colors from the two Typst
renders, so there is no separate palette to maintain.

The wordmark uses Libertinus Serif Semibold (weight 600), which is
[embedded in the standard Typst CLI](https://typst.app/docs/reference/text/text/#parameters-font).
The renderer disables system font discovery, so no font files or extra font
installation are needed.
