# RelFlow branding

The logo uses one softly lit canopy shared by the `r` and `f` stems.

Logo sources and exports live in `logos/`; banner sources and exports live in
`banners/`. `logos/logo.svg` and `banners/banner.svg` switch between the warm
ivory light palette and Forest dark palette using `prefers-color-scheme`.
Each contains one vector drawing, with light colors as the fallback. The banner
lettering is outlined, so viewers do not need the original font installed.

The `.light.svg` and `.dark.svg` files have fixed colors. The repository README
selects these with GitHub's supported [`<picture>` markup](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/quickstart-for-writing-on-github),
so the banner follows the reader's GitHub appearance setting.

PNGs use fixed light or dark colors and are rendered at 144 ppi:

- `logos/logo.light.png` and `logos/logo.dark.png`: 390 × 390 pixels.
- `banners/banner.light.png` and `banners/banner.dark.png`: 2240 × 600 pixels.

To regenerate all six SVGs and four PNGs, install Typst, then run from the
repository root:

```sh
python3 docs/branding/render.py
```

The geometry and palettes live in `logos/logo.typ`; the banner layout lives in
`banners/banner.typ`. The renderer derives the adaptive SVG's dark colors from
the two Typst renders, so there is no separate palette to maintain.

The wordmark uses Libertinus Serif Semibold (weight 600), which is
[embedded in the standard Typst CLI](https://typst.app/docs/reference/text/text/#parameters-font).
The renderer disables system font discovery, so no font files or extra font
installation are needed.
