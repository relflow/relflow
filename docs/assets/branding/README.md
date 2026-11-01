# relflow branding

The **official relflow logo** has fine `rf` branches in front of staggered green
columns, with a blurred duplicate of the columns beneath them at **35% opacity**.
The glow expands the columns by 1.8 pt and blurs them by 3 pt in
the 195 pt logo. The solid columns and branches remain fully opaque and sharp.
The same effect scales with the logo in banners and previews.
Every vertical gap between stacked column segments is 0.052 cm in the native
2.6 cm canvas, equivalent to 3.9 pt in the 195 pt logo.

All main light, dark, and adaptive branding exports use this design. The dark
version uses the existing Forest palette: background `#192d24`, columns and
glow `#465f4b`, and branches and lettering `#dce5ce`. The light palette remains
ivory `#f2f0e6`, sage `#a5b989`, and ink `#234631`.

Sources and exports are grouped in `logos/`, `banners/`, and `previews/`.
`logos/logo.svg`, `banners/banner.svg`, and `previews/preview.svg` switch between
the warm ivory light palette and Forest dark palette using `prefers-color-scheme`.
Each contains one vector drawing, with light colors as the fallback. The glow
uses live SVG filters and shares the column paths; it changes color with the
theme. The banner lettering is outlined, so viewers do not need the original
font installed.

The `.light.svg` and `.dark.svg` files have fixed colors. Documentation uses
`logos/logo.navbar.svg`, which retains the dark palette's columns and branches
with a transparent background. The navbar and home-page wordmark use a locally
served copy of Libertinus Serif Semibold, matching the banner's weight,
tracking, and disabled ligatures. The repository README uses the light/dark
banners; preview artwork remains available for sharing.

PNGs use fixed light or dark colors and are rendered at 144 ppi:

- `logos/logo.light.png` and `logos/logo.dark.png`: 390 × 390 pixels.
- `banners/banner.light.png` and `banners/banner.dark.png`: 2240 × 600 pixels.
- `previews/preview.light.png` and `previews/preview.dark.png`: 1280 × 640 pixels.

The previews center the banner artwork with extra space on a solid background
for sharing.

To regenerate all ten SVGs and six PNGs, install Typst and `rsvg-convert`, then run from the
repository root:

```sh
python3 docs/assets/branding/render.py
```

The palettes and composition live in `logos/logo.typ`. `logos/branches.typ`
owns the fine branches; `logos/canopy.typ` owns the column profile
and division centers, with one shared vertical gap. The banner layout lives in
`banners/banner.typ`; `previews/preview.typ` reuses the banner artwork in a roomier
layout. The renderer derives the adaptive SVG's dark colors from the two Typst
renders, so there is no separate palette to maintain. `render.py` owns the 35% glow:
it adds the effect inside the logo's native canvas after compiling the Typst
geometry. It then renders the finished SVGs to PNG, so both formats include
the same glow. Direct Typst compilation produces the geometry before the glow.

The wordmark uses Libertinus Serif Semibold (weight 600), which is
[embedded in the standard Typst CLI](https://typst.app/docs/reference/text/text/#parameters-font).
The renderer disables system font discovery, so no extra font installation is
needed for branding exports. The docs font and its license are stored in
[`docs/assets/fonts`](../fonts/README.md).
