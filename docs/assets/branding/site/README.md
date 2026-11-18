# relflow logo

The selected soft-canopy mark is owned by the checked-in
`docs/starlight/src/assets/branding/mark.{light,dark}.svg` files. They preserve
the approved Black pine ink, upright trunks, closer spacing, and soft canopy.
These are the canonical site mark sources; regeneration needs no exploration
worktree or deleted source directory.

`header.typ` composes the selected mark with the matching Libertinus Serif
Semibold wordmark, using the selected preview's relative tracking and inks.
It removes the icon's background tile for the documentation's neutral header.
Glyphs remain vector outlines, with no browser font dependency. The source font
and its license already live in `docs/assets/fonts/`.

Prose mentions use a small inline wordmark in the same face. The prepared
`docs/starlight/src/assets/branding/wordmark.woff2` contains only the six letters in `relflow`,
with copyright and SIL Open Font License metadata retained. Regenerate this
subset from the existing source font at the repository root with:

```sh
uv tool run --from 'fonttools[woff]' pyftsubset \
  docs/assets/fonts/LibertinusSerif-Semibold.otf \
  --text=relflow --flavor=woff2 --layout-features= \
  --name-IDs=0,1,2,3,4,5,6,13,14 \
  --output-file=docs/starlight/src/assets/branding/wordmark.woff2
```

This is asset preparation, not part of the documentation build. The browser
downloads only the prepared subset. The wordmark’s typography and theme color
live in `docs/starlight/src/styles/site.css`; `docs/starlight/src/site/brand.mjs` wraps standalone prose
mentions in server-rendered HTML. Authored content stays plain text. Code,
commands, URLs, exact data, and recorded results retain their original markup.

From the repository root, regenerate the two header assets with:

```sh
python3 docs/assets/branding/site/render.py
```

The renderer inlines the mark's paths alongside the outlined wordmark, preserving
sharp scaling without embedded bitmap artwork. The favicon uses the exact
selected mark geometry and switches between its
original light and dark fills with `prefers-color-scheme`. Normal documentation
builds consume the prepared SVGs and do not require Typst or a font installation.

All branding authoring lives under `docs/assets/branding/`. Its root renderer
owns the package/share exports; this `site/` renderer composes the selected
website header and favicon from the canonical site marks. Prepared site assets
remain under `docs/starlight/src/assets/branding/` for the static build.
