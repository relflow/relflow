# relflow logo

The selected mark comes from the `logo-rework` worktree:
`docs/assets/branding/explorations/simplified/marks/soft-leaves.{light,dark}.svg`.
The exploration README and its `inks.json` and `spacing.json` identify the
approved Black pine ink, closer B spacing, upright trunks, and soft canopy.
The two `src/assets/branding/mark.*.svg` files are exact copies of those exports.

`header.typ` composes the selected mark with the matching Libertinus Serif
Semibold wordmark, using the selected preview's relative tracking and inks.
It removes the icon's background tile for the documentation's neutral header.
Glyphs remain vector outlines, with no browser font dependency. The source font
and its license already live in `docs/assets/fonts/`.

From the repository root, regenerate the two header assets with:

```sh
python3 docs/starlight/branding/render.py
```

The renderer inlines the mark's paths alongside the outlined wordmark, preserving
sharp scaling without embedded bitmap artwork. The favicon uses the exact
selected mark geometry and switches between its
original light and dark fills with `prefers-color-scheme`. Normal documentation
builds consume the prepared SVGs and do not require Typst or a font installation.
