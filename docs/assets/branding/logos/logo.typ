#import "@preview/cetz:0.4.2"
#import "branches.typ": branches
#import "canopy.typ": canopy

#let palettes = (
  light: (background: rgb("#f2f0e6"), ink: rgb("#234631"), canopy: rgb("#a5b989")),
  dark: (background: rgb("#192d24"), ink: rgb("#dce5ce"), canopy: rgb("#465f4b")),
)

// Official logo geometry. The branding renderer adds the 35% SVG glow.
#let logo(theme: "light", size: 195pt, tile: true) = {
  let palette = palettes.at(theme)
  let factor = size / 2.6cm * 100%
  std.scale(x: factor, y: factor, reflow: true, cetz.canvas({
    import cetz.draw: *
    rect((0, 0), (2.6, 2.6), radius: .38,
      fill: if tile { palette.background } else { none }, stroke: none)
    canopy(palette.canopy)
    branches(palette.ink)
  }))
}

#set page(width: auto, height: auto, margin: 0pt, fill: none)
#logo(theme: sys.inputs.at("theme", default: "light"))
