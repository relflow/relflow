#import "../logos/logo.typ": logo, palettes

#set page(width: 1120pt, height: 300pt, margin: 0pt, fill: none)

#let artwork(theme: "light") = {
  let palette = palettes.at(theme)
  set text(font: "Libertinus Serif", fill: palette.ink)
  set par(leading: 0pt)
  grid(
    columns: (280pt, auto), column-gutter: 32pt, align: left + horizon,
    logo(theme: theme, size: 280pt, tile: false),
    move(dy: 24pt,
      text(size: 132pt, weight: 600, tracking: 3pt, ligatures: false, top-edge: "bounds", bottom-edge: "bounds")[relflow],
    ),
  )
}

#let banner(theme: "light") = {
  let palette = palettes.at(theme)
  block(width: 1120pt, height: 300pt, radius: 28pt, fill: palette.background)[
    #align(center + horizon, artwork(theme: theme))
  ]
}

#banner(theme: sys.inputs.at("theme", default: "light"))
