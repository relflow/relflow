#import "../logos/logo.typ": logo, palettes

#set page(width: 1120pt, height: 300pt, margin: 0pt, fill: none)

#let banner(theme: "light") = {
  let palette = palettes.at(theme)
  set text(font: "Libertinus Serif", fill: palette.ink)
  set par(leading: 0pt)
  block(width: 1120pt, height: 300pt, radius: 28pt, fill: palette.background)[
    #align(center + horizon, grid(
      columns: (280pt, auto), column-gutter: 32pt, align: left + horizon,
      logo(theme: theme, size: 280pt, tile: false),
      move(dy: 24pt,
        text(size: 132pt, weight: 600, tracking: 3pt, ligatures: false, top-edge: "bounds", bottom-edge: "bounds")[relflow],
      ),
    ))
  ]
}

#banner(theme: sys.inputs.at("theme", default: "light"))
