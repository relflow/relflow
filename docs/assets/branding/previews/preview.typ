#import "../banners/banner.typ": artwork
#import "../logos/logo.typ": palettes

// At the renderer's 144 ppi, this canvas exports to exactly 1280 × 640 pixels.
#set page(width: 640pt, height: 320pt, margin: 0pt, fill: none)

#let preview(theme: "light") = {
  let palette = palettes.at(theme)
  block(width: 640pt, height: 320pt, fill: palette.background)[
    #align(center + horizon,
      scale(x: 60%, y: 60%, reflow: true, artwork(theme: theme)),
    )
  ]
}

#preview(theme: sys.inputs.at("theme", default: "light"))
