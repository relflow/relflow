#let theme = sys.inputs.at("theme", default: "light")
#assert(theme in ("light", "dark"))
#let background = if theme == "light" { "#f2f0e6" } else { "#192d24" }
#let ink = if theme == "light" { rgb("#10241a") } else { rgb("#dce5ce") }
#let mark = read("../../../starlight/src/assets/branding/mark." + theme + ".svg").replace("fill=\"" + background + "\"", "fill=\"none\"")

#set page(width: auto, height: auto, margin: 0pt, fill: none)
#set text(font: "Libertinus Serif", weight: 600, fill: ink,
  ligatures: false, tracking: .5pt)
#grid(columns: (auto, auto), column-gutter: 5pt, align: horizon,
  image(bytes(mark), format: "svg", height: 32pt),
  text(size: 24pt)[relflow],
)
