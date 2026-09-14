// Diagram descriptions are independent of Python models and training data.
#let node(label, kind: "field", detail: none, repeated: false, children: ()) = {
  assert(kind in ("root", "branch", "field", "target"),
    message: "Node '" + label + "': kind must be root, branch, field, or target.")
  assert(children.len() == 0 or kind in ("root", "branch"),
    message: "Node '" + label + "': only roots and branches can contain children.")
  assert(not repeated or kind == "branch",
    message: "Node '" + label + "': only a branch can be repeated.")
  (label: label, kind: kind, detail: detail, repeated: repeated, children: children)
}

// Typst Render supplies this background for the Quarto dark-mode variant.
#let dark = sys.inputs.at("typst-render-background", default: "") == "#181c1b"
#let palette = if dark { (
  canvas: rgb("#181c1b"),
  connector: rgb("#75827b"),
  root: (ink: rgb("#c9dec5"), fill: rgb("#28392d"), border: rgb("#627a60")),
  branch: (ink: rgb("#b6d1e2"), fill: rgb("#23303a"), border: rgb("#566f80")),
  field: (ink: rgb("#e0e7e1"), fill: rgb("#202725"), border: rgb("#52645b")),
  target: (ink: rgb("#e5c99d"), fill: rgb("#392f24"), border: rgb("#947950")),
) } else { (
  canvas: rgb("#f9faf7"),
  connector: rgb("#b1beb5"),
  root: (ink: rgb("#234631"), fill: rgb("#e6eddf"), border: rgb("#94aa83")),
  branch: (ink: rgb("#31576b"), fill: rgb("#edf3f6"), border: rgb("#a2bdcb")),
  field: (ink: rgb("#35463b"), fill: rgb("#ffffff"), border: rgb("#ccd6cf")),
  target: (ink: rgb("#805224"), fill: rgb("#fbf0de"), border: rgb("#d8b47e")),
) }

#let card(item) = {
  let colors = palette.at(item.kind)
  let role = (root: "ROOT CONTEXT", branch: "BRANCH CONTEXT", field: "INPUT", target: "PREDICTION").at(item.kind)
  if item.repeated { role = "REPEATED BRANCH" }
  block(
    width: 120pt,
    inset: 9pt,
    radius: 5pt,
    fill: colors.fill,
    stroke: 0.7pt + colors.border,
    stack(dir: ttb, spacing: 4pt,
      text(size: 6.5pt, weight: "semibold", fill: colors.ink, tracking: 0.4pt, role),
      text(size: 11pt, weight: "semibold", fill: colors.ink, item.label),
      if item.detail != none { text(size: 8pt, fill: colors.ink, item.detail) },
    ),
  )
}

// Measure each subtree once; parents sit midway between their outer children.
#let arrange(item) = {
  let body = card(item)
  let size = measure(body)
  let children = item.children.map(arrange)
  let height = calc.max(size.height,
    children.map(child => child.height).sum(default: 0pt) + calc.max(0, children.len() - 1) * 10pt)
  let cursor = 0pt
  let placed = ()
  for child in children {
    placed.push((tree: child, top: cursor))
    cursor += child.height + 10pt
  }
  let center = if placed.len() == 0 { height / 2 } else {
    (placed.first().tree.center + placed.last().top + placed.last().tree.center) / 2
  }
  let above = calc.max(0pt, size.height / 2 - center)
  (
    body: body, size: size, children: placed,
    center: center + above, offset: above,
    height: calc.max(height + above, center + above + size.height / 2),
    width: size.width + if children.len() == 0 { 0pt } else {
      30pt + calc.max(..children.map(child => child.width))
    },
  )
}

#let draw(item, x: 0pt, y: 0pt) = {
  let right = x + item.size.width
  let middle = y + item.center
  for child in item.children {
    let child-y = y + item.offset + child.top
    let destination = child-y + child.tree.center
    place(top + left, line(start: (right, middle), end: (right + 15pt, middle), stroke: 0.8pt + palette.connector))
    place(top + left, line(start: (right + 15pt, middle), end: (right + 15pt, destination), stroke: 0.8pt + palette.connector))
    place(top + left, line(start: (right + 15pt, destination), end: (right + 30pt, destination), stroke: 0.8pt + palette.connector))
    draw(child.tree, x: right + 30pt, y: child-y)
  }
  place(top + left, dx: x, dy: middle - item.size.height / 2, item.body)
}

// Links show schema ownership. Prediction cards never represent encoded inputs.
#let tree(root) = context {
  set text(font: "Libertinus Serif", hyphenate: false)
  set par(leading: 0.6em)
  let arranged = arrange(root)
  block(fill: palette.canvas, radius: 8pt, inset: 16pt,
    block(width: arranged.width, height: arranged.height, draw(arranged)),
  )
}
