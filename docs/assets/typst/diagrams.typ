// Explanatory diagrams sharing the model-tree palette: pipeline lanes,
// decision-time timelines, and token grids. Like model-tree.typ, these
// describe ideas without importing relflow or reading data.
#import "model-tree.typ": palette

#let serif = "Libertinus Serif"
#let mono = "DejaVu Sans Mono"

// One step in a flow lane. `kind` selects the palette entry.
#let step(label, kind: "field", note: none, width: 108pt) = {
  assert(kind in ("root", "branch", "field", "target"),
    message: "Step " + repr(label) + ": kind must be root, branch, field, or target.")
  (label: label, kind: kind, note: note, width: width)
}

#let arrow(color) = polygon(fill: color,
  (0pt, 3.5pt), (10pt, 3.5pt), (10pt, 0pt), (17pt, 5.5pt), (10pt, 11pt), (10pt, 7.5pt), (0pt, 7.5pt))

#let flow-card(item) = {
  let colors = palette.at(item.kind)
  block(width: item.width, inset: 8pt, radius: 5pt, fill: colors.fill, stroke: 0.7pt + colors.border,
    stack(dir: ttb, spacing: 3pt,
      text(size: 9pt, weight: "semibold", fill: colors.ink, item.label),
      if item.note != none { text(size: 7.5pt, fill: colors.ink, item.note) },
    ),
  )
}

// Each lane is a dictionary: (title: "Feature pipeline", steps: (step(...), ...)).
// Lanes render as rows; steps in a lane are joined by arrows.
#let flow(..lanes, title-width: 64pt) = context {
  set text(font: serif, hyphenate: false)
  set par(leading: 0.6em)
  let lanes = lanes.pos()
  let longest = calc.max(..lanes.map(lane => lane.steps.len()))
  let columns = (title-width,) + range(2 * longest - 1).map(i => if calc.rem(i, 2) == 0 { auto } else { 20pt })
  let cells = ()
  for lane in lanes {
    cells.push(text(size: 7pt, weight: "semibold", tracking: 0.4pt, fill: palette.field.ink, upper(lane.title)))
    let index = 0
    for item in lane.steps {
      if index > 0 { cells.push(align(center + horizon, arrow(palette.connector))) }
      cells.push(flow-card(item))
      index += 1
    }
    for _ in range(2 * (longest - lane.steps.len())) { cells.push([]) }
  }
  block(fill: palette.canvas, radius: 8pt, inset: 16pt,
    grid(columns: columns, align: horizon, column-gutter: 4pt, row-gutter: 16pt, ..cells),
  )
}

// Events: (at: number, label: content, level: 0 or 1, kind: "input" | "excluded" | "label").
// Inputs are known at the cutoff; excluded events happened later; a label event
// defines an outcome. `cutoff` is (at:, label:); `window` is (from:, to:, label:).
#let timeline(events, cutoff: none, window: none, start: 0, end: 1, width: 430pt) = context {
  set text(font: serif, hyphenate: false, size: 8pt)
  set par(leading: 0.55em)
  let x(value) = (value - start) / (end - start) * width
  let axis = 56pt
  let height = 108pt
  block(fill: palette.canvas, radius: 8pt, inset: 16pt,
    box(width: width, height: height, {
      if window != none {
        place(top + left, dx: x(window.from), dy: axis - 9pt,
          rect(width: x(window.to) - x(window.from), height: 18pt, radius: 3pt, fill: palette.target.fill,
            stroke: (paint: palette.target.border, thickness: 0.7pt, dash: "dashed")))
        place(top + left, dx: x(window.from) + 6pt, dy: axis - 27pt,
          text(fill: palette.target.ink, weight: "semibold", window.label))
      }
      place(top + left, line(start: (0pt, axis), end: (width, axis), stroke: 1pt + palette.connector))
      for event in events {
        let kind = event.at("kind", default: "input")
        assert(kind in ("input", "excluded", "label"),
          message: "Event " + repr(event.label) + ": kind must be input, excluded, or label.")
        let colors = if kind == "input" { palette.branch } else if kind == "label" { palette.target } else { palette.field }
        let level = event.at("level", default: 0)
        place(top + left, dx: x(event.at) - 4.5pt, dy: axis - 4.5pt,
          circle(radius: 4.5pt, fill: if kind == "excluded" { palette.canvas } else { colors.fill },
            stroke: 1pt + colors.border))
        place(top + left, dx: x(event.at) - 45pt, dy: axis + 12pt + level * 13pt,
          box(width: 90pt, align(center, text(fill: colors.ink, event.label))))
      }
      if cutoff != none {
        place(top + left, dx: x(cutoff.at), dy: 2pt,
          line(start: (0pt, 0pt), end: (0pt, axis + 7pt),
            stroke: (paint: palette.root.ink, thickness: 1pt, dash: "dashed")))
        place(top + left, dx: x(cutoff.at) + 5pt, dy: 0pt,
          text(fill: palette.root.ink, weight: "semibold", cutoff.label))
      }
    }),
  )
}

// Value states shared by `policies` and `tokens`: how one position enters the
// model. "input" is visible; "masked" is present as a learned mask; "skipped"
// is absent from attention; "target" is hidden and reconstructed or predicted;
// "none" leaves the slot empty, for a field that is not in the schema yet.
#let states = ("input", "masked", "skipped", "target", "none")

#let repeat(state, count) = range(count).map(_ => state)

#let state-style(state) = {
  assert(state in states, message: "State " + repr(state) + " must be one of " + repr(states) + ".")
  if state == "input" { (fill: palette.field.fill, stroke: 0.7pt + palette.field.border, ink: palette.field.ink) }
  else if state == "masked" {
    (fill: palette.branch.fill, stroke: (paint: palette.branch.border, thickness: 0.7pt, dash: "dashed"), ink: palette.branch.ink)
  } else if state == "skipped" {
    (fill: none, stroke: (paint: palette.connector, thickness: 0.7pt, dash: "dotted"), ink: palette.connector)
  } else if state == "target" { (fill: palette.target.fill, stroke: 0.7pt + palette.target.border, ink: palette.target.ink) }
  else { (fill: none, stroke: none, ink: palette.field.ink) }
}

#let state-glyph(state, size: 11pt) = {
  let style = state-style(state)
  rect(width: size, height: size, radius: 2pt, fill: style.fill, stroke: style.stroke)
}

#let state-legend() = {
  let item(state, label) = box(stack(dir: ltr, spacing: 4pt,
    box(baseline: 15%, state-glyph(state, size: 9pt)), text(size: 7.5pt, fill: palette.field.ink, label)))
  stack(dir: ltr, spacing: 14pt,
    item("input", [Visible input]),
    item("masked", [Learned mask: present, content hidden]),
    item("skipped", [Skipped: absent from attention]),
    item("target", [Hidden and reconstructed or predicted]),
  )
}

// One row per policy or field, one column per stage. Each row is
// (label: content, cells: ((state, ...), ...)) with one state tuple per stage;
// each tuple draws the retained positions of one repeated field.
#let policies(stages, rows, label-width: 200pt, legend: true) = context {
  set text(font: serif, hyphenate: false, size: 8pt)
  set par(leading: 0.55em)
  show raw: set text(size: 7.5pt)
  let sequence(cell) = stack(dir: ltr, spacing: 3pt, ..cell.map(state => state-glyph(state)))
  let cells = ([],) + stages.map(stage =>
    text(size: 7pt, weight: "semibold", tracking: 0.4pt, fill: palette.field.ink, upper(stage)))
  for row in rows {
    assert(row.cells.len() == stages.len(),
      message: "Row " + repr(row.label) + ": give one state tuple per stage.")
    cells.push(text(fill: palette.field.ink, row.label))
    for cell in row.cells { cells.push(sequence(cell)) }
  }
  let footer = if legend { (state-legend(),) } else { () }
  block(fill: palette.canvas, radius: 8pt, inset: 16pt,
    stack(dir: ttb, spacing: 14pt,
      grid(columns: (label-width,) + stages.map(_ => auto),
        align: (left + horizon,) + stages.map(_ => center + horizon),
        row-gutter: 9pt, column-gutter: 18pt, ..cells),
      ..footer,
    ),
  )
}

// A grid of field tokens for repeated records. `fields` names the columns;
// each record is (label: content, values: (content, ...)). With the default
// frames, the first record is framed as one coordinate and the `highlight`
// column is framed across records; pass `highlight: none` or
// `frame-first: false` to omit a frame, and `summary: none` to omit the
// reduction arrow. `states` is an optional matrix of value states per cell.
#let tokens(fields, records, summary: [Learned summary], within: none, across: none, reduce: none,
    highlight: 1, frame-first: true, states: none) = context {
  set text(font: serif, hyphenate: false, size: 8pt)
  set par(leading: 0.55em)
  let cell-w = 86pt
  let cell-h = 34pt
  let gx = 10pt
  let gy = 10pt
  let pad-left = 62pt
  let pad-top = if within == none and not frame-first { 6pt } else { 28pt }
  let count = fields.len()
  let rows = records.len()
  let grid-w = count * cell-w + (count - 1) * gx
  let grid-h = rows * cell-h + (rows - 1) * gy
  let summary-w = 100pt
  let width = pad-left + grid-w + if summary == none { 6pt } else { 48pt + summary-w }
  let height = pad-top + grid-h + if across == none { 6pt } else { 44pt }
  block(fill: palette.canvas, radius: 8pt, inset: 16pt,
    box(width: width, height: height, {
      if frame-first {
        place(top + left, dx: pad-left - 6pt, dy: pad-top - 6pt,
          rect(width: grid-w + 12pt, height: cell-h + 12pt, radius: 6pt, fill: none,
            stroke: (paint: palette.root.border, thickness: 1pt, dash: "dashed")))
      }
      if within != none {
        place(top + left, dx: pad-left, dy: pad-top - 23pt,
          text(fill: palette.root.ink, weight: "semibold", within))
      }
      if highlight != none {
        let column-x = pad-left + highlight * (cell-w + gx)
        place(top + left, dx: column-x - 6pt, dy: pad-top - 6pt,
          rect(width: cell-w + 12pt, height: grid-h + 12pt, radius: 6pt, fill: none,
            stroke: (paint: palette.branch.border, thickness: 1pt, dash: "dashed")))
        if across != none {
          place(top + left, dx: column-x - 6pt, dy: pad-top + grid-h + 13pt,
            box(width: grid-w, text(fill: palette.branch.ink, weight: "semibold", across)))
        }
      }
      for (row, record) in records.enumerate() {
        let y = pad-top + row * (cell-h + gy)
        place(top + left, dx: 0pt, dy: y,
          box(width: pad-left - 10pt, height: cell-h,
            align(right + horizon, text(fill: palette.field.ink, record.label))))
        for (column, value) in record.values.enumerate() {
          let style = state-style(if states == none { "input" } else { states.at(row).at(column) })
          place(top + left, dx: pad-left + column * (cell-w + gx), dy: y,
            block(width: cell-w, height: cell-h, inset: 6pt, radius: 4pt, fill: style.fill, stroke: style.stroke,
              stack(dir: ttb, spacing: 2pt,
                text(font: mono, size: 7.5pt, weight: "semibold", fill: style.ink, fields.at(column)),
                text(size: 7.5pt, fill: style.ink, value))))
        }
      }
      if summary != none {
        let arrow-x = pad-left + grid-w + 14pt
        let arrow-y = pad-top + grid-h / 2
        place(top + left, dx: arrow-x, dy: arrow-y - 5.5pt, arrow(palette.connector))
        place(top + left, dx: arrow-x + 24pt, dy: arrow-y - 16pt,
          block(width: summary-w, inset: 7pt, radius: 5pt, fill: palette.root.fill, stroke: 0.7pt + palette.root.border,
            text(size: 8.5pt, weight: "semibold", fill: palette.root.ink, summary)))
        if reduce != none {
          place(top + left, dx: arrow-x + 24pt, dy: arrow-y + 20pt,
            box(width: summary-w + 8pt, text(fill: palette.root.ink, reduce)))
        }
      }
    }),
  )
}
