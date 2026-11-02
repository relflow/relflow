# Model trees

`model-tree.typ` exports `node` and `tree`. These describe a model's structure
without importing relflow, instantiating models, or accessing data. The Quarto
project imports both functions through its `typst-render` preamble.

Use a Typst block with a unique figure label, caption, and text alternative:

````markdown
---
engine: markdown
---

```{typst}
//| label: fig-orders-overview
//| fig-cap: "Each order combines its line items before predicting a return."
//| fig-alt: "An order contains line items with Mean reduction and branch attention disabled. Product is an input; half the quantity values are masked for reconstruction. Returned is always hidden from input."
#tree(node("order", kind: "root", children: (
  node("line_items", kind: "branch", repeated: true, width: 150pt, body: [
    - *Reduction:* Mean
    - Branch attention off
  ], children: (
    node("product", type: "Category"),
    node("quantity", type: "Number", width: 150pt, body: [
      - *50% masked* for reconstruction
      - Other values remain visible
    ]),
  )),
  node("returned", kind: "target", type: "Boolean", body: [
    *Always hidden* from input.
  ]),
)))
```
````

The page-level `engine: markdown` keeps Quarto from selecting a Python execution
engine for the diagram fence. Python examples use ordinary `python` fences.

```typst
node(label, kind: "field", type: none, detail: none, body: none, width: 120pt,
     repeated: false, children: ())
```

`node` returns one diagram description. `tree(node(...))` draws it, including a
single node when there are no children. Give the root a descriptive label such
as `order`, `customer`, or `record` to identify the observation. This label is
illustrative only: it is not defined in `Model(...)` and is not an address prefix;
the model root remains anonymous at `rf.Address()`.
Use the exact schema names for branches and leaves, preserving case and
underscores as in the sample inputs.
Names render in monospace. For nodes with an explicit query, keep the schema
name as the label and show the source path in `detail` or `body`.

Every leaf supplies its datatype with `type`, such as `type: "Number"`.
It appears above the name; roots and child branches both show `BRANCH`.
`kind` chooses the color, while `detail` and `body` explain masking,
prediction roles, source paths, or exported embeddings. Input and prediction
roles do not appear as headers.

Pass any [Typst content](https://typst.app/docs/reference/foundations/content/)
in `body: [...]`: paragraphs, bullet lists, emphasis, inline code, equations,
tables, or custom blocks. It is rendered directly below the title and detail.
Labels and details also accept formatted content. The renderer supplies readable
text and list defaults in the node's light or dark palette; local Typst styling
can override them.

For example, a body can combine formatted text and a small layout:

```typst
#tree(node([*quantity*], type: "Number", width: 165pt, body: [
  *Reconstruction mask* with $p = 0.5$.

  #grid(columns: (1fr, 1fr), gutter: 4pt,
    [*Selected*], [*Other values*],
    [Hidden], [Visible],
  )
]))
```

`width` changes the diagram card, not the model. Give longer notes or tables
more room; height grows with the formatted content. All of the card is measured
before placing children and connectors, so a tall note reserves its own space.

| Kind | Meaning |
| --- | --- |
| `root` | Shared context for one observation. |
| `branch` | Local context containing aligned child fields or nested branches. `repeated: true` marks a collection. |
| `field` | An encoded input. A reconstructing input remains a field; describe its masking in the node body. |
| `target` | A supervised leaf whose value is excluded from embedding (`mask=True`). Its decoder reads model context. |

Connections show schema ownership. Inputs are combined toward their parent
contexts and the root; prediction leaves use ancestor context. They do not
represent a sequence of database joins or a claim that every decoder reads only
its immediate parent. For a branch diagram, note its containing model context in
the caption.

Use node bodies to explain settings that matter to the diagram: a masking rate,
which values are hidden, Mean versus Attention reduction, retained tokens, or
the date coordinates exposed to the model. Put each note on the node it
describes. Label alternatives when one tree compares separately trained
configurations, and include the important distinctions in the figure's text
alternative. Keep unrelated settings out of the diagram and avoid footnotes.
Each child should correspond to the nearby schema; label omissions in prose.

The renderer measures complete cards before laying out subtrees, so labels and
formatted bodies wrap and nested branches do not overlap. It uses native Typst
and its bundled DejaVu Sans Mono and Libertinus Serif fonts, without downloaded
diagram packages or system fonts.
`make render` builds the SVG figures and static HTML; `make check-docs` checks the
same pipeline in a temporary directory.

The project configures Typst Render with light/dark backgrounds. The extension
compiles each diagram declaration to two SVGs and wraps them in Quarto's
`light-content` and `dark-content` classes. This follows the site's automatic
device color-scheme selection, including changes while the page is open; no
extra diagram markup or JavaScript is needed.

The dark variant uses a charcoal background, `#181c1b`. Typst Render passes
that value through `sys.inputs["typst-render-background"]` to select the dark
palette in `model-tree.typ`. The preamble keeps the page transparent so the
tree canvas has visible 8pt rounded corners in both themes.
The light canvas is `#f9faf7`. Green, blue, and amber
accents distinguish root contexts, branch contexts, and prediction targets.
If the dark canvas changes, update both the project setting and the palette selector.
Standalone Typst rendering defaults to the light palette; use
`--input typst-render-background=#181c1b` for the dark variant.

# Explanatory diagrams

`diagrams.typ` exports `flow`, `step`, `timeline`, and `tokens`. They share the
model-tree palette and canvas, so they sit beside model trees without a change
of style. The preamble imports all four. Use them where a tree cannot show the
idea: a sequence of stages, a position in time, or the tokens inside a branch.

```typst
flow(..lanes, title-width: 64pt)
step(label, kind: "field", note: none, width: 108pt)
```

Each lane is `(title: "...", steps: (step(...), ...))` and renders as one row
of cards joined by arrows. `kind` selects the same colors as tree nodes: use
`root` and `branch` for what relflow owns, `target` for work the application
must maintain, and `field` for neutral stages. Put a short `note` on a step
rather than a footnote.

```typst
timeline(events, cutoff: none, window: none, start: 0, end: 1, width: 430pt)
```

Events are `(at: number, label: content, level: 0 or 1, kind: "input")`.
`input` events are known at the cutoff, `excluded` events happened later and
stay out of the inputs, and a `label` event defines an outcome. `cutoff` is
`(at:, label:)` and `window` is `(from:, to:, label:)`. Positions are plain
numbers between `start` and `end`, such as days; stagger nearby labels with
`level: 1`.

```typst
tokens(fields, records, summary: [Learned summary], within: none, across: none,
       reduce: none, highlight: 1)
```

`fields` names the columns and each record is `(label: content, values: (...))`.
The first record is framed as one coordinate and the `highlight` column is
framed across records; `within`, `across`, and `reduce` label those frames and
the summary card. Pass `frame-first: false`, `highlight: none`, or
`summary: none` to omit a frame or the reduction arrow. An optional `states`
matrix, one value state per cell, shades cells that a masking policy hides.
Keep values short, and use the same example records as the nearby YAML.

```typst
policies(stages, rows, label-width: 200pt, legend: true)
repeat(state, count)
```

`policies` draws one row per policy or field and one column per stage. Each
row is `(label: content, cells: ((state, ...), ...))` with one state tuple per
stage; each tuple draws the retained positions of one repeated field. Value
states are `input` (visible), `masked` (present as a learned mask), `skipped`
(absent from attention), `target` (hidden and reconstructed or predicted), and
`none` (not in the schema). `repeat("input", 5)` builds a uniform tuple. The
legend names the four drawn states; turn it off for a small follow-up grid.
