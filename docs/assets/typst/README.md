# Model trees

`model-tree.typ` exports `node` and `tree`. These describe a model's structure
without importing RelFlow, instantiating models, or accessing data. The Quarto
project imports both functions through its `typst-render` preamble.

Use a Typst block with a unique figure label, caption, and text alternative:

````markdown
---
engine: markdown
---

```{typst}
//| label: fig-orders-overview
//| fig-cap: "Each order combines its line items before predicting a return."
//| fig-alt: "An order contains a repeated line-items branch with product and quantity inputs, and a returned prediction target."
#tree(node("Order", kind: "root", children: (
  node("Line items", kind: "branch", repeated: true, children: (
    node("Product", detail: "Category"),
    node("Quantity", detail: "Number"),
  )),
  node("Returned", kind: "target", detail: "Boolean"),
)))
```
````

The page-level `engine: markdown` keeps Quarto from selecting a Python execution
engine for the diagram fence. Python examples use ordinary `python` fences.

`node(label, kind: "field", detail: none, repeated: false, children: ())`
returns one diagram description. `tree(node(...))` draws it, including a single
node when there are no children. Names can be human-readable versions of schema
keys; use `detail` for a datatype or a short role such as "Exported embedding".

| Kind | Meaning |
| --- | --- |
| `root` | Shared context for one observation. |
| `branch` | Local context containing aligned child fields or nested branches. `repeated: true` marks a collection. |
| `field` | An encoded input. A reconstructing input remains a field; explain its masking in the caption or detail. |
| `target` | A supervised leaf whose value is excluded from embedding (`mask=True`). Its decoder reads model context. |

Connections show schema ownership. Inputs are combined toward their parent
contexts and the root; prediction leaves use ancestor context. They do not
represent a sequence of database joins or a claim that every decoder reads only
its immediate parent. For a branch diagram, note its containing model context in
the caption.

Keep diagrams focused on structure, repetition, and roles. Do not list widths,
head counts, layer counts, batch sizes, losses, or optimizer settings. Each child
should correspond to the nearby schema; when showing an excerpt, label the
omission in prose. Use captions and adjacent explanations for processing details.

The renderer measures labels before laying out subtrees, so longer labels wrap
and nested branches do not overlap. It uses native Typst and its bundled
Libertinus Serif font, without downloaded diagram packages or system fonts.
`make render` builds the SVG figures and static HTML; `make check-docs` checks the
same pipeline in a temporary directory.

The project configures Typst Render with light/dark backgrounds. The extension
compiles each diagram declaration to two SVGs and wraps them in Quarto's
`light-content` and `dark-content` classes. This follows the site's manual theme
toggle as well as its initial color-scheme selection; no extra diagram markup
or JavaScript is needed.

The dark variant uses a charcoal background, `#181c1b`. Typst Render passes
that value through `sys.inputs["typst-render-background"]` to select the dark
palette in `model-tree.typ`. The light canvas is `#f9faf7`. Green, blue, and amber
accents distinguish root contexts, branch contexts, and prediction targets.
If the dark canvas changes, update both the project setting and the palette selector.
Standalone Typst rendering defaults to the light palette; use
`--input typst-render-background=#181c1b` for the dark variant.
