# GitHub Copilot Instructions

Read `AGENTS.md` first and follow its implementation style and ownership
boundaries. Use `CONTRIBUTING.md` for the complete extension, Arrow data,
documentation, and testing contracts. Those files are authoritative if this
summary differs.

Use `import relflow as rf` in generated examples. Build public schemas with
`rf.Model(...)`, `rf.Branch(...)`, and top-level tensorfield constructors. Do
not invent a public `Struct(...)` API.

Keep code direct: use short semantic names, inline one-use forwarding helpers,
never prefix functions or classes with `_`, preserve Arrow as the canonical CPU
representation, and keep datatype behavior inside its extension. Update tests
with public behavior and coordinate user documentation changes in the separate
[relflow/docs repository](https://github.com/relflow/docs). Follow its authoring
guide for static examples and Typst diagrams; site builds must not run model
code. Executable modeling proofs, their canonical results, and all branding
assets live in that documentation repository; package tests remain here.
