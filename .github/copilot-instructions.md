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
with public behavior and coordinate user documentation changes in `docs/`
within the same branch. Follow `docs/AGENTS.md` for static examples and Typst
diagrams; site builds use the isolated `docs/pyproject.toml` environment and
must not import or run model code. Canonical learning experiments, their runtime
tests, and recorded results live in `evidence/`; branding assets live under
`docs/assets/branding/`. Package tests remain in `tests/`, and static
documentation tests live in `docs/tests/`.

Read the Docs owns versioned hosting and pull request previews through the root
`.readthedocs.yaml`; GitHub documentation CI validates and retains an artifact.
Preserve the Quarto layout and isolated build. Use `docs.publish` and the host's
native version flyout; follow `docs/README.md` for external account, Gold,
project ad-free, and domain setup. Verify a hosted preview before changing DNS.
