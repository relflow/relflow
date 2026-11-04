# relflow agent skills

The canonical [schema inference skill](relflow-schema/SKILL.md) supports investigation
by an agent with file and Python execution tools. Its local scripts profile data and
validate proposals; an agent supplies semantic interpretation. No Jev account, MCP
server, or model-provider call is required by these scripts.

## Use locally

Open `relflow-schema/SKILL.md` explicitly in your agent, or install the local skill
with the cross-agent installer:

```bash
npx skills add ./skills --skill relflow-schema
```

The installer is an optional third-party tool. Choose the desired agents and scope.
For manual repository discovery, place the complete skill directory in
`.agents/skills/relflow-schema/` for Codex, `.claude/skills/relflow-schema/` for Claude
Code, or `.cursor/skills/relflow-schema/` for Cursor. Do not maintain edited copies;
use installer-managed copies or a symlink to this canonical directory.

From this checkout, use the package environment for scripts:

```bash
uv run python skills/relflow-schema/scripts/profile_data.py observations.parquet --rows 2048 > profile.json
uv run python skills/relflow-schema/scripts/validate.py model.yaml --data observations.parquet > validation.json
uv run python skills/relflow-schema/scripts/validate.py model.yaml --data observations.parquet --encode
```

Profiling needs Python 3.12 and PyArrow 20 or newer, without importing relflow.
Validation was checked with relflow 0.4.4 and uses its live template/preset resolver,
ragged projection, and extension registry. It needs that package and its normal
dependencies, but default validation does not construct models, fetch encoders,
or train. It checks Arrow compatibility in both train and validate strata, including
labeled supervised targets. `--encode` additionally constructs a disposable CPU
model and tensorizes those observations; it may need optional extension resources.
Use `--extension your.module` to import an explicitly selected extension registration
module. Scripts never run user-authored preprocessing code: supply its resulting
Arrow observations. Missing visible fields fail the live SDK checks; empty contexts
are reported as unverified. Semantic null/state constraints may require encoding.

`profile_data.py` reads a prefix, not a random or representative population sample.
Defaults bound rows (2048), retained Arrow/JSON bytes (32 MiB), values per field
(100,000), schema traversal depth (8), and field count (2048). JSON files must fit
the byte budget; use JSONL for larger sources. Parquet and IPC use bounded batches.
CSV values stay strings to preserve leading zeros and literal null-like labels;
conversions must be proposed explicitly. Examples are truncated to 160 characters;
use `--examples 0` to omit them. Lists report nulls, length percentiles, and sampled
child structure. Parser/decompression peak allocations can exceed retained-data
limits; these scripts are local tools, not an untrusted-upload sandbox.

Validation requires finite branch lengths unless `--allow-unbounded` is explicit.
Its `--max-slots` guard defaults to 1,000,000 declared structural/leaf slots per
sample batch. This is a conservative cap envelope, not a measured token count or
VRAM estimate. Lower `--rows` when needed. Arrow normalization is not full semantic
tensorization, output coverage, prediction quality, or a resource benchmark.

## Build publishing bundles

```bash
uv run python scripts/skills.py check
uv run python scripts/skills.py build --output dist/skills
uv run pytest tests/test_skills.py
```

Use a fresh output directory; existing artifacts are never overwritten. The builder
produces a standalone skill ZIP, a plugin ZIP, an unpacked plugin directory, a local
Claude marketplace catalog, and `build.json` with per-file and artifact SHA256 hashes.
ZIP member ordering, timestamps, and permissions are deterministic. Both bundles
contain the exact canonical skill bytes and license, with no local data or caches.

The plugin combines a portable root `plugin.json` for Agent Plugins hosts (including
Codex and Cursor) with `.claude-plugin/plugin.json` for Claude Code. It contains no
MCP server. The canonical manifest template is `skills/plugin.json`; its version
must match the skill metadata. To test with a locally installed Claude Code CLI:

```bash
claude plugin validate ./dist/skills/relflow-schema --strict
claude --plugin-dir ./dist/skills/relflow-schema
```

These are manual host checks, not claims that every host has been exercised. Run
the cases in [behavioral evaluation](evals/README.md) in each intended agent before
advertising behavioral compatibility. A successful packaging check alone does not
establish automatic selection, resource availability, or model reasoning quality.

## Publish when requested

Publish the canonical source with a versioned Git revision first. After publication,
users can install with `npx skills add relflow/relflow --skill relflow-schema`.
Ship the generated plugin directory from a dedicated publication branch/repository
or reviewed release artifact. Native marketplace workflows consume that staged
directory, not this repository root. Keep the runtime wheel and skill revision
identities separate. Marketplace listings have their own submission/review processes;
the local builder performs none of them and does not install global agent settings.

Sources: [Agent Skills](https://agentskills.io/specification),
[skills CLI](https://github.com/vercel-labs/skills),
[OpenAI plugin packaging](https://developers.openai.com/plugins/build/plugins),
[Claude plugin manifests](https://code.claude.com/docs/en/plugins-reference), and
[Cursor plugins](https://cursor.com/docs/plugins).
