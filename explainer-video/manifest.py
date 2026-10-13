"""Scene order for the final cut: (module, Scene class, chapter title)."""

TITLE = "relflow, Visualized: How a Nested Schema Becomes a Transformer"

SCENES = [
    ("s01_hook", "Hook", "Records have shape"),
    ("s02_schema", "SchemaIsArchitecture", "The schema is the architecture"),
    ("s03_ragged", "RaggedToDense", "From records to rectangles"),
    ("s04_state", "ValueState", "Null is not zero"),
    ("s05_number", "Numbers", "How a number becomes a vector"),
    ("s06_vocabulary", "Vocabulary", "A vocabulary that grows while it trains"),
    ("s07_hash", "Hashing", "Equality without identity"),
    ("s08_branch", "InsideABranch", "Inside a branch"),
    ("s09_packing", "Packing", "Don't pay for padding"),
    ("s10_reduction", "Reduction", "Why softmax can't count"),
    ("s11_routing", "Routing", "Climbing the tree"),
    ("s12_decoders", "Decoders", "Asking the tree a question"),
    ("s13_real_run", "RealRun", "Watching it learn"),
    ("s14_evidence", "Evidence", "What the proofs show, and what they don't"),
]

# Soundtrack: a quiet generated music bed under the narration (tools/music.py), ducked while the narrator speaks
# and lifted for the title, the chapter cards and the ending.
MUSIC = {"seed": 7, "under_db": -22.0, "gap_db": -16.0, "feature_db": -9.0}
