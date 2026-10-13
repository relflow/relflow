"""The series: one scene per episode.  ``SCENES`` (module, Scene class, chapter title) is the full cut;
``EPISODES`` publishes each scene on its own, followed by its end card (``tools/build.py relflow --episodes``)."""

SERIES = "relflow, Visualized"
TITLE = "relflow, Visualized: A Model Factory for Structured Data"
DOCS = "relflow.github.io/relflow"

# (slug, module, Scene class, title, docs pages the episode belongs on)
_EPISODES = [
    ("records-have-shape", "s01_hook", "Hook", "Records have shape", ["index.qmd"]),
    ("three-lines", "s02_three_lines", "ThreeLines", "Three lines to a trained model",
     ["getting-started.qmd", "guides/lightning.qmd"]),
    ("schema", "s03_schema", "SchemaIsArchitecture", "The schema is the architecture",
     ["core-concepts/model-tree.qmd"]),
    ("records-to-rectangles", "s04_ragged", "RaggedToDense", "From records to rectangles",
     ["core-concepts/binding-data.qmd", "data-types/branch.qmd"]),
    ("null-is-not-zero", "s05_state", "ValueState", "Null is not zero", ["core-concepts/data-types.qmd"]),
    ("numbers", "s06_number", "Numbers", "How a number becomes a vector", ["data-types/number.qmd"]),
    ("vocabulary", "s07_vocabulary", "Vocabulary", "A vocabulary that grows while it trains",
     ["core-concepts/vocabulary.qmd", "data-types/category.qmd"]),
    ("hash", "s08_hash", "Hashing", "Equality without identity", ["data-types/hash.qmd"]),
    ("quantile-and-enum", "s09_quantile_enum", "QuantileEnum", "Ranks and fixed labels",
     ["data-types/quantile.qmd", "data-types/enum.qmd"]),
    ("time-vectors-text", "s10_time_vector_text", "TimeVectorText", "Time, vectors and text",
     ["data-types/dateparts.qmd", "data-types/vector.qmd", "data-types/text.qmd"]),
    ("inside-a-branch", "s11_branch", "InsideABranch", "Inside a branch", ["data-types/branch.qmd"]),
    ("packing", "s12_packing", "Packing", "Don't pay for padding", ["guides/performance.qmd"]),
    ("softmax-cant-count", "s13_reduction", "Reduction", "Why softmax can't count", ["data-types/branch.qmd"]),
    ("climbing-the-tree", "s14_routing", "Routing", "Climbing the tree", ["core-concepts/model-tree.qmd"]),
    ("decoders", "s15_decoders", "Decoders", "Asking the tree a question", ["guides/prediction-output.qmd"]),
    ("masks", "s16_masks", "Masks", "One mask, many objectives", ["core-concepts/dynamic-masking.qmd"]),
    ("pretraining", "s17_pretraining", "Pretraining", "Learning without labels", ["guides/pretraining.qmd"]),
    ("processors", "s18_processors", "Processors", "Before and after the model",
     ["guides/preprocessors.qmd", "guides/postprocessors.qmd"]),
    ("watching-it-learn", "s19_real_run", "RealRun", "Watching it learn",
     ["guides/evaluation.qmd", "core-concepts/embeddings.qmd"]),
    ("benchmarks", "s20_benchmarks", "Benchmarks", "What it costs: time and memory", ["guides/performance.qmd"]),
    ("proofs", "s21_evidence", "Evidence", "What the proofs show, and what they don't", ["proofs.qmd"]),
]

EPISODES = [
    {"number": i, "slug": slug, "module": module, "cls": cls, "title": title, "docs": docs,
     "end": ("endcards", f"End{i:02d}")}
    for i, (slug, module, cls, title, docs) in enumerate(_EPISODES, 1)
]
SCENES = [(e["module"], e["cls"], e["title"]) for e in EPISODES]


def episode(cls: str) -> dict:
    return next(e for e in EPISODES if e["cls"] == cls)


# Soundtrack: a quiet generated music bed under the narration (tools/music.py), ducked while the narrator speaks
# and lifted for the title, the episode cards and the ending.
MUSIC = {"seed": 7, "under_db": -22.0, "gap_db": -16.0, "feature_db": -9.0}
