"""Experiment registration is inspectable without importing a selected suite."""

from pathlib import Path

import pytest

from evidence.experiments.source import Experiment, declaration, registry


def script(decorator: str = '@experiment("P901", seed=123)', function: str = "run(seed, steps, accelerator)") -> str:
    return (
        "# %% [markdown]\n# ---\n# title: Static declaration\n# ---\n# %%\n"
        f"{decorator}\ndef {function}:\n    raise RuntimeError('must not execute')\n"
    )


def test_static_declaration_does_not_evaluate_imports_or_function_bodies(tmp_path):
    marker = tmp_path / "executed"
    text = script() + f"\nopen({str(marker)!r}, 'w').write('executed')\n"
    assert declaration(text) == Experiment("P901", 123)
    assert not marker.exists()


@pytest.mark.parametrize(
    "decorator",
    [
        "@experiment",
        '@experiment("P901")',
        '@experiment("P901", 123)',
        '@experiment("P901", seed=123, budget=50)',
        '@experiment(identifier="P901", seed=123)',
        '@experiment(*("P901",), seed=123)',
        '@experiment("P901", **{"seed": 123})',
    ],
)
def test_static_declaration_rejects_unsupported_arguments(decorator):
    with pytest.raises(ValueError, match="Declare @experiment"):
        declaration(script(decorator))


@pytest.mark.parametrize(
    "decorator",
    ["@experiment(ID, seed=123)", '@experiment("P901", seed=next_seed())'],
)
def test_static_declaration_requires_literals(decorator):
    with pytest.raises(ValueError, match="must be literals"):
        declaration(script(decorator))


@pytest.mark.parametrize("seed", ["True", "-1", "4294967296", "1.5", '"123"'])
def test_static_declaration_rejects_invalid_seed(seed):
    with pytest.raises(ValueError, match="seed must be an integer"):
        declaration(script(f'@experiment("P901", seed={seed})'))


@pytest.mark.parametrize("identifier", ['"901"', '"P"', "901", "None"])
def test_static_declaration_requires_stable_id(identifier):
    with pytest.raises(ValueError, match="stable P identifier"):
        declaration(script(f"@experiment({identifier}, seed=123)"))


@pytest.mark.parametrize(
    "function",
    [
        "run(seed, steps)",
        "run(seed, steps, device)",
        "run(seed, steps, accelerator='cpu')",
        "run(seed, steps, accelerator, **kwargs)",
        "run(seed, /, steps, accelerator)",
        "run(seed, steps, *, accelerator)",
    ],
)
def test_static_declaration_rejects_unsupported_signature(function):
    with pytest.raises(ValueError, match="requires the signature"):
        declaration(script(function=function))


def test_static_declaration_rejects_multiple_entrypoints():
    with pytest.raises(ValueError, match="exactly one"):
        declaration(script() + '\n@experiment("P902", seed=456)\ndef other(seed, steps, accelerator):\n    pass\n')


@pytest.mark.parametrize(
    "source",
    [
        '@experiment("P901", seed=123)\nasync def run(seed, steps, accelerator):\n    pass\n',
        'def outer():\n    @experiment("P901", seed=123)\n    def run(seed, steps, accelerator):\n        pass\n',
        '@experiment("P901", seed=123)\nclass Example:\n    pass\n',
    ],
)
def test_static_declaration_rejects_nested_async_and_class_entrypoints(source):
    with pytest.raises(ValueError, match="synchronous top-level function"):
        declaration(source)


def test_catalog_discovers_new_files_on_each_call_and_supplies_presentation_id(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    first = family / "first.py"
    first.write_text(script())
    entries = registry(tmp_path, measurements=False)
    assert list(entries) == ["P901"]
    assert entries["P901"]["header"]["evidence-id"] == "P901"
    assert entries["P901"]["script"] == "experiments/family/first.py"
    (family / "second.py").write_text(script('@experiment("P902", seed=456)'))
    assert list(registry(tmp_path, measurements=False)) == ["P901", "P902"]


def test_catalog_rejects_duplicate_decorated_ids(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    for name in ("first", "second"):
        (family / f"{name}.py").write_text(script())
    with pytest.raises(ValueError, match="duplicate evidence ID"):
        registry(tmp_path, measurements=False)


def test_catalog_rejects_disagreeing_header_and_decorator(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    (family / "first.py").write_text(script().replace("# title:", "# evidence-id: P902\n# title:"))
    with pytest.raises(ValueError, match="differs from @experiment"):
        registry(tmp_path, measurements=False)


def test_malformed_declaration_cannot_fall_back_to_archived_header(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    (family / "first.py").write_text(script("@experiment").replace("# title:", "# evidence-id: P901\n# title:"))
    with pytest.raises(ValueError, match="Declare @experiment"):
        registry(tmp_path, measurements=False)


def test_archived_catalog_can_still_be_read_without_importing_it(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    (family / "archive.py").write_text(
        "# %% [markdown]\n# ---\n# evidence-id: P901\n# ---\n# %%\n"
        "raise RuntimeError('must not execute')\nreport(PROOF_ID, run, seed=123)\n"
    )
    assert declaration((family / "archive.py").read_text()) is None
    assert list(registry(tmp_path, measurements=False)) == ["P901"]


def test_complete_catalog_uses_one_declaration_per_experiment():
    root = Path(__file__).resolve().parents[2] / "evidence"
    entries = registry(root, measurements=False, lane="all")
    assert entries
    for identifier, entry in entries.items():
        text = (root / entry["script"]).read_text()
        assert declaration(text).id == identifier
        assert "PROOF_ID =" not in text
        assert "# evidence-id:" not in text


def test_lanes_are_explicit_static_and_default_to_complete_core(tmp_path):
    core = tmp_path / "experiments/family/core.py"
    research = tmp_path / "research/family/research.py"
    core.parent.mkdir(parents=True)
    research.parent.mkdir(parents=True)
    core.write_text(script())
    research.write_text(
        script('@experiment("P902", seed=456)').replace(
            "# title:", '# evidence-requires: ["offline encoder snapshot"]\n# title:'
        )
    )

    assert list(registry(tmp_path, measurements=False)) == ["P901"]
    selected = registry(tmp_path, measurements=False, lane="research")
    assert list(selected) == ["P902"]
    assert selected["P902"]["lane"] == "research"
    assert selected["P902"]["script"] == "research/family/research.py"
    assert set(registry(tmp_path, measurements=False, lane="all")) == {"P901", "P902"}
    assert not (tmp_path / "executed").exists()


def test_research_ids_cannot_collide_with_core_or_be_reclassified_by_metadata(tmp_path):
    core = tmp_path / "experiments/family/core.py"
    research = tmp_path / "research/family/research.py"
    core.parent.mkdir(parents=True)
    research.parent.mkdir(parents=True)
    core.write_text(script())
    research.write_text(script().replace("# title:", '# evidence-requires: ["extra"]\n# title:'))
    with pytest.raises(ValueError, match="duplicate evidence ID"):
        registry(tmp_path, measurements=False)
    research.unlink()
    core.write_text(script().replace("# title:", "# evidence-lane: research\n# title:"))
    with pytest.raises(ValueError, match="directory owns its lane"):
        registry(tmp_path, measurements=False)


@pytest.mark.parametrize("requirement", [None, '"extra"', "[]", '[""]', "[1]"])
def test_research_requires_explicit_nonempty_provisioning_descriptions(tmp_path, requirement):
    path = tmp_path / "research/family/research.py"
    path.parent.mkdir(parents=True)
    text = script()
    if requirement is not None:
        text = text.replace("# title:", f"# evidence-requires: {requirement}\n# title:")
    path.write_text(text)
    with pytest.raises(ValueError, match="evidence-requires"):
        registry(tmp_path, measurements=False, lane="research")


def test_invalid_lane_does_not_silently_select_an_empty_suite(tmp_path):
    with pytest.raises(ValueError, match="Evidence lane"):
        registry(tmp_path, measurements=False, lane="typo")
