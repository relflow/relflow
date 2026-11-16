"""Experiment revisions are data on the host, and execute only inside containers."""

from evidence_ci import plan


def test_planning_uses_trusted_parser_without_importing_suite_helpers(tmp_path):
    family = tmp_path / "experiments/family"
    family.mkdir(parents=True)
    (tmp_path / "experiments/source.py").write_text("raise RuntimeError('suite helper imported on host')\n")
    (family / "signal.py").write_text(
        "# %% [markdown]\n# ---\n# title: Signal\n# ---\n"
        "# %%\nfrom reporting import experiment, report\n"
        "@experiment('P901', seed=123)\n"
        "def run(seed, steps, accelerator):\n    raise RuntimeError('experiment executed on host')\n"
        "raise RuntimeError('suite module executed on host')\n"
        "if __name__ == '__main__':\n    report(run)\n"
    )
    manifest = plan(
        run_id="separate-suite", candidate={}, baseline={}, evidence_sha="suite", executor_sha="executor", root=tmp_path
    )
    assert len(manifest["tasks"]) == 2
    assert {task["id"] for task in manifest["tasks"]} == {"P901"}
    assert {task["seed"] for task in manifest["tasks"]} == {123}
    assert {task["executor_sha"] for task in manifest["tasks"]} == {"executor"}
