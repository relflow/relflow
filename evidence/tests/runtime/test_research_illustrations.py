"""Every optional study must render without importing a model runtime."""

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


def test_all_research_illustrations_render_without_model_dependencies(tmp_path):
    pytest.importorskip("matplotlib")
    script = textwrap.dedent("""
        import builtins
        import sys
        from pathlib import Path

        original = builtins.__import__
        def guarded(name, *args, **kwargs):
            if name.split('.')[0] in {'relflow', 'torch', 'lightning'}:
                raise AssertionError(f'generator illustration imported {name}')
            return original(name, *args, **kwargs)
        builtins.__import__ = guarded
        from evidence.diagnostics.illustrations import illustrate
        from evidence.experiments.source import registry
        import matplotlib.pyplot as plt

        catalog = registry(Path('evidence'), measurements=False, lane='research')
        for identifier in catalog:
            figure, note = illustrate(identifier)
            assert 'illustrat' in note.lower(), identifier
            figure.savefig(Path(sys.argv[1]) / f'{identifier}.svg')
            plt.close(figure)
        print(len(catalog))
    """)
    root = Path(__file__).resolve().parents[3]
    subprocess.run([sys.executable, "-c", script, str(tmp_path)], cwd=root, check=True)
    assert list(tmp_path.glob("*.svg"))
