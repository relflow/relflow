"""CPU runtime resolution preserves lock versions without installing unrelated backends."""

import pytest
from resolve import check, pins


def test_cpu_metadata_selects_a_subset_without_stripping_dependency_names(tmp_path):
    constraints, requirements = tmp_path / "constraints.txt", tmp_path / "requirements.txt"
    constraints.write_text("torch==2.12.0\nnvidia-example==1.0 ; sys_platform == 'linux'\nnumpy==2.4.6\n")
    requirements.write_text("numpy==2.4.6\ntorch==2.12.0+cpu\n")
    assert check(constraints, requirements) == {"numpy": "2.4.6", "torch": "2.12.0+cpu"}


@pytest.mark.parametrize(
    "requirement",
    ["torch==2.12.0", "torch==2.13.0+cpu", "torch==2.12.0+cpu\nnumpy==3.0", "torch==2.12.0+cpu\nnew-dependency==1.0"],
)
def test_cpu_resolution_rejects_backend_and_lock_drift(tmp_path, requirement):
    constraints, requirements = tmp_path / "constraints.txt", tmp_path / "requirements.txt"
    constraints.write_text("torch==2.12.0\nnumpy==2.4.6\n")
    requirements.write_text(requirement + "\n")
    with pytest.raises(ValueError):
        check(constraints, requirements)


@pytest.mark.parametrize(
    "requirement",
    ["", "numpy>=2", "numpy @ https://example.com/numpy.whl", "--extra-index-url https://example.com/simple"],
)
def test_runtime_pins_reject_unbounded_or_executable_input(tmp_path, requirement):
    path = tmp_path / "requirements.txt"
    path.write_text(requirement + "\n")
    with pytest.raises(ValueError):
        pins(path)
