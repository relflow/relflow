"""Optional acceleration must distinguish absence, broken installs and incompatibility."""

from types import SimpleNamespace

import pytest

from relflow.tensorfields.shared import cache


@pytest.mark.parametrize("version", [None, 0, cache.PROTOCOL + 1])
def test_incompatible_native_cache_is_rejected(monkeypatch, version):
    monkeypatch.setattr(
        cache.importlib, "import_module", lambda name: SimpleNamespace(cache=SimpleNamespace(PROTOCOL=version))
    )
    with pytest.raises(RuntimeError, match="install matching relflow and relflow-native"):
        cache.native_cache()


@pytest.mark.parametrize("missing", ["relflow_native", "relflow_native.native", "dependency"])
def test_only_an_absent_optional_package_can_fall_back(monkeypatch, missing):
    def load(name):
        raise ModuleNotFoundError(name=missing)

    monkeypatch.setattr(cache.importlib, "import_module", load)
    if missing == "relflow_native":
        assert cache.native_cache() is None
    else:
        with pytest.raises(ModuleNotFoundError):
            cache.native_cache()
    with pytest.raises(ModuleNotFoundError):
        cache.native_cache(required=True)
