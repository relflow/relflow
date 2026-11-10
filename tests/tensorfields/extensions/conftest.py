"""Keep runtime cache artifacts local to each test."""

import pytest


@pytest.fixture(autouse=True)
def text_cache_directory(monkeypatch, tmp_path):
    from relflow.tensorfields.extensions.text_cache.store import Owner

    path = tmp_path / "text-cache"
    monkeypatch.setenv("RELFLOW_TEXT_CACHE_DIR", str(path))
    yield
    for owner in list(Owner.instances.values()):
        if owner.path == path:
            owner.close()
