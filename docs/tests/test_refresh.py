import pytest

from docs import refresh

VERSION = "0.5.4"
REVISION = "a" * 40
EXPECTED = (VERSION, REVISION)
STALE = ("0.5.3", "b" * 40)


def test_refresh_skips_an_already_published_selection(monkeypatch):
    monkeypatch.delenv("READTHEDOCS_TOKEN", raising=False)
    monkeypatch.setattr(refresh, "selection", lambda: EXPECTED)
    refresh.refresh(*EXPECTED)


@pytest.mark.parametrize("success", [True, False])
def test_refresh_checks_the_triggered_build_and_live_selection(monkeypatch, success):
    selections = iter([STALE, STALE, EXPECTED] if success else [STALE])
    statuses = iter(
        [
            {"build": {"id": 42}},
            {"state": {"code": "building"}},
            {"state": {"code": "finished"}, "success": success},
        ]
    )
    calls = []
    queries = []

    def api(path, token, *, method="GET"):
        assert token == "test-token"
        calls.append((method, path))
        return next(statuses)

    def selection(query=""):
        queries.append(query)
        return next(selections)

    monkeypatch.setenv("READTHEDOCS_TOKEN", "test-token")
    monkeypatch.setattr(refresh, "api", api)
    monkeypatch.setattr(refresh, "selection", selection)
    monkeypatch.setattr(refresh.time, "sleep", lambda duration: None)
    if success:
        refresh.refresh(*EXPECTED)
        assert queries == ["", "?build=42&attempt=0", "?build=42&attempt=1"]
    else:
        with pytest.raises(RuntimeError, match=r"failed or was cancelled.+/builds/42/"):
            refresh.refresh(*EXPECTED)
        assert queries == [""]
    assert calls == [("POST", "versions/latest/builds/"), ("GET", "builds/42/"), ("GET", "builds/42/")]


def test_refresh_reports_stale_evidence_after_a_successful_build(monkeypatch):
    monkeypatch.setenv("READTHEDOCS_TOKEN", "test-token")
    monkeypatch.setattr(refresh, "selection", lambda query="": STALE)
    monkeypatch.setattr(refresh.time, "sleep", lambda duration: None)

    def api(path, token, *, method="GET"):
        if method == "POST":
            return {"build": {"id": 42}}
        return {"state": {"code": "finished"}, "success": True}

    monkeypatch.setattr(refresh, "api", api)
    with pytest.raises(ValueError, match=r"serves \('0.5.3'.+Expected \('0.5.4'"):
        refresh.refresh(*EXPECTED)
