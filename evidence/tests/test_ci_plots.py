"""Exact-head CI selection and publication must never silently use stale plots."""

import json

import preview
import pytest

from evidence import assets


@pytest.mark.parametrize("conclusion", [None, "failure", "success"])
def test_reader_selects_latest_attempt_without_falling_back(tmp_path, monkeypatch, conclusion):
    head = "a" * 40
    requests = []
    runs = [
        dict(
            id=8,
            run_attempt=2,
            head_sha=head,
            event="pull_request",
            status="completed",
            conclusion=conclusion,
            head_repository={"full_name": "relflow/relflow"},
        ),
        dict(
            id=7, run_attempt=1, head_sha=head, event="pull_request", head_repository={"full_name": "relflow/relflow"}
        ),
    ]

    def download(url, **options):
        requests.append(url)
        if "/runs?" in url:
            return json.dumps({"workflow_runs": runs}).encode()
        assert url.endswith("/evidence-ci-8-attempt-2")
        return None

    monkeypatch.setattr(assets, "download", download)
    stale = tmp_path / "docs/assets/evidence-plots/review.json"
    stale.parent.mkdir(parents=True)
    stale.write_text("{}")
    result = json.loads(assets.ci(tmp_path, head).read_text())
    assert result["version"] == "ci-pending" and result["results"] == {}
    assert "attempt 2" in result["report"]["status"]
    assert not stale.exists() and len(requests) == 2


def test_publication_requires_matching_plot_asset_before_downloading(tmp_path):
    selected = {"report": {"plots": {"sha256": "a" * 64, "size": 10, "url": "wrong"}}}
    with pytest.raises(ValueError, match="retained GitHub release asset"):
        assets.plots(tmp_path, selected, {"assets": []})


def test_refresh_skips_superseded_pr_without_reading_a_token(monkeypatch):
    monkeypatch.setattr(
        preview,
        "api",
        lambda *args: [
            {"state": "open", "number": 86, "head": {"sha": "b" * 40, "repo": {"full_name": "relflow/relflow"}}}
        ],
    )
    monkeypatch.delenv("READTHEDOCS_TOKEN", raising=False)
    assert preview.refresh({"event": "pull_request", "head_sha": "a" * 40}) is None


def test_refresh_requires_token_only_after_confirming_the_current_pr(monkeypatch):
    monkeypatch.setattr(
        preview,
        "api",
        lambda *args: [
            {"state": "open", "number": 86, "head": {"sha": "a" * 40, "repo": {"full_name": "relflow/relflow"}}}
        ],
    )
    monkeypatch.delenv("READTHEDOCS_TOKEN", raising=False)
    with pytest.raises(ValueError, match="assets are published.*READTHEDOCS_TOKEN"):
        preview.refresh({"event": "pull_request", "head_sha": "a" * 40})


def test_collect_rejects_fork_before_reading_artifacts(tmp_path, monkeypatch):
    monkeypatch.setattr(preview, "api", lambda *args: {"id": 1})
    monkeypatch.setattr(preview, "artifact", lambda *args: pytest.fail("Fork artifacts must not be read"))
    with pytest.raises(ValueError, match="original internal CI"):
        preview.collect(
            tmp_path,
            tmp_path / "output",
            {"id": 8, "workflow_id": 1, "event": "pull_request", "head_repository": {"full_name": "external/fork"}},
            helper="a" * 40,
            executor="b" * 40,
        )
