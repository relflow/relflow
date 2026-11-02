"""Remote report transfer must not follow paths or links into the host filesystem."""

import io
import tarfile

import pytest
from ssh_run import connection, extract_reports


def test_remote_reports_extract_as_data(tmp_path):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo("summary.json")
        member.size = 2
        archive.addfile(member, io.BytesIO(b"{}"))
    extract_reports(data.getvalue(), tmp_path / "reports")
    assert (tmp_path / "reports/summary.json").read_text() == "{}"


@pytest.mark.parametrize(
    "name,kind", [("../escape", tarfile.REGTYPE), ("/escape", tarfile.REGTYPE), ("link", tarfile.SYMTYPE)]
)
def test_remote_report_transfer_rejects_unsafe_files(tmp_path, name, kind):
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode="w:gz") as archive:
        member = tarfile.TarInfo(name)
        member.type = kind
        member.linkname = "/etc/passwd"
        archive.addfile(member)
    with pytest.raises(ValueError):
        extract_reports(data.getvalue(), tmp_path / "reports")


@pytest.mark.parametrize("target", ["-oProxyCommand=evil", "user@lab;command", "user@lab\ncommand"])
def test_connection_rejects_shell_or_option_injection(target):
    with pytest.raises(ValueError):
        connection(target, "ssh")


def test_scheduler_starts_long_experiments_first_and_ignores_invalid_hints(tmp_path, monkeypatch):
    import lab_run

    monkeypatch.setattr(
        lab_run,
        "registry",
        lambda root: {
            "P001": {"run": {"duration_seconds": 10}},
            "P002": {"run": {"duration_seconds": 400}},
            "P003": {"run": {"duration_seconds": float("nan")}},
            "P004": {"run": None},
        },
    )
    tasks = [
        {"id": identifier, "role": "candidate", "accelerator": "cpu", "seed": 7, "attempt": 1}
        for identifier in ("P001", "P002", "P003", "P004")
    ]
    assert [task["id"] for task in lab_run.schedule(tasks, tmp_path)] == ["P002", "P001", "P003", "P004"]
    assert [task["id"] for task in tasks] == ["P001", "P002", "P003", "P004"]


def test_wheel_installation_is_in_an_offline_cached_layer(tmp_path, monkeypatch):
    import lab_run

    wheel = tmp_path / "relflow-0.4.3-py3-none-any.whl"
    wheel.write_bytes(b"verified-wheel")
    recipes = []

    def build(command, **kwargs):
        context = lab_run.Path(command[-1])
        recipes.append((context / "Dockerfile").read_text())
        assert (context / wheel.name).read_bytes() == wheel.read_bytes()
        assert command[command.index("--network") + 1] == "none"

    monkeypatch.setattr(lab_run.subprocess, "run", build)
    monkeypatch.setattr(lab_run.subprocess, "check_output", lambda *args, **kwargs: "sha256:installed-wheel\n")
    assert lab_run.wheel_image({"image": "verified-base", "image_id": "sha256:base"}, wheel) == "sha256:installed-wheel"
    assert "FROM verified-base\n" in recipes[0]
    assert "pip install --no-deps --no-index" in recipes[0]
