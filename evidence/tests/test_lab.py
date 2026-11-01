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
