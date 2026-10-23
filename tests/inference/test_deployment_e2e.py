from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest
import torch

import relflow as rf

SERVER_SCRIPT = """
import sys

from relflow.inference.deployment import Deployment

checkpoint = sys.argv[1]
port = int(sys.argv[2])

Deployment(
    checkpoint=checkpoint,
    accelerator="cpu",
    max_batch_size=1,
    batch_timeout=0.0,
    host="127.0.0.1",
    port=port,
    log_level="error",
).serve()
"""


def free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        sock.listen(1)
        return int(sock.getsockname()[1])


def tail_text(path: Path, lines: int = 40) -> str:
    if not path.exists():
        return ""

    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def wait_for_server(base_url: str, process: subprocess.Popen[str], log_path: Path, timeout: float = 60.0) -> None:
    deadline = time.monotonic() + timeout
    last_error = "server did not respond"

    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise AssertionError(
                f"deployment exited before readiness probe succeeded\nlog tail:\n{tail_text(log_path)}"
            )

        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=1.0) as response:
                if response.status == 200:
                    return
        except urllib.error.URLError as exc:
            last_error = str(exc.reason)
        except OSError as exc:
            last_error = str(exc)

        time.sleep(0.1)

    raise AssertionError(f"timed out waiting for deployment readiness: {last_error}\nlog tail:\n{tail_text(log_path)}")


def stop_process(process: subprocess.Popen[str], timeout: float = 10.0) -> None:
    try:
        for action in ("interrupt", "terminate", "kill"):
            if process.poll() is not None:
                return
            try:
                if os.name == "nt":
                    if action == "interrupt":
                        process.send_signal(signal.CTRL_BREAK_EVENT)
                    elif action == "terminate":
                        process.terminate()
                    else:
                        process.kill()
                else:
                    signals = {"interrupt": signal.SIGINT, "terminate": signal.SIGTERM, "kill": signal.SIGKILL}
                    os.killpg(process.pid, signals[action])
            except ProcessLookupError:
                process.wait(timeout=timeout)
                return
            except OSError:
                # Headless Windows runners may not provide a console for CTRL_BREAK.
                if os.name == "nt" and action == "interrupt":
                    continue
                raise
            try:
                process.wait(timeout=timeout)
                return
            except subprocess.TimeoutExpired:
                if action == "kill":
                    raise
    finally:
        log_handle = getattr(process, "log_handle", None)
        if log_handle is not None:
            log_handle.close()


def post_json(url: str, payload: Any, timeout: float = 30.0) -> tuple[int, Any, dict[str, str]]:
    request = urllib.request.Request(
        url=url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8")), dict(response.headers)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise AssertionError(f"deployment returned HTTP {exc.code}: {body}") from exc


def launch_deployment(checkpoint: Path, port: int, log_path: Path) -> subprocess.Popen[str]:
    log_handle = log_path.open("w", encoding="utf-8")
    try:
        process = subprocess.Popen(
            [sys.executable, "-u", "-c", SERVER_SCRIPT, str(checkpoint), str(port)],
            cwd=Path(__file__).resolve().parents[2],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=os.name != "nt",
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
        )
    except BaseException:
        log_handle.close()
        raise
    process.log_handle = log_handle
    return process


@pytest.fixture(scope="module")
def deployment(tmp_path_factory):
    """Share an inference-only HTTP server; predictions do not learn vocabulary."""
    directory = tmp_path_factory.mktemp("deployment")
    model = rf.Model.xs(d_model=8, label=rf.Category(embed=True))
    model.encode(pa.table({"label": ["alpha", "beta"]}), strata="train")
    with torch.no_grad():
        # Admitted labels embed as unknown until trained; stand in for the training that separates them.
        model.nodes["/label"].embedder.embeddings["content"].weight[:2].normal_()
    checkpoint_path = model.save(directory / "model.ckpt")
    port = free_port()
    base_url = f"http://127.0.0.1:{port}"
    log_path = directory / "deployment.log"
    process = launch_deployment(checkpoint=checkpoint_path, port=port, log_path=log_path)
    try:
        wait_for_server(base_url=base_url, process=process, log_path=log_path)
        yield base_url, model.schema.d_model
    finally:
        stop_process(process)


def test_deployment_serves_embeddings_from_temporary_checkpoint(deployment) -> None:
    base_url, d_model = deployment
    status, payload, headers = post_json(f"{base_url}/predict", {"label": "alpha"})

    assert status == 200
    assert set(payload) == {"representations"}
    representation = payload["representations"][0]
    assert representation["path"] == ["label"]
    assert representation["parents"] == []
    assert representation["output_index"] == 0
    embedding = representation["embedding"]
    assert len(embedding) == d_model
    assert all(isinstance(value, float) for value in embedding)
    assert headers["x-relflow-prediction-format"] == "2"
    assert headers["x-relflow-model-version"]


def test_deployment_accepts_multiple_inputs_in_one_request(deployment) -> None:
    base_url, d_model = deployment
    status, payload, _ = post_json(f"{base_url}/predict", [{"label": "alpha"}, {"label": "beta"}])

    assert status == 200
    assert isinstance(payload, list)
    assert len(payload) == 2
    for item in payload:
        assert set(item) == {"representations"}
        assert item["representations"][0]["path"] == ["label"]
        embedding = item["representations"][0]["embedding"]
        assert len(embedding) == d_model


def test_deployment_accepts_unseen_category_values_at_runtime(deployment) -> None:
    base_url, _ = deployment
    _, alpha_payload, _ = post_json(f"{base_url}/predict", {"label": "alpha"})
    status, gamma_payload, _ = post_json(f"{base_url}/predict", {"label": "gamma"})

    assert status == 200
    assert alpha_payload["representations"][0]["path"] == ["label"]
    assert gamma_payload["representations"][0]["path"] == ["label"]
    assert alpha_payload["representations"][0]["embedding"] != gamma_payload["representations"][0]["embedding"]
