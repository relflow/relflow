"""Exercise installed native and relflow wheels without importing the model stack."""

import importlib.metadata
import importlib.util
import json
import socket
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pytest
from relflow_native import native

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def cache_module():
    path = importlib.metadata.distribution("relflow").locate_file("relflow/tensorfields/shared/cache/__init__.py")
    specification = importlib.util.spec_from_file_location("artifact_cache", path)
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def test_installed_package_has_matching_version_types_and_no_model_dependencies():
    assert native.__version__ == importlib.metadata.version("relflow-native")
    package = Path(native.__file__).parent
    assert (package / "py.typed").is_file()
    assert (package / "native.pyi").is_file()
    assert not {"relflow", "torch", "transformers"} & sys.modules.keys()


def test_automatic_selection_starts_rust_owner(cache_module, tmp_path):
    cache = cache_module.Cache(tmp_path)
    try:
        stats = cache.stats()
        assert stats["backend"] == stats["transport"] == "rust"
        assert stats["protocol"] == native.cache.PROTOCOL == cache_module.PROTOCOL
        assert cache.owner.process.args[1:3] == ["-m", "relflow_native.cache"]
    finally:
        cache.owner.close()


def test_incompatible_owner_startup_releases_the_child(cache_module, tmp_path, monkeypatch):
    spawn = subprocess.Popen
    children = []

    def launch(command, **kwargs):
        child = spawn(
            [
                sys.executable,
                "-c",
                "import json, sys; sys.stdin.buffer.read(32); "
                "print(json.dumps(dict(protocol=999)), flush=True); sys.stdin.buffer.read()",
            ],
            **kwargs,
        )
        children.append(child)
        return child

    monkeypatch.setattr(subprocess, "Popen", launch)
    cache = cache_module.Cache(tmp_path)
    try:
        with pytest.raises(RuntimeError, match="install matching relflow"):
            cache.share()
        assert cache.owner.process is None
        assert len(children) == 1 and children[0].poll() == 0
    finally:
        cache.owner.close()


@pytest.fixture(params=["rust", "python"])
def owner(request, cache_module, tmp_path):
    backend = Path(cache_module.__file__).with_name("backend.py")
    command = ["-m", "relflow_native.cache"] if request.param == "rust" else [str(backend)]
    auth = b"t" * 32
    process = subprocess.Popen(
        [sys.executable, *command, str(tmp_path / "cache.sqlite3"), "4096", "0"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    try:
        process.stdin.write(auth)
        process.stdin.flush()
        ready = json.loads(process.stdout.readline())
        assert ready["backend"] == request.param
        assert ready["protocol"] == cache_module.PROTOCOL
        yield cache_module.Endpoint(ready["port"], auth, ready["identity"], 0)
    finally:
        process.stdin.close()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        error = process.stderr.read().decode()
        process.stdout.close()
        process.stderr.close()
        assert process.returncode == 0, error


@pytest.mark.parametrize("transport", [True, False])
def test_both_transports_and_owners_share_artifacts(owner, transport, cache_module):
    cache = cache_module.Cache(endpoint=owner, native=transport)
    try:
        refs = cache.lookup_arrow(pa.array(["café", "second", "café"]), "source/v1")
        ids = cache.bind(refs, cache_module.Payloads.from_values([b"a", b"b", b"a"]), "prepare/v1")
        np.testing.assert_array_equal(cache.fetch(ids, "encoder/v1").status, [0, 0, 0])
        values = [b"result", b"", b"result"]
        cache.publish(ids, "encoder/v1", cache_module.Payloads.from_values(values))
        assert [bytes(value) for value in cache.fetch(ids, "encoder/v1").hits] == values
        assert cache.stats()["transport"] == ("rust" if transport else "python")
    finally:
        cache.connection.close()


def test_owner_rejects_incompatible_wire_protocol(owner, cache_module):
    with socket.create_connection(("127.0.0.1", owner.port), timeout=10) as stream:
        stream.sendall(owner.auth + struct.pack("<I", cache_module.PROTOCOL + 1))
        assert struct.unpack("<I", cache_module.receive(stream, 4))[0] == cache_module.PROTOCOL
        assert stream.recv(1) == b""


@pytest.mark.parametrize("transport", [True, False])
def test_client_rejects_incompatible_wire_protocol(transport, cache_module):
    import threading

    def reply():
        stream, _ = listener.accept()
        with stream:
            cache_module.receive(stream, 36)
            stream.sendall(struct.pack("<I", cache_module.PROTOCOL + 1))

    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        listener.settimeout(10)
        thread = threading.Thread(target=reply)
        thread.start()
        client = cache_module.Connection(
            cache_module.Endpoint(listener.getsockname()[1], b"x" * 32, "test", 0), native=transport
        )
        try:
            with pytest.raises(RuntimeError, match="install matching relflow"):
                client.request(5)
            assert client.stream is None
        finally:
            client.close()
            thread.join(timeout=15)
        assert not thread.is_alive()
