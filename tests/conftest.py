import os
import sys

import pytest
import torch


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    """Tiny test models should not multiply CPU thread pools across pytest workers."""
    previous = torch.get_num_threads()
    with pytest.MonkeyPatch.context() as environment:
        if not any(name in os.environ for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS")):
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
                environment.setenv(name, "1")
            torch.set_num_threads(1)
        try:
            yield
        finally:
            torch.set_num_threads(previous)


@pytest.fixture(scope="session", autouse=True)
def gloo_interface():
    """All distributed tests run their ranks on the same machine."""
    with pytest.MonkeyPatch.context() as environment:
        if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
            environment.setenv("GLOO_SOCKET_IFNAME", "lo0")
        yield
