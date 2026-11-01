import os
import sys

import pytest


def pytest_addoption(parser):
    parser.addoption("--full", action="store_true", help="Include integration tests in the test selection.")


def pytest_configure(config):
    if not config.getoption("--full") and not config.option.markexpr:
        config.option.markexpr = "not integration"

    # Set pool sizes before test collection imports Torch, NumPy, and Polars.
    threads = os.environ.get("MKL_NUM_THREADS", os.environ.get("OMP_NUM_THREADS", "1"))
    for name, value in {
        "OMP_NUM_THREADS": threads,
        "MKL_NUM_THREADS": threads,
        "POLARS_MAX_THREADS": "2",
        "HF_HUB_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    }.items():
        os.environ.setdefault(name, value)


@pytest.fixture(scope="session", autouse=True)
def cpu_threads():
    """Tiny test models should not multiply CPU thread pools across pytest workers."""
    import torch

    previous = torch.get_num_threads()
    torch.set_num_threads(int(os.environ["MKL_NUM_THREADS"]))
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
