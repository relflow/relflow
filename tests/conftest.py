import os
import sys

import pytest


@pytest.fixture(scope="session", autouse=True)
def gloo_interface():
    """All distributed tests run their ranks on the same machine."""
    with pytest.MonkeyPatch.context() as environment:
        if sys.platform == "darwin" and "GLOO_SOCKET_IFNAME" not in os.environ:
            environment.setenv("GLOO_SOCKET_IFNAME", "lo0")
        yield
