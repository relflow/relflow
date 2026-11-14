from pathlib import Path
from typing import ClassVar

import pyarrow as pa

__version__: str

class cache:
    PROTOCOL: ClassVar[int]

    class Client:
        def __init__(self, port: int, auth: bytes) -> None: ...
        def request(self, op: int, meta: bytes, payload: bytes) -> bytes: ...
        def identify_arrow(self, input: pa.Array, namespace: str) -> tuple[bytes, bytes]: ...
        def close(self) -> None: ...

    @staticmethod
    def run(path: str | Path, memory: int, capacity: int) -> None: ...
