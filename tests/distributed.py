"""Launch CPU DDP ranks without returning Lightning metrics through a pipe."""

import os
import socket
from collections.abc import Callable
from typing import Any

import torch.distributed as distributed
import torch.multiprocessing as multiprocessing

__all__ = ["launch"]


def worker(rank: int, function: Callable[..., None], args: tuple[Any, ...], port: int) -> None:
    os.environ.update(
        MASTER_ADDR="127.0.0.1",
        MASTER_PORT=str(port),
        NODE_RANK="0",
        LOCAL_RANK=str(rank),
        RANK=str(rank),
        WORLD_SIZE="2",
        LOCAL_WORLD_SIZE="2",
    )
    try:
        function(*args)
    finally:
        if distributed.is_initialized():
            distributed.destroy_process_group()


def launch(function: Callable[..., None], *args: Any) -> None:
    """Use external ranks so large metric dictionaries never block a result pipe."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    multiprocessing.spawn(worker, args=(function, args, port), nprocs=2, join=True)
