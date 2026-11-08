"""Validation-only prototype: use soft group mixtures during training."""

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import cast

import torch

from relflow.structs.enums import TensorKey, Tokens
from relflow.structs.packages import Parcel
from relflow.tensorfields.extensions.group import Embedder


def forward(embedder, inputs):
    """Train with a mixture, retaining ordinary hard routing at evaluation."""
    n, *dims = inputs.state.shape
    state = inputs.state.reshape(-1)
    content = cast(torch.Tensor, inputs.content).reshape(-1)
    valued = state.eq(Tokens.valued.value)
    safe = content.masked_fill(content.eq(-1), embedder.capacity).masked_fill(~valued, 0)
    logits = embedder.embeddings[TensorKey.group.name](safe)
    if embedder.training:
        assignment = torch.softmax(logits, dim=-1)
    else:
        assignment = torch.zeros_like(logits)
        assignment.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    assignment = assignment * valued.unsqueeze(-1)
    group = embedder.embeddings[TensorKey.content.name](assignment)
    values = (embedder.embeddings[TensorKey.state.name](state) + group).reshape(n, *dims, -1)
    return Parcel(
        payload=values,
        present=torch.ones(n, dtype=torch.bool, device=values.device),
        origin=embedder.origin,
        destination=embedder.destination,
        batch_size=n,
    )


def main() -> None:
    import group_lab

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--steps", type=int, choices=(2, 1800), default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output.with_suffix(".variant.json")
    if args.output.exists() or receipt.exists():
        raise FileExistsError(args.output)
    Embedder.forward = forward
    sys.argv = [
        "group_lab.py",
        "--seed",
        str(args.seed),
        "--groups",
        "4",
        "--learning-rate",
        "0.003",
        "--steps",
        str(args.steps),
        "--accelerator",
        "cpu",
        "--output",
        str(args.output),
    ]
    group_lab.main()
    with receipt.open("x") as stream:
        json.dump(
            {
                "purpose": "validation-only soft training prototype; not canonical Evidence",
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
