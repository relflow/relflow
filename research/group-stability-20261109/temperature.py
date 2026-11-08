"""Validation-only prototype for Group's straight-through gradient temperature."""

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


def forward(embedder, inputs, temperature):
    """Keep hard routing unchanged and scale only the soft backward surrogate."""
    n, *dims = inputs.state.shape
    state = inputs.state.reshape(-1)
    content = cast(torch.Tensor, inputs.content).reshape(-1)
    valued = state.eq(Tokens.valued.value)
    safe = content.masked_fill(content.eq(-1), embedder.capacity).masked_fill(~valued, 0)
    logits = embedder.embeddings[TensorKey.group.name](safe)
    hard = torch.zeros_like(logits)
    hard.scatter_(-1, logits.argmax(dim=-1, keepdim=True), 1.0)
    if embedder.training:
        soft = torch.softmax(logits / temperature, dim=-1)
        assignment = hard.detach() - soft.detach() + soft
    else:
        assignment = hard
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
    parser.add_argument("--temperature", type=float, choices=(0.5, 1.0, 2.0), required=True)
    parser.add_argument("--steps", type=int, choices=(2, 1800), default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output.with_suffix(".variant.json")
    if args.output.exists() or receipt.exists():
        raise FileExistsError(args.output)
    Embedder.forward = lambda self, inputs: forward(self, inputs, args.temperature)
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
                "purpose": "validation-only gradient-surrogate prototype; not canonical Evidence",
                "temperature": args.temperature,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
