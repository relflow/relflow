"""Follow up the gradient-surrogate search at a fixed 2,400-update budget."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

from temperature import forward

from relflow.tensorfields.extensions.group import Embedder


def main() -> None:
    import group_lab
    import temperature

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--temperature", type=float, choices=(0.5, 1.0), required=True)
    parser.add_argument("--smoke", action="store_true")
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
        "2" if args.smoke else "2400",
        "--accelerator",
        "cpu",
        "--output",
        str(args.output),
    ]
    group_lab.main()
    with receipt.open("x") as stream:
        json.dump(
            {
                "purpose": "validation-only gradient-surrogate follow-up; not canonical Evidence",
                "temperature": args.temperature,
                "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "surrogate_sha256": hashlib.sha256(Path(temperature.__file__).read_bytes()).hexdigest(),
                "result_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            stream,
            indent=2,
        )
        stream.write("\n")


if __name__ == "__main__":
    main()
