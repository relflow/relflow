"""Hash every existing P078 prediction call while comparing input sources."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import ingress
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("synthetic", "arrow"), required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = args.output.with_suffix(".predictions.json")
    if args.output.exists() or receipt.exists():
        raise FileExistsError(args.output)
    original = ingress.proof.predict
    calls = []

    def predict(model, rows):
        result = original(model, rows)
        values = np.ascontiguousarray(result)
        calls.append(
            {
                "rows_sha256": hashlib.sha256(json.dumps(rows, sort_keys=True, allow_nan=False).encode()).hexdigest(),
                "prediction_sha256": hashlib.sha256(values.tobytes()).hexdigest(),
                "dtype": str(values.dtype),
                "shape": list(values.shape),
            }
        )
        return result

    ingress.proof.predict = predict
    sys.argv = ["ingress.py", "--seed", "7801", "--mode", args.mode, "--output", str(args.output)]
    if args.smoke:
        sys.argv.append("--smoke")
    ingress.main()
    with receipt.open("x") as stream:
        json.dump({"mode": args.mode, "calls": calls}, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
