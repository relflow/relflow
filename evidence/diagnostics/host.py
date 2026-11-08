"""Schedule immutable diagnostic captures in bounded offline CPU containers."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evidence.diagnostics.capture import digest, write
from evidence.experiments.source import registry

ROOT = Path(__file__).resolve().parents[2]


def policy() -> list[str]:
    """Keep preflight and proof workers on the same bounded offline profile."""
    return [
        "--network",
        "none",
        "--read-only",
        "--cpus",
        "2",
        "--memory",
        "4g",
        "--pids-limit",
        "512",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--tmpfs",
        "/tmp:rw,nosuid,nodev,size=2g,mode=1777",
        "--env",
        "USER=evidence",
        "--env",
        "LOGNAME=evidence",
        "--env",
        "XDG_CACHE_HOME=/tmp/cache",
        "--env",
        "TORCHINDUCTOR_CACHE_DIR=/tmp/inductor",
    ]


def prepare(output: Path) -> None:
    sys.path.insert(0, str(ROOT / "evidence"))
    from evidence.resolve import resolve
    from evidence.runtime import runtime_recipe

    output.mkdir(parents=True, exist_ok=False)
    subprocess.run(["uv", "build", "--wheel", "--out-dir", str(output / "wheel")], cwd=ROOT, check=True)
    resolve(ROOT, output / "runtime")
    wheel = next((output / "wheel").glob("*.whl"))
    context = output / "image"
    context.mkdir()
    shutil.copy2(output / "runtime/requirements.txt", context / "requirements.txt")
    shutil.copy2(wheel, context / wheel.name)
    recipe = (
        runtime_recipe(context / "requirements.txt")
        + f"\nCOPY {wheel.name} /tmp/{wheel.name}\nRUN python -m pip install --no-deps --no-index /tmp/{wheel.name}\n"
    )
    (context / "Dockerfile").write_text(recipe)
    image = "relflow-diagnostics:" + digest(wheel)[:16]
    subprocess.run(["docker", "build", "--tag", image, str(context)], check=True)
    subprocess.run(
        [
            "docker",
            "run",
            "--rm",
            *policy(),
            image,
            "python",
            "-c",
            "import getpass, lightning.pytorch, relflow; print('CPU worker preflight:', getpass.getuser())",
        ],
        check=True,
    )
    image_id = subprocess.check_output(["docker", "image", "inspect", image, "--format", "{{.Id}}"], text=True).strip()
    with (output / "runtime-image.tar.zst").open("xb") as stream:
        save = subprocess.Popen(["docker", "save", image], stdout=subprocess.PIPE)
        compress = subprocess.run(["zstd", "-T0", "-3"], stdin=save.stdout, stdout=stream, check=False)
        save.stdout.close()
        if save.wait() or compress.returncode:
            raise RuntimeError("Could not archive the diagnostic CPU image")
    entries = registry(ROOT / "evidence", measurements=False)
    timings = json.loads((ROOT / "evidence/timings.json").read_text())["tasks"]
    jobs, totals = [[] for _ in range(12)], [0.0] * 12
    for identifier, entry in sorted(entries.items(), key=lambda item: -timings.get(item[0], {}).get("candidate", 180)):
        shard = min(range(12), key=lambda index: totals[index])
        cost = timings.get(identifier, {}).get("candidate", 180)
        jobs[shard].append(identifier)
        totals[shard] += cost
    plan = {
        "schema": 1,
        "purpose": "Unpaired visualization capture; not the required learning gate",
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "wheel": wheel.name,
        "wheel_sha256": digest(wheel),
        "image": image,
        "image_id": image_id,
        "image_sha256": digest(output / "runtime-image.tar.zst"),
        "capture_sha256": digest(Path(__file__).with_name("capture.py")),
        "shards": [{"index": i, "ids": ids, "estimated_seconds": totals[i]} for i, ids in enumerate(jobs)],
        "experiments": {
            key: {"script": value["script"], "source_sha256": digest(ROOT / "evidence" / value["script"])}
            for key, value in entries.items()
        },
    }
    write(output / "plan.json", plan)
    shutil.rmtree(context)
    print(json.dumps({"experiments": len(entries), "shards": len(jobs), "plan_sha256": digest(output / "plan.json")}))


def task(identifier: str, plan: dict, inputs: Path, output: Path) -> dict:
    directory = output / identifier
    directory.mkdir(parents=True)
    view = directory / "view"
    source = Path("evidence") / plan["experiments"][identifier]["script"]
    for relative in [source, Path("evidence/experiments/reporting.py"), Path("evidence/experiments/source.py")]:
        target = view / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / relative, target)
    if digest(view / source) != plan["experiments"][identifier]["source_sha256"]:
        raise ValueError(f"{identifier}: source differs from the immutable diagnostic plan")
    capture = Path(__file__).with_name("capture.py")
    if digest(capture) != plan["capture_sha256"]:
        raise ValueError("Diagnostic capture differs from the prepared image request")
    wheel = inputs / "wheel" / plan["wheel"]
    if digest(wheel) != plan["wheel_sha256"]:
        raise ValueError("Diagnostic wheel differs from the prepared input")
    container = "diagnostics-" + os.environ.get("GITHUB_RUN_ID", str(os.getpid())) + "-" + identifier.lower()
    command = [
        "docker",
        "run",
        "--rm",
        "--name",
        container,
        *policy(),
        "--env",
        f"EVIDENCE_REVISION={plan['revision']}",
        "--env",
        f"EVIDENCE_RELFLOW_REVISION={plan['revision']}",
        "--env",
        "GITHUB_RUN_ID",
        "--env",
        "GITHUB_RUN_ATTEMPT",
        "--mount",
        f"type=bind,src={view.resolve()},dst=/suite,readonly",
        "--mount",
        f"type=bind,src={capture.resolve()},dst=/executor/capture.py,readonly",
        "--mount",
        f"type=bind,src={wheel.resolve()},dst=/inputs/{wheel.name},readonly",
        "--mount",
        f"type=bind,src={directory.resolve()},dst=/results",
        plan["image_id"],
        "python",
        "/executor/capture.py",
        "--source",
        "/suite/" + source.as_posix(),
        "--output",
        "/results/bundle",
        "--wheel",
        "/inputs/" + wheel.name,
    ]
    started = time.perf_counter()
    with (directory / "worker.log").open("x") as stream:
        try:
            process = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=1850, check=False)
            state = {"id": identifier, "exit_code": process.returncode}
        except subprocess.TimeoutExpired:
            subprocess.run(["docker", "rm", "--force", container], capture_output=True, check=False)
            state = {"id": identifier, "error": "Full-budget diagnostic capture exceeded 1850 seconds"}
    state["seconds"] = round(time.perf_counter() - started, 3)
    write(directory / "task.json", state)
    shutil.rmtree(view)
    print(json.dumps(state), flush=True)
    return state


def shard(inputs: Path, output: Path, index: int) -> None:
    plan = json.loads((inputs / "plan.json").read_text())
    if digest(inputs / "runtime-image.tar.zst") != plan["image_sha256"]:
        raise ValueError("Diagnostic runtime image archive changed")
    unpack = subprocess.Popen(
        ["zstd", "--decompress", "--stdout", str(inputs / "runtime-image.tar.zst")], stdout=subprocess.PIPE
    )
    loaded = subprocess.run(["docker", "load"], stdin=unpack.stdout, check=False)
    unpack.stdout.close()
    if unpack.wait() or loaded.returncode:
        raise RuntimeError("Could not load the verified diagnostic CPU image")
    actual = subprocess.check_output(
        ["docker", "image", "inspect", plan["image_id"], "--format", "{{.Id}}"], text=True
    ).strip()
    if actual != plan["image_id"]:
        raise ValueError("Loaded diagnostic runtime differs from its prepared identity")
    ids = plan["shards"][index]["ids"]
    output.mkdir(parents=True, exist_ok=False)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(lambda identifier: task(identifier, plan, inputs, output), ids))
    write(
        output / "shard.json",
        {
            "index": index,
            "ids": ids,
            "results": results,
            "plan_sha256": digest(inputs / "plan.json"),
            "host": {
                "uname": list(os.uname()),
                "cpu_count": os.cpu_count(),
                "cpuinfo": Path("/proc/cpuinfo").read_text(),
                "runner_name": os.environ.get("RUNNER_NAME"),
            },
        },
    )
    if any(result.get("exit_code") != 0 for result in results):
        raise SystemExit(1)


def collect(inputs: Path, shards: Path, output: Path) -> None:
    plan = json.loads((inputs / "plan.json").read_text())
    output.mkdir(parents=True, exist_ok=False)
    results, errors = {}, []
    for expected in plan["shards"]:
        folder = shards / f"shard-{expected['index']}"
        receipt = folder / "shard.json"
        if not receipt.is_file():
            errors.append(f"Missing diagnostic shard {expected['index']}")
            continue
        declared = json.loads(receipt.read_text())
        if declared["ids"] != expected["ids"] or declared["plan_sha256"] != digest(inputs / "plan.json"):
            raise ValueError("Diagnostic shard differs from the prepared request")
        for identifier in expected["ids"]:
            bundle = folder / identifier / "bundle"
            if not (bundle / "manifest.json").is_file():
                errors.append(f"{identifier}: missing diagnostic bundle")
                continue
            manifest = json.loads((bundle / "manifest.json").read_text())
            if (
                manifest["source_sha256"] != plan["experiments"][identifier]["source_sha256"]
                or manifest["wheel_sha256"] != plan["wheel_sha256"]
            ):
                raise ValueError(f"{identifier}: source or wheel differs from prepared inputs")
            for name, asset in manifest["artifacts"].items():
                if Path(name).is_absolute() or ".." in Path(name).parts or digest(bundle / name) != asset["sha256"]:
                    raise ValueError(f"{identifier}: invalid diagnostic artifact {name}")
            record = json.loads((bundle / "record.json").read_text())
            if record.get("mode") != "full" or record.get("steps_override") is not None:
                raise ValueError(f"{identifier}: diagnostic record is not full-budget")
            results[identifier] = record
            shutil.copytree(bundle, output / "bundles" / identifier)
            shutil.copy2(folder / identifier / "worker.log", output / "bundles" / identifier / "worker.log")
            if record["outcome"] == "error":
                errors.append(f"{identifier}: proof execution error")
    write(
        output / "selection.json",
        {"schema": 1, "version": "diagnostic-preview", "revision": plan["revision"], "report": {}, "results": results},
    )
    write(
        output / "summary.json",
        {
            "expected": len(plan["experiments"]),
            "recorded": len(results),
            "errors": errors,
            "outcomes": {key: value["outcome"] for key, value in results.items()},
            "plan_sha256": digest(inputs / "plan.json"),
        },
    )
    shutil.copy2(inputs / "plan.json", output / "plan.json")
    print(json.dumps({"expected": len(plan["experiments"]), "recorded": len(results), "errors": errors}))
    if errors:
        raise SystemExit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("prepare", "shard", "collect"))
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--shards", type=Path)
    parser.add_argument("--index", type=int)
    parser.add_argument("--output", type=Path, required=True)
    options = parser.parse_args()
    if options.operation == "prepare":
        prepare(options.output)
    elif options.operation == "shard":
        shard(options.inputs, options.output, options.index)
    else:
        collect(options.inputs, options.shards, options.output)


if __name__ == "__main__":
    main()
