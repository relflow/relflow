#!/usr/bin/env bash
set -uo pipefail

# Standalone research capture: the source is not yet committed and this is not CI.
run_dir=$(pwd)
if [[ -e "$run_dir/output/p101-full" || -e "$run_dir/output/p101-full.zip" ]]; then
    printf 'Refusing to overwrite the P101 capture\n' >&2
    exit 1
fi
docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
    --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
    -e PYTHONHASHSEED=10101 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
    -e EVIDENCE_REVISION=uncommitted-working-tree \
    -e EVIDENCE_RELFLOW_REVISION=e5d6f42010b1802edc6687ea55c1032c3badd3af \
    -e EVIDENCE_WHEEL_SHA256=cabe120b55c0e552e64469ecf337171650c3fd087375e7d56dd629ddccde5e8b \
    -e PYTHONPATH=/work/input/package:/work/input/evidence:/work/input/evidence/experiments \
    -v "$run_dir:/work" --entrypoint python relflow-evidence-wheel:81a7beaf1c547e4f53c6 \
    /work/input/evidence/diagnostics/capture.py --source /work/input/evidence/research/group/crossing_regimes.py \
    --output /work/output/p101-full --archive /work/output/p101-full.zip -- --threads 1 --accelerator cpu \
    > "$run_dir/output/p101-full.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run_dir/output/p101-full.exit"
exit "$status"
