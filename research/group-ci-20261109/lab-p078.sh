#!/usr/bin/env bash
set -uo pipefail
run_dir=$(pwd)
if [[ -e "$run_dir/output/p078-full.log" ]]; then
    echo 'Refusing to overwrite P078 capture' >&2
    exit 1
fi
docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
    --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
    -e PYTHONHASHSEED=7801 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
    -e EVIDENCE_REVISION=fbc310352efaaad229e93737e81d876fbf344d5a \
    -e EVIDENCE_RELFLOW_REVISION=fbc310352efaaad229e93737e81d876fbf344d5a \
    -e EVIDENCE_WHEEL_SHA256=2bd7e085acd646d38dac2d7ff6d0cd2acb0108bc5babd43cc79f3b4d9e41d777 \
    -e PYTHONPATH=/work/input/package:/work/input/evidence:/work/input/evidence/experiments \
    -v "$run_dir:/work" --entrypoint python relflow-evidence-wheel:81a7beaf1c547e4f53c6 \
    /work/input/evidence/diagnostics/capture.py --source /work/input/evidence/experiments/group/regime_recovery.py \
    --output /work/output/p078-full --archive /work/output/p078-full.zip -- --threads 1 --accelerator cpu \
    > "$run_dir/output/p078-full.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run_dir/output/p078-full.exit"
exit "$status"
