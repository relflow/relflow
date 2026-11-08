#!/usr/bin/env bash
set -uo pipefail

run_dir=$(pwd)
name=checkpoint-crossing-seed-10101-validation
if [[ -e "$run_dir/output/$name.log" || -e "$run_dir/output/$name.json" ]]; then
    printf 'Refusing to overwrite %s\n' "$name" >&2
    exit 1
fi
docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
    --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
    -e PYTHONHASHSEED=10101 -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
    -e GROUP_WHEEL_SHA256=cabe120b55c0e552e64469ecf337171650c3fd087375e7d56dd629ddccde5e8b \
    -e PYTHONPATH=/work/input/package:/work/input/evidence/experiments:/work/input/evidence/research/group:/work/input/research/group-stability-20261109 \
    -v "$run_dir:/work" --entrypoint python relflow-evidence-wheel:81a7beaf1c547e4f53c6 \
    /work/input/research/group-stability-20261109/checkpoint_crossing.py \
    --seed 10101 --policy validation --output "/work/output/$name.json" > "$run_dir/output/$name.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run_dir/output/$name.exit"
exit "$status"
