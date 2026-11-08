#!/usr/bin/env bash
set -uo pipefail

run_dir=$(pwd)
image=relflow-evidence-wheel:81a7beaf1c547e4f53c6
export run_dir image

trial() {
    local seed=$1 name="soft-seed-$1" status
    if [[ -e "$run_dir/output/$name.log" || -e "$run_dir/output/$name.json" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e PYTHONPATH=/work/input/package:/work/input/scripts:/work/input/evidence/experiments:/work/input/evidence/experiments/group \
        -v "$run_dir:/work" --entrypoint python "$image" /work/input/research/group-stability-20261109/soft.py \
        --seed "$seed" --output "/work/output/$name.json" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f trial

printf '%s\n' 7801 7802 7803 8801 8802 8803 | xargs -n 1 -P 6 bash -c 'trial "$@"' _
status=$?
printf '%s\n' "$status" > "$run_dir/output/soft.exit"
exit "$status"
