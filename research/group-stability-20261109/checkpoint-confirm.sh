#!/usr/bin/env bash
set -uo pipefail

run_dir=$(pwd)
image=relflow-evidence-wheel:81a7beaf1c547e4f53c6
export run_dir image

trial() {
    local seed=$1 policy=$2 name="checkpoint-seed-$1-$2" status
    if [[ -e "$run_dir/output/$name.log" || -e "$run_dir/output/$name.json" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e GROUP_WHEEL_SHA256=cabe120b55c0e552e64469ecf337171650c3fd087375e7d56dd629ddccde5e8b \
        -e PYTHONPATH=/work/input/package:/work/input/evidence/experiments:/work/input/evidence/experiments/group \
        -v "$run_dir:/work" --entrypoint python "$image" /work/input/research/group-stability-20261109/checkpoint.py \
        --seed "$seed" --policy "$policy" --output "/work/output/$name.json" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f trial

for seed in 10901 10902 10903; do
    for policy in fixed validation; do
        printf '%s %s\n' "$seed" "$policy"
    done
done | xargs -n 2 -P 6 bash -c 'trial "$@"' _
status=$?
printf '%s\n' "$status" > "$run_dir/output/checkpoint-confirm.exit"
exit "$status"
