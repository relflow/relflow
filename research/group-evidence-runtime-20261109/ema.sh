#!/usr/bin/env bash
set -uo pipefail

run_dir=$(pwd)
image=relflow-evidence-wheel:81a7beaf1c547e4f53c6
export run_dir image

trial() {
    local seed=$1 rate=$2 ema=$3 name="ema-seed-$1-lr-$2-$3" status
    if [[ -e "$run_dir/output/$name.log" || -e "$run_dir/output/$name.json" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e PYTHONPATH=/work/input/package:/work/input/evidence/experiments:/work/input/evidence/experiments/group \
        -v "$run_dir:/work" --entrypoint python "$image" /work/input/research/group-evidence-runtime-20261109/ema.py \
        --seed "$seed" --learning-rate "$rate" --ema "$ema" \
        --output "/work/output/$name.json" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f trial

for seed in 7801 7802 7803; do
    for ema in none 0.9 0.99 0.999; do
        printf '%s 0.003 %s\n' "$seed" "$ema"
    done
    printf '%s 0.006 none\n' "$seed"
done | xargs -n 3 -P 6 bash -c 'trial "$@"' _
status=$?
printf '%s\n' "$status" > "$run_dir/output/ema.exit"
exit "$status"
