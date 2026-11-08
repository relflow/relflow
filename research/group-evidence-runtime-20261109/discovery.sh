#!/usr/bin/env bash
set -uo pipefail

# Run from a fresh lab directory with the recorded candidate wheel and sources.
run_dir=$(pwd)
image=relflow-evidence-wheel:81a7beaf1c547e4f53c6
export run_dir image

trial() {
    local seed=$1 rate=$2 name="seed-$1-lr-$2" status
    if [[ -e "$run_dir/output/$name.log" || -e "$run_dir/output/$name.json" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e PYTHONPATH=/work/input/package:/work/input/evidence/experiments:/work/input/evidence/experiments/group \
        -v "$run_dir:/work" --entrypoint python "$image" /work/input/scripts/group_lab.py \
        --seed "$seed" --groups 4 --learning-rate "$rate" --steps 1200 --accelerator cpu \
        --output "/work/output/$name.json" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f trial

for seed in 7801 7802 7803; do
    for rate in 0.003 0.006 0.01 0.02; do
        printf '%s %s\n' "$seed" "$rate"
    done
done | xargs -n 2 -P 6 bash -c 'trial "$@"' _
status=$?
printf '%s\n' "$status" > "$run_dir/output/discovery.exit"
exit "$status"
