#!/usr/bin/env bash
set -uo pipefail

# Run inside a prepared directory containing input/ and output/. The input
# wheel must be extracted to input/package; SHA256SUMS records the inputs.
run_dir=$(pwd)
image=${GROUP_LAB_IMAGE:-relflow-evidence-wheel:81a7beaf1c547e4f53c6}
export run_dir image

trial() {
    local seed=$1 groups=$2 rate=$3 name
    name="seed-${seed}-groups-${groups}-lr-${rate}"
    if [[ -e "$run_dir/output/$name.log" || -e "$run_dir/output/$name.json" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONPATH=/work/input/package:/work/input/evidence/experiments/group:/work/input/evidence/experiments \
        -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -v "$run_dir:/work" \
        --entrypoint python "$image" /work/input/scripts/group_lab.py \
        --seed "$seed" --groups "$groups" --learning-rate "$rate" \
        --steps 600 --accelerator cpu --output "/work/output/$name.json" \
        > "$run_dir/output/$name.log" 2>&1
    local result=$?
    printf '%s\n' "$result" > "$run_dir/output/$name.exit"
    return "$result"
}
export -f trial

for seed in 7801 7802 7803; do
    for groups in 4 8; do
        for rate in 0.001 0.003 0.01; do
            printf '%s %s %s\n' "$seed" "$groups" "$rate"
        done
    done
done | xargs -n 3 -P 2 bash -c 'trial "$@"' _
result=$?
printf '%s\n' "$result" > "$run_dir/output/sweep.exit"
exit "$result"
