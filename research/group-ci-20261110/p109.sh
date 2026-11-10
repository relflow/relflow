#!/usr/bin/env bash
set -uo pipefail
run_dir=$(pwd)
export run_dir
run_one() {
    local seed=$1 mode=$2 name="P109-$1-$2" status
    if [[ -e "$run_dir/output/$name.log" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    local -a cap=()
    if [[ "$mode" == smoke ]]; then cap=(--steps 32); fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e EVIDENCE_REVISION=uncommitted-working-tree \
        -e EVIDENCE_RELFLOW_REVISION=e24d757f74a4da5d92c33a950b42920cf7364d93 \
        -e EVIDENCE_WHEEL_SHA256=69f7162ed8ca5460036c96e5f26604d7ba72c8493392224e75c3553bbbc9a030 \
        -e PYTHONPATH=/work/input/package:/work/input/evidence:/work/input/evidence/experiments \
        -v "$run_dir:/work" --entrypoint python relflow-evidence-wheel:81a7beaf1c547e4f53c6 \
        /work/input/evidence/diagnostics/capture.py --source /work/input/evidence/experiments/group/plain_input_is_dormant.py \
        --output "/work/output/$name" --archive "/work/output/$name.zip" -- --seed "$seed" --threads 1 \
        --accelerator cpu "${cap[@]}" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f run_one
mode=${1:?smoke or full}
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
for seed in 10901 10902 10903; do printf '%s %s\n' "$seed" "$mode"; done | xargs -n 2 -P 3 bash -c 'run_one "$@"' _
status=$?
printf '%s\n' "$status" > "$run_dir/output/$mode.exit"
exit "$status"
