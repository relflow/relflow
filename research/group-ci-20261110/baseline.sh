#!/usr/bin/env bash
set -uo pipefail
run_dir=$(pwd)
export run_dir
run_one() {
    local id=$1 source=$2 seed=$3 mode=$4 name="$1-$3-$4" status
    if [[ -e "$run_dir/output/$name.log" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    local -a cap=()
    if [[ "$mode" == smoke ]]; then cap=(--steps 32); fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e EVIDENCE_REVISION=a912f2912c90514aee295ef35c4fd0740e740d38 \
        -e EVIDENCE_RELFLOW_REVISION=a912f2912c90514aee295ef35c4fd0740e740d38 \
        -e EVIDENCE_WHEEL_SHA256=32abf377febb4d884fdf981a381fcf9f17a7b20ba9f1a5f7d5a8e788690b9cd6 \
        -e PYTHONPATH=/work/input/package:/work/input/evidence:/work/input/evidence/experiments \
        -v "$run_dir:/work" --entrypoint python relflow-evidence-wheel:81a7beaf1c547e4f53c6 \
        /work/input/evidence/diagnostics/capture.py --source "/work/input/evidence/experiments/$source.py" \
        --output "/work/output/$name" --archive "/work/output/$name.zip" -- --seed "$seed" --threads 1 \
        --accelerator cpu "${cap[@]}" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f run_one
mode=${1:?smoke or full}
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
while read -r id source seed; do
    printf '%s %s %s %s\n' "$id" "$source" "$seed" "$mode"
done <<'CASES' | xargs -n 4 -P 6 bash -c 'run_one "$@"' _
P018 cluster/hidden_regression_regimes 0
P019 cluster/plain_input_is_dormant 0
P020 cluster/reconstructing_category_labels 0
P065 vocabulary/cluster_pretraining_coverage 7651
P078 cluster/regime_recovery 7801
P095 cluster/predictive_tradeoff 9501
CASES
status=$?
printf '%s\n' "$status" > "$run_dir/output/$mode.exit"
exit "$status"
