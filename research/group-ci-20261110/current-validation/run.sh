#!/usr/bin/env bash
set -uo pipefail
run_dir=$(pwd)
export run_dir
run_one() {
    local role=$1 id=$2 seed=$3 source=$4 mode=$5 name="$1-$2-$5" revision wheel image status
    [[ ! -e "$run_dir/output/$name.log" ]] || { echo "Refusing to overwrite $name" >&2; return 1; }
    if [[ "$role" == baseline ]]; then
        revision=c0cae6df38010e232c0f12c966eaa6a06fae4b94
        wheel=a2e8e74c5c751a71fcdb83fa3659318551eda0f8241471e3883cdb9c14e8fa38
    else
        revision=20469b2c3e7b119ca1067e6df1f6fa46d33f708f
        wheel=ca9c1a5fde9f17c867cd7e64a1411d3854d4c1172a31c23f3fa1e65b547a8297
    fi
    image="relflow-group-c0ca-$role:20261110"
    local -a cap=()
    if [[ "$mode" == smoke ]]; then cap=(--steps 32); fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e EVIDENCE_REVISION=uncommitted-working-tree \
        -e EVIDENCE_RELFLOW_REVISION="$revision" -e EVIDENCE_WHEEL_SHA256="$wheel" \
        -e PYTHONPATH=/work/input/evidence:/work/input/evidence/experiments \
        -v "$run_dir:/work" --entrypoint python "$image" \
        /work/input/evidence/diagnostics/capture.py --source "/work/input/evidence/$source" \
        --output "/work/output/$name" --archive "/work/output/$name.zip" -- --seed "$seed" \
        --threads 1 --accelerator cpu "${cap[@]}" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f run_one
mode=${1:?smoke or full}
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
cat <<EOF | xargs -n 5 -P 6 bash -c 'run_one "$@"' _
candidate P112 11201 experiments/group/predictive_tradeoff.py $mode
candidate P111 11101 experiments/group/regime_recovery.py $mode
candidate P108 10801 experiments/group/hidden_regression_regimes.py $mode
candidate P109 10901 experiments/group/plain_input_is_dormant.py $mode
candidate P110 11001 experiments/group/reconstructing_category_labels.py $mode
baseline P018 0 migrations/cluster_to_group/P018.py $mode
baseline P019 0 migrations/cluster_to_group/P019.py $mode
baseline P020 0 migrations/cluster_to_group/P020.py $mode
baseline P065 7651 migrations/cluster_to_group/P065.py $mode
baseline P078 7801 migrations/cluster_to_group/P078.py $mode
baseline P095 9501 migrations/cluster_to_group/P095.py $mode
EOF
status=$?
printf '%s\n' "$status" > "$run_dir/output/$mode.exit"
exit "$status"
