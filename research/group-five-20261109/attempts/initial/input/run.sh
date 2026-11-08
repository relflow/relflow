#!/usr/bin/env bash
set -uo pipefail
run_dir=$(pwd)
export run_dir
run_one() {
    local id=$1 source=$2 seed=$3 mode=$4 name status
    name="$id-$seed-$mode"
    if [[ -e "$run_dir/output/$name.log" ]]; then
        printf 'Refusing to overwrite %s\n' "$name" >&2
        return 1
    fi
    local -a steps=()
    if [[ "$mode" == smoke ]]; then steps=(--steps 32); fi
    docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
        --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
        -e PYTHONHASHSEED="$seed" -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
        -e EVIDENCE_REVISION=uncommitted-working-tree \
        -e EVIDENCE_RELFLOW_REVISION=e5d6f42010b1802edc6687ea55c1032c3badd3af \
        -e EVIDENCE_WHEEL_SHA256=cabe120b55c0e552e64469ecf337171650c3fd087375e7d56dd629ddccde5e8b \
        -e PYTHONPATH=/work/input/package:/work/input/evidence:/work/input/evidence/experiments \
        -v "$run_dir:/work" --entrypoint python relflow-evidence-wheel:81a7beaf1c547e4f53c6 \
        /work/input/evidence/diagnostics/capture.py --source "/work/input/evidence/research/group/$source.py" \
        --output "/work/output/$name" --archive "/work/output/$name.zip" -- --seed "$seed" --threads 1 \
        --accelerator cpu "${steps[@]}" > "$run_dir/output/$name.log" 2>&1
    status=$?
    printf '%s\n' "$status" > "$run_dir/output/$name.exit"
    return "$status"
}
export -f run_one
mode=${1:?smoke or full}
case "$mode" in smoke|full) ;; *) exit 2 ;; esac
while read -r id source seed; do
    printf '%s %s %s %s\n' "$id" "$source" "$seed" "$mode"
    if [[ "$mode" == full ]]; then
        printf '%s %s %s %s\n' "$id" "$source" "$((seed+1))" "$mode"
        printf '%s %s %s %s\n' "$id" "$source" "$((seed+2))" "$mode"
    fi
done <<'CASES' | xargs -n 4 -P 10 bash -c 'run_one "$@"' _
P102 rare_identities 10201
P103 group_capacity 10301
P104 unavailable_identities 10401
P105 assignment_intervention 10501
P106 behavior_drift 10601
CASES
status=$?
printf '%s\n' "$status" > "$run_dir/output/$mode.exit"
exit "$status"
