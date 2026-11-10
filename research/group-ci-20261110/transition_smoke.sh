#!/usr/bin/env bash
set -uo pipefail
run_dir=$(pwd)
if [[ -e "$run_dir/output/transition.log" ]]; then
    echo 'Refusing to overwrite transition smoke' >&2
    exit 1
fi
cd "$run_dir/input" || exit 1
test "$(sha256sum relflow-0.4.5-py3-none-any.whl | cut -d ' ' -f 1)" = 32abf377febb4d884fdf981a381fcf9f17a7b20ba9f1a5f7d5a8e788690b9cd6 || exit 1
tar -xzf group-transition-smoke-input-20261110.tar.gz
cat > Dockerfile <<'DOCKER'
FROM relflow-evidence-wheel:81a7beaf1c547e4f53c6
COPY relflow-0.4.5-py3-none-any.whl /tmp/baseline.whl
RUN python -m pip install --no-deps --no-index --force-reinstall /tmp/baseline.whl
DOCKER
# pip validates the wheel filename, so retain its actual name in the build recipe.
sed -i 's@baseline.whl@relflow-0.4.5-py3-none-any.whl@g' Dockerfile
docker build --network none --tag relflow-group-transition-baseline:20261110 . > "$run_dir/output/build.log" 2>&1 || exit 1
docker image inspect relflow-group-transition-baseline:20261110 --format '{{.Id}}' > "$run_dir/output/image-id.txt"
sha256sum group-transition-smoke-input-20261110.tar.gz relflow-0.4.5-py3-none-any.whl transition_probe.py > "$run_dir/output/inputs.sha256"
docker run --rm --network none --cpus 2 --memory 4g --pids-limit 512 \
    --user "$(id -u):$(id -g)" -e USER=lab -e HOME=/tmp \
    -e OMP_NUM_THREADS=1 -e MKL_NUM_THREADS=1 -e POLARS_MAX_THREADS=2 \
    -e EVIDENCE_REVISION=uncommitted-working-tree \
    -e EVIDENCE_RELFLOW_REVISION=a912f2912c90514aee295ef35c4fd0740e740d38 \
    -e EVIDENCE_WHEEL_SHA256=32abf377febb4d884fdf981a381fcf9f17a7b20ba9f1a5f7d5a8e788690b9cd6 \
    -e PYTHONPATH=/work/input/evidence:/work/input/evidence/experiments \
    -v "$run_dir:/work" --entrypoint python relflow-group-transition-baseline:20261110 \
    /work/input/transition_probe.py > "$run_dir/output/transition.log" 2>&1
status=$?
printf '%s\n' "$status" > "$run_dir/output/transition.exit"
exit "$status"
