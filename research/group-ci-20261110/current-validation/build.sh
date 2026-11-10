#!/usr/bin/env bash
set -euo pipefail
cd input
printf '%s  %s\n' a2e8e74c5c751a71fcdb83fa3659318551eda0f8241471e3883cdb9c14e8fa38 wheels/baseline/relflow-0.4.6-py3-none-any.whl | sha256sum -c -
printf '%s  %s\n' ca9c1a5fde9f17c867cd7e64a1411d3854d4c1172a31c23f3fa1e65b547a8297 wheels/candidate/relflow-0.4.6-py3-none-any.whl | sha256sum -c -
for role in baseline candidate; do
    cat > "Dockerfile.$role" <<EOF
FROM relflow-evidence-wheel:81a7beaf1c547e4f53c6
COPY wheels/$role/relflow-0.4.6-py3-none-any.whl /tmp/relflow-0.4.6-py3-none-any.whl
RUN python -m pip install --no-deps --no-index --force-reinstall /tmp/relflow-0.4.6-py3-none-any.whl
EOF
    docker build --network none -f "Dockerfile.$role" -t "relflow-group-c0ca-$role:20261110" .
    docker image inspect "relflow-group-c0ca-$role:20261110" --format '{{.Id}}' > "../output/$role-image-id.txt"
done
