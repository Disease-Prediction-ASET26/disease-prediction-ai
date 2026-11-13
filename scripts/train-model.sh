#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
dataset="${1:-$repo_dir/data/dataset.csv}"
image="disease-prediction-ai:trainer"

if [[ ! -f "$dataset" ]]; then
  echo "Dataset not found: $dataset" >&2
  echo "Download dataset.csv from Kaggle into disease-prediction-ai/data first." >&2
  exit 1
fi

if [[ -n "$(git -C "$repo_dir" status --porcelain)" ]]; then
  echo "Commit or stash AI source changes before training a release model." >&2
  exit 1
fi

mkdir -p "$repo_dir/models"
docker build \
  --file "$repo_dir/Dockerfile.local" \
  --build-arg "LOCAL_UID=$(id -u)" \
  --build-arg "LOCAL_GID=$(id -g)" \
  --tag "$image" \
  "$repo_dir"

docker run --rm \
  --user "$(id -u):$(id -g)" \
  --volume "$(dirname -- "$dataset"):/training-data:ro" \
  --volume "$repo_dir/models:/app/models" \
  --entrypoint python \
  "$image" \
  -m prediction_service.train \
  --data "/training-data/$(basename -- "$dataset")" \
  --out /app/models \
  --source-revision "$(git -C "$repo_dir" rev-parse HEAD)"
