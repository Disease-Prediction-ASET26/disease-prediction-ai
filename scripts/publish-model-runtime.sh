#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage: scripts/publish-model-runtime.sh <dev|prod> <ai-base-image@sha256:digest> [artifact-dir]

Package a trained best.joblib and pinned ClinicalBERT/MiniLM inference weights
into the matching AI image, then publish an ARM64 GHCR image. Authenticate to
ghcr.io with a token that can read and write packages. Trivy must be installed
locally; the script scans before it pushes.
EOF
}

if [[ $# -lt 2 || $# -gt 3 ]]; then
  usage >&2
  exit 2
fi

channel="$1"
base_image="$2"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
artifact_dir="${3:-$repo_dir/models}"
image="${AI_RUNTIME_IMAGE:-ghcr.io/disease-prediction-aset26/disease-prediction-ai-runtime}"

if [[ "$channel" != "dev" && "$channel" != "prod" ]]; then
  echo "Channel must be 'dev' or 'prod'." >&2
  exit 2
fi

if [[ ! "$base_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "Base image must be pinned by sha256 digest." >&2
  exit 2
fi
if ! command -v trivy >/dev/null 2>&1; then
  echo "Trivy is required to scan the model runtime before publishing." >&2
  exit 1
fi

case "$channel" in
  dev)
    expected_base="ghcr.io/disease-prediction-aset26/disease-prediction-ai-dev@sha256:"
    ;;
  prod)
    expected_base="ghcr.io/disease-prediction-aset26/disease-prediction-ai@sha256:"
    ;;
esac
if [[ "$base_image" != "$expected_base"* ]]; then
  echo "The $channel runtime must use $expected_base<digest>." >&2
  exit 2
fi

model_path="$artifact_dir/best.joblib"
manifest_path="$artifact_dir/best.manifest.json"
if [[ ! -f "$model_path" || ! -f "$manifest_path" ]]; then
  echo "Expected $model_path and $manifest_path. Train a model first." >&2
  exit 1
fi

PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m prediction_service.artifact "$model_path" --require-manifest

image_version="$(python3 - "$manifest_path" <<'PY'
import hashlib
import json
import sys

with open(sys.argv[1], encoding="utf-8") as manifest_file:
    manifest_bytes = manifest_file.read().encode("utf-8")
manifest = json.loads(manifest_bytes)
manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()[:8]
print(f"{manifest['model_version']}-{manifest_hash}")
PY
)"

model_assets_digest="$(sha256sum "$repo_dir/model-assets.json" | cut -d ' ' -f 1)"
model_source_revision="$(python3 - "$manifest_path" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest.get("training", {}).get("source_revision") or "local")
PY
)"
model_build_digest="$(sha256sum \
  "$repo_dir/Dockerfile.model-runtime" \
  "$repo_dir/download-model-assets.py" \
  | sha256sum | cut -d ' ' -f 1)"
model_source_digest="$(python3 - "$repo_dir/prediction_service" <<'PY'
import hashlib
import sys
from pathlib import Path

root = Path(sys.argv[1])
digest = hashlib.sha256()
for path in sorted(root.rglob("*.py")):
    relative = path.relative_to(root).as_posix().encode("utf-8")
    content = path.read_bytes()
    digest.update(len(relative).to_bytes(4, "big"))
    digest.update(relative)
    digest.update(len(content).to_bytes(8, "big"))
    digest.update(content)
print(digest.hexdigest())
PY
)"

if [[ ! "$model_source_revision" =~ ^(local|[0-9a-f]{40,64})$ ]]; then
  echo "The model manifest has an invalid source revision." >&2
  exit 1
fi
if [[ ! "$image_version" =~ ^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "The model manifest contains an invalid image tag version." >&2
  exit 1
fi

base_digest="${base_image##*@sha256:}"
tag="$channel-$image_version-assets${model_assets_digest:0:8}-src${model_source_digest:0:8}-build${model_build_digest:0:8}-${base_digest:0:8}"
context="$(mktemp -d)"
metadata_file="$(mktemp)"
cleanup() {
  rm -rf "$context"
  rm -f "$metadata_file"
}
trap cleanup EXIT

cp "$script_dir/../Dockerfile.model-runtime" "$context/Dockerfile"
cp "$repo_dir/model-assets.json" "$context/model-assets.json"
cp "$repo_dir/download-model-assets.py" "$context/download-model-assets.py"
cp -R "$repo_dir/prediction_service" "$context/prediction_service"
find "$context/prediction_service" -type d -name __pycache__ -prune -exec rm -rf {} +
cp "$model_path" "$context/best.joblib"
cp "$manifest_path" "$context/best.manifest.json"

docker buildx build \
  --platform linux/arm64 \
  --load \
  --provenance=false \
  --sbom=false \
  --build-arg "AI_BASE_IMAGE=$base_image" \
  --build-arg "MODEL_ASSETS_SHA256=$model_assets_digest" \
  --build-arg "MODEL_SOURCE_SHA256=$model_source_digest" \
  --build-arg "MODEL_BUILD_SHA256=$model_build_digest" \
  --build-arg "MODEL_SOURCE_REVISION=$model_source_revision" \
  --build-arg "MODEL_VERSION=$tag" \
  --tag "$image:$tag" \
  "$context"

trivy image \
  --scanners vuln \
  --severity HIGH,CRITICAL \
  --ignore-unfixed \
  --exit-code 1 \
  --no-progress \
  "$image:$tag"

docker buildx build \
  --platform linux/arm64 \
  --push \
  --provenance=false \
  --sbom=false \
  --build-arg "AI_BASE_IMAGE=$base_image" \
  --build-arg "MODEL_ASSETS_SHA256=$model_assets_digest" \
  --build-arg "MODEL_SOURCE_SHA256=$model_source_digest" \
  --build-arg "MODEL_BUILD_SHA256=$model_build_digest" \
  --build-arg "MODEL_SOURCE_REVISION=$model_source_revision" \
  --build-arg "MODEL_VERSION=$tag" \
  --tag "$image:$tag" \
  --metadata-file "$metadata_file" \
  "$context"

digest="$(python3 - "$metadata_file" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as metadata_file:
    print(json.load(metadata_file).get("containerimage.digest", ""))
PY
)"

if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Buildx did not return a valid image digest." >&2
  exit 1
fi

echo "Published model runtime: $image:$tag"
echo "Pin this image in GitOps: $image@$digest"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'image-ref=%s@%s\n' "$image" "$digest" >> "$GITHUB_OUTPUT"
fi
