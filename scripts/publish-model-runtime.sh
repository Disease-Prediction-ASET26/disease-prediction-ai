#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/publish-model-runtime.sh <dev|prod> <base-image@sha256:digest> <amd64|arm64> <image-tag> [artifact-dir]

Build, scan, check offline inference, and publish one platform of a runtime image.
Use a native host for the selected platform. The artifact directory must contain
an already reviewed best.joblib and best.manifest.json. The CI release workflow
combines the tested platform digests into a multi-platform image.
USAGE
}

if [[ $# -lt 4 || $# -gt 5 ]]; then
  usage >&2
  exit 2
fi

channel="$1"
base_image="$2"
arch="$3"
tag="$4"
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/.." && pwd)"
artifact_dir="${5:-$repo_dir/models}"
image="${AI_RUNTIME_IMAGE:-ghcr.io/disease-prediction-aset26/disease-prediction-ai-runtime}"

case "$channel" in
  dev) expected_base=ghcr.io/disease-prediction-aset26/disease-prediction-ai-dev ;;
  prod) expected_base=ghcr.io/disease-prediction-aset26/disease-prediction-ai ;;
  *) echo "Channel must be dev or prod." >&2; exit 2 ;;
esac
if [[ "$base_image" != "$expected_base"@sha256:* || ! "$base_image" =~ @sha256:[0-9a-f]{64}$ ]]; then
  echo "The $channel runtime requires a digest-pinned $expected_base base image." >&2
  exit 2
fi
case "$arch" in
  amd64|arm64) ;;
  *) echo "Architecture must be amd64 or arm64." >&2; exit 2 ;;
esac
if [[ ! "$tag" =~ ^[A-Za-z0-9_][A-Za-z0-9._-]{0,127}$ ]]; then
  echo "Invalid image tag." >&2
  exit 2
fi
if ! command -v trivy >/dev/null 2>&1; then
  echo "Trivy is required to scan the model runtime before publishing." >&2
  exit 1
fi

model_path="$artifact_dir/best.joblib"
manifest_path="$artifact_dir/best.manifest.json"
if [[ ! -f "$model_path" || ! -f "$manifest_path" ]]; then
  echo "Expected an approved best.joblib and best.manifest.json in $artifact_dir." >&2
  exit 1
fi
PYTHONPATH="$repo_dir${PYTHONPATH:+:$PYTHONPATH}" \
  python3 -m prediction_service.artifact "$model_path" --require-manifest

model_assets_digest="$(sha256sum "$repo_dir/model-assets.json" | cut -d ' ' -f 1)"
model_training_revision="$(python3 - "$manifest_path" <<'PY'
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
model_source_revision="$(git -C "$repo_dir" rev-parse HEAD)"
model_classifier_digest="$(sha256sum "$model_path" | cut -d ' ' -f 1)"
if [[ ! "$model_training_revision" =~ ^(local|[0-9a-f]{40,64})$ ]]; then
  echo "The model manifest has an invalid source revision." >&2
  exit 1
fi

context="$(mktemp -d)"
trap 'rm -rf "$context"' EXIT
cp "$repo_dir/Dockerfile.model-runtime" "$context/Dockerfile"
cp "$repo_dir/model-assets.json" "$context/model-assets.json"
cp "$repo_dir/download-model-assets.py" "$context/download-model-assets.py"
cp -R "$repo_dir/prediction_service" "$context/prediction_service"
find "$context/prediction_service" -type d -name __pycache__ -prune -exec rm -rf {} +
cp "$model_path" "$context/best.joblib"
cp "$manifest_path" "$context/best.manifest.json"

docker buildx build \
  --platform "linux/$arch" \
  --load \
  --provenance=false \
  --sbom=false \
  --build-arg "AI_BASE_IMAGE=$base_image" \
  --build-arg "MODEL_ASSETS_SHA256=$model_assets_digest" \
  --build-arg "MODEL_SOURCE_SHA256=$model_source_digest" \
  --build-arg "MODEL_BUILD_SHA256=$model_build_digest" \
  --build-arg "MODEL_SOURCE_REVISION=$model_source_revision" \
  --build-arg "MODEL_TRAINING_REVISION=$model_training_revision" \
  --build-arg "MODEL_CLASSIFIER_SHA256=$model_classifier_digest" \
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

# Exercise the real classifier and parser without network access. This catches
# an incompatible runtime or missing model files before the image is published.
docker run --rm -i --network none --platform "linux/$arch" \
  --env OMP_NUM_THREADS=1 --env OPENBLAS_NUM_THREADS=1 \
  --entrypoint python "$image:$tag" - <<'PY'
from prediction_service.api import _build_pipeline

pipeline = _build_pipeline()
pipeline.warm_up()
parsed, predictions = pipeline.predict_detailed("I have a high fever and a cough")
if not parsed.active_symptoms() or not predictions:
    raise SystemExit("Offline prediction smoke check returned no symptoms or predictions")
print("Offline model readiness and prediction succeeded")
PY

docker push "$image:$tag"
digest="$(docker buildx imagetools inspect "$image:$tag" | awk '/^Digest:/ {print $2; exit}')"
if [[ ! "$digest" =~ ^sha256:[0-9a-f]{64}$ ]]; then
  echo "Registry did not return a valid image digest." >&2
  exit 1
fi
printf 'Published %s runtime: %s@%s\n' "$arch" "$image" "$digest"
if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  printf 'image-ref=%s@%s\n' "$image" "$digest" >> "$GITHUB_OUTPUT"
fi
