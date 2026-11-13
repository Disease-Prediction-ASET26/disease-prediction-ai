"""Create and verify manifests for deployable classifier artifacts.

The classifier is trained outside the API process. Its manifest records the
artifact hash, training data hash, selected strategy, scores and Python runtime
so a packaged image can be tied back to the model that it serves.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import platform
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MANIFEST_SCHEMA_VERSION = 1


def manifest_path_for(model_path: str | Path) -> Path:
    """Return the sidecar manifest path for ``model_path``."""
    path = Path(model_path)
    return path.with_suffix(".manifest.json")


def sha256_file(path: str | Path) -> str:
    """Return a file's SHA-256 digest without loading it all into memory."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_model_manifest(
    model_path: str | Path,
    dataset_path: str | Path,
    *,
    scores: dict[str, float],
    selected_strategy: str,
    calibrated: bool,
    calibration_method: str,
    training_revision: str | None = None,
) -> dict[str, Any]:
    """Write a provenance manifest beside a trained model and return it."""
    model = Path(model_path)
    dataset = Path(dataset_path)
    model_digest = sha256_file(model)
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "model_version": f"sha256-{model_digest[:12]}",
        "created_at": datetime.now(UTC).isoformat(),
        "artifact": {
            "file": model.name,
            "sha256": model_digest,
            "size_bytes": model.stat().st_size,
        },
        "training": {
            "dataset_file": dataset.name,
            "dataset_sha256": sha256_file(dataset),
            "source_revision": training_revision,
            "selected_strategy": selected_strategy,
            "held_out_accuracy": scores,
            "calibrated": calibrated,
            "calibration_method": calibration_method if calibrated else "none",
        },
        "runtime": {
            "python": platform.python_version(),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        },
    }

    target = manifest_path_for(model)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(target)
    return manifest


def verify_model_artifact(
    model_path: str | Path,
    *,
    require_manifest: bool = False,
    check_runtime: bool = False,
) -> dict[str, Any] | None:
    """Check a model's sidecar, digest and optionally runtime compatibility.

    Older local models without manifests remain loadable unless the caller opts
    into ``require_manifest``. Distribution images should require the manifest.
    """
    model = Path(model_path)
    if not model.is_file():
        raise FileNotFoundError(f"Model artifact does not exist: {model}")

    manifest_path = manifest_path_for(model)
    if not manifest_path.is_file():
        if require_manifest:
            raise FileNotFoundError(f"Model manifest does not exist: {manifest_path}")
        return None

    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid model manifest: {manifest_path}") from exc
    if not isinstance(manifest, dict):
        raise TypeError(f"Invalid model manifest object: {manifest_path}")

    if manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported model manifest schema: {manifest.get('schema_version')}"
        )

    artifact = manifest.get("artifact", {})
    if not isinstance(artifact, dict):
        raise TypeError("Model manifest has an invalid artifact section.")
    if artifact.get("file") != model.name:
        raise ValueError("Model manifest file name does not match the model artifact.")

    expected_digest = artifact.get("sha256")
    if not isinstance(expected_digest, str) or not re.fullmatch(
        r"[0-9a-f]{64}", expected_digest
    ):
        raise ValueError("Model manifest has an invalid SHA-256 digest.")
    actual_digest = sha256_file(model)
    if actual_digest != expected_digest:
        raise ValueError(f"Model artifact checksum failed: {model}")
    if artifact.get("size_bytes") != model.stat().st_size:
        raise ValueError(f"Model artifact size does not match its manifest: {model}")

    model_version = manifest.get("model_version")
    if not isinstance(model_version, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model_version
    ):
        raise ValueError("Model manifest has an invalid model version.")
    if model_version != f"sha256-{expected_digest[:12]}":
        raise ValueError("Model version does not match the artifact checksum.")

    if check_runtime:
        runtime = manifest.get("runtime", {})
        if not isinstance(runtime, dict):
            raise ValueError("Model manifest has an invalid runtime section.")
        actual_runtime = {
            "python": platform.python_version(),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
        }
        if runtime != actual_runtime:
            raise ValueError(
                "Model runtime mismatch: "
                f"trained with {runtime}, running with {actual_runtime}. "
                "Train with the same pinned AI environment used by the image."
            )

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify a trained model artifact.")
    parser.add_argument("model_path", type=Path)
    parser.add_argument("--require-manifest", action="store_true")
    parser.add_argument("--check-runtime", action="store_true")
    args = parser.parse_args()

    manifest = verify_model_artifact(
        args.model_path,
        require_manifest=args.require_manifest,
        check_runtime=args.check_runtime,
    )
    if manifest is None:
        print(f"Verified legacy model without manifest: {args.model_path}")
    else:
        print(
            f"Verified model {manifest['model_version']}: "
            f"{manifest['artifact']['sha256']}"
        )


if __name__ == "__main__":
    main()
