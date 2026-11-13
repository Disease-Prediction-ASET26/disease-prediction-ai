"""Download the inference-only Hugging Face assets declared in model-assets.json."""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: download-model-assets.py <output-directory>")

    spec_path = Path(__file__).with_name("model-assets.json")
    specs = json.loads(spec_path.read_text(encoding="utf-8"))
    output_root = Path(sys.argv[1])

    for asset_name, spec in specs.items():
        target = output_root / asset_name
        snapshot_download(
            repo_id=spec["repo_id"],
            revision=spec["revision"],
            allow_patterns=spec["files"],
            local_dir=target,
        )
        # local_dir stores download metadata here. The model runtime needs only
        # the pinned inference files and must not keep a runtime download cache.
        shutil.rmtree(target / ".cache", ignore_errors=True)


if __name__ == "__main__":
    main()
