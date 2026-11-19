import os
import subprocess
import unittest
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "publish-model-runtime.sh"
DIGEST = "sha256:" + "a" * 64
SOURCE_REVISION = subprocess.run(
    ["git", "rev-parse", "HEAD"],
    cwd=SCRIPT.parents[1],
    check=True,
    capture_output=True,
    text=True,
).stdout.strip()
SHORT_REVISION = SOURCE_REVISION[:7]
DEV_REPOSITORY = "ghcr.io/disease-prediction-aset26/disease-prediction-ai-dev"
PROD_REPOSITORY = "ghcr.io/disease-prediction-aset26/disease-prediction-ai"


class PublishModelRuntimeBaseImageTests(unittest.TestCase):
    def run_script(
        self, channel: str, base_image: str, model_version: str
    ) -> subprocess.CompletedProcess[str]:
        env = os.environ.copy()
        env["MODEL_VERSION"] = model_version
        return subprocess.run(
            [str(SCRIPT), channel, base_image, "unsupported", "stage-1"],
            cwd=SCRIPT.parents[1],
            env=env,
            check=False,
            capture_output=True,
            text=True,
        )

    def assert_base_accepted(
        self, channel: str, base_image: str, model_version: str
    ) -> None:
        result = self.run_script(channel, base_image, model_version)
        self.assertEqual(result.returncode, 2)
        self.assertIn("Architecture must be amd64 or arm64", result.stderr)

    def assert_base_rejected(
        self, channel: str, base_image: str, model_version: str
    ) -> None:
        result = self.run_script(channel, base_image, model_version)
        self.assertEqual(result.returncode, 2)
        self.assertNotIn("Architecture must be amd64 or arm64", result.stderr)

    def test_accepts_dev_tag_and_digest_for_same_revision(self) -> None:
        version = f"0.1.2-dev.{SHORT_REVISION}.44400575595.1"
        base = f"{DEV_REPOSITORY}:{version}@{DIGEST}"
        self.assert_base_accepted("dev", base, version)

    def test_accepts_earlier_dev_image_for_same_revision(self) -> None:
        runtime_version = f"0.1.2-dev.{SHORT_REVISION}.44400575595.1"
        base = f"{DEV_REPOSITORY}:0.1.2-dev.{SHORT_REVISION}.12345.2@{DIGEST}"
        self.assert_base_accepted("dev", base, runtime_version)

    def test_accepts_stable_tag_and_digest(self) -> None:
        base = f"{PROD_REPOSITORY}:0.1.2@{DIGEST}"
        self.assert_base_accepted("prod", base, "0.1.2")

    def test_rejects_wrong_package_bare_digest_and_non_semver_tags(self) -> None:
        version = f"0.1.2-dev.{SHORT_REVISION}.44400575595.1"
        invalid_refs = (
            f"ghcr.io/other/image:{version}@{DIGEST}",
            f"{DEV_REPOSITORY}@{DIGEST}",
            f"{DEV_REPOSITORY}:{version}@sha256:invalid",
            f"{DEV_REPOSITORY}:0.1.2+build@{DIGEST}",
            f"{DEV_REPOSITORY}:0.1.2-dev.{SHORT_REVISION}.01.1@{DIGEST}",
            f"{DEV_REPOSITORY}:0.1.2-dev.{SHORT_REVISION}.044400575595.1@{DIGEST}",
            f"{DEV_REPOSITORY}:0.1.2-dev.deadbee.12345.1@{DIGEST}",
        )
        for base in invalid_refs:
            with self.subTest(base=base):
                self.assert_base_rejected("dev", base, version)

    def test_rejects_invalid_runtime_semver(self) -> None:
        base = f"{DEV_REPOSITORY}:0.1.2-dev.{SHORT_REVISION}.44400575595.1@{DIGEST}"
        version = f"0.1.2-dev.{SHORT_REVISION}.044400575595.1"
        self.assert_base_rejected("dev", base, version)

    def test_rejects_mismatched_production_version(self) -> None:
        base = f"{PROD_REPOSITORY}:1.2.3@{DIGEST}"
        self.assert_base_rejected("prod", base, "0.1.2")


if __name__ == "__main__":
    unittest.main()
