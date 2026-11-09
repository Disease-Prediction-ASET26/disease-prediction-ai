"""Persist and load trained classifier strategies as joblib artifacts.

Two layers are provided:

* The module-level :func:`save_strategy` / :func:`load_strategy` helpers persist
  a single fitted strategy to one ``.joblib`` file. They are the simple path the
  training script and API default to.
* :class:`ModelStore` is a small **model registry**: it keeps versioned model
  artifacts together with their :class:`MLModelMetadata` sidecar and records
  which version is *active* (the one the API should serve). Use it when you need
  to keep several trained models around and promote between them.

joblib is used rather than :mod:`pickle` because it serialises the large NumPy
arrays inside scikit-learn estimators far more efficiently, and it is the
format scikit-learn itself recommends for model persistence.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import UUID, uuid4

import joblib

from .classifier import SklearnStrategy
from .models import MLModelMetadata

DEFAULT_MODEL_PATH = Path("models/best.joblib")

# The pre-joblib artifacts used this suffix; still honoured on load so older
# ``models/best.pkl`` files keep working.
_LEGACY_SUFFIX = ".pkl"


def _train_hint(path: Path) -> str:
    return (
        f"No trained model at {path}. Train one first:\n"
        "    python -m prediction_service.train --data data/dataset.csv --out models"
    )


def _resolve_existing(path: Path) -> Path:
    """Return ``path`` if it exists, else a legacy ``.pkl`` sibling if present."""
    if path.exists():
        return path
    legacy = path.with_suffix(_LEGACY_SUFFIX)
    if path.suffix != _LEGACY_SUFFIX and legacy.exists():
        return legacy
    raise FileNotFoundError(_train_hint(path))


def save_strategy(strategy: SklearnStrategy, path: str | Path) -> None:
    """Persist a fitted strategy to ``path`` using joblib."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(strategy, target)


def load_strategy(path: str | Path = DEFAULT_MODEL_PATH) -> SklearnStrategy:
    """Load a fitted strategy from disk (joblib, falling back to legacy pickle)."""
    source = _resolve_existing(Path(path))
    strategy = joblib.load(source)
    if not isinstance(strategy, SklearnStrategy):
        raise TypeError(f"{source} does not contain a classifier strategy.")
    return strategy


class ModelStore:
    """A versioned store of trained strategies plus their metadata.

    Layout under ``root``::

        <root>/
            <version>.joblib        # the fitted strategy
            <version>.json          # its MLModelMetadata sidecar
            active.json             # {"version": "<version>"} pointer

    The store never deletes artifacts implicitly; :meth:`save` adds a version
    and :meth:`set_active` repoints ``active.json`` at an existing one.
    """

    def __init__(self, root: str | Path = "models") -> None:
        self.root = Path(root)

    # -- paths -------------------------------------------------------------
    def _model_path(self, version: str) -> Path:
        return self.root / f"{version}.joblib"

    def _meta_path(self, version: str) -> Path:
        return self.root / f"{version}.json"

    @property
    def _active_pointer(self) -> Path:
        return self.root / "active.json"

    # -- writing -----------------------------------------------------------
    def save(
        self,
        strategy: SklearnStrategy,
        *,
        version: str | None = None,
        calibration_method: str = "none",
        make_active: bool = True,
    ) -> MLModelMetadata:
        """Persist ``strategy`` as a new version and return its metadata.

        ``version`` defaults to a UTC timestamp. When ``make_active`` is true
        (the default) the new version becomes the one :meth:`load_active`
        returns.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        now = datetime.now(UTC)
        version = version or now.strftime("%Y%m%dT%H%M%S")

        metadata = MLModelMetadata(
            model_id=uuid4(),
            version=version,
            algorithm=strategy.algorithm,
            calibration_method=calibration_method,
            trained_at=now,
            is_active=make_active,
        )

        joblib.dump(strategy, self._model_path(version))
        self._write_metadata(metadata)
        if make_active:
            self.set_active(version)
        return metadata

    def _write_metadata(self, metadata: MLModelMetadata) -> None:
        payload = asdict(metadata)
        payload["model_id"] = str(metadata.model_id)
        payload["trained_at"] = metadata.trained_at.isoformat()
        self._meta_path(metadata.version).write_text(
            json.dumps(payload, indent=2), encoding="utf-8"
        )

    def set_active(self, version: str) -> None:
        """Point the store's active model at an existing ``version``."""
        if not self._model_path(version).exists():
            raise FileNotFoundError(f"No model version {version!r} in {self.root}.")
        self._active_pointer.write_text(
            json.dumps({"version": version}), encoding="utf-8"
        )

    # -- reading -----------------------------------------------------------
    def versions(self) -> list[str]:
        """Return the stored versions, newest filename last (sorted)."""
        if not self.root.exists():
            return []
        return sorted(p.stem for p in self.root.glob("*.joblib"))

    def active_version(self) -> str | None:
        if not self._active_pointer.exists():
            return None
        data = json.loads(self._active_pointer.read_text(encoding="utf-8"))
        return data.get("version")

    def load(self, version: str) -> SklearnStrategy:
        path = self._model_path(version)
        if not path.exists():
            raise FileNotFoundError(f"No model version {version!r} in {self.root}.")
        strategy = joblib.load(path)
        if not isinstance(strategy, SklearnStrategy):
            raise TypeError(f"{path} does not contain a classifier strategy.")
        return strategy

    def load_active(self) -> SklearnStrategy:
        """Load the active strategy, or raise if none has been marked active."""
        version = self.active_version()
        if version is None:
            raise FileNotFoundError(_train_hint(self.root / "active.json"))
        return self.load(version)

    def metadata(self, version: str) -> MLModelMetadata:
        path = self._meta_path(version)
        if not path.exists():
            raise FileNotFoundError(f"No metadata for version {version!r}.")
        data = json.loads(path.read_text(encoding="utf-8"))
        return MLModelMetadata(
            model_id=UUID(data["model_id"]),
            version=data["version"],
            algorithm=data["algorithm"],
            calibration_method=data["calibration_method"],
            trained_at=datetime.fromisoformat(data["trained_at"]),
            is_active=data["version"] == self.active_version(),
        )
