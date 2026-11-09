"""Load and save trained classifier strategies as pickle artifacts."""

from __future__ import annotations

import pickle
from pathlib import Path

from .classifier import SklearnStrategy

DEFAULT_MODEL_PATH = Path("models/best.pkl")


def save_strategy(strategy: SklearnStrategy, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("wb") as handle:
        pickle.dump(strategy, handle)


def load_strategy(path: str | Path = DEFAULT_MODEL_PATH) -> SklearnStrategy:
    """Load a pickled, fitted strategy from disk."""
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(
            f"No trained model at {source}. Train one first:\n"
            "    python -m prediction_service.train --data data/dataset.csv --out models"
        )
    with source.open("rb") as handle:
        strategy = pickle.load(handle)
    if not isinstance(strategy, SklearnStrategy):
        raise TypeError(f"{source} does not contain a classifier strategy.")
    return strategy
