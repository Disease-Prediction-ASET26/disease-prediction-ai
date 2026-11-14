"""Train disease classifier strategies on the Kaggle dataset and persist them.

Usage (from the project root)::

    python -m prediction_service.train --data data/dataset.csv --out models

Trains every strategy, prints a held-out accuracy comparison, and writes each
fitted strategy to ``<out>/<name>.joblib``. The best-scoring strategy is then
**calibrated** (``CalibratedClassifierCV``) and saved to ``<out>/best.joblib`` --
the artifact the FastAPI app serves. Calibration makes the probabilities the API
returns trustworthy rather than merely rank-ordered; disable it with
``--no-calibrate`` if you need the raw estimator.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from .artifact import manifest_path_for, write_model_manifest
from .classifier import (
    CalibratedStrategy,
    DecisionTreeStrategy,
    KNNStrategy,
    NaiveBayesStrategy,
    RandomForestStrategy,
    SklearnStrategy,
)
from .dataset import load_dataset
from .persistence import save_strategy

STRATEGIES: dict[str, type[SklearnStrategy]] = {
    "random_forest": RandomForestStrategy,
    "knn": KNNStrategy,
    "decision_tree": DecisionTreeStrategy,
    "naive_bayes": NaiveBayesStrategy,
}


def _accuracy(strategy: SklearnStrategy, features, labels) -> float:
    estimator = strategy._estimator
    predicted = estimator.predict(list(features))
    correct = sum(1 for p, y in zip(predicted, labels) if p == y)
    return correct / len(labels) if labels else 0.0


def _calibration_folds(labels, requested: int) -> int:
    """Clamp the CV fold count to what the smallest class can support.

    ``CalibratedClassifierCV`` needs at least one sample per class in every
    fold. Returning a value < 2 signals that calibration isn't viable.
    """
    if not labels:
        return 0
    smallest_class = min(Counter(labels).values())
    return min(requested, smallest_class)


def calibrate_best(
    strategy: SklearnStrategy,
    features,
    labels,
    *,
    method: str = "sigmoid",
    cv: int = 3,
) -> SklearnStrategy:
    """Return a calibrated copy of ``strategy`` fitted on the full dataset.

    Falls back to the original (uncalibrated) strategy when the dataset has too
    few samples per class for ``cv`` folds, so training never crashes on small
    inputs.
    """
    folds = _calibration_folds(labels, cv)
    if folds < 2:
        return strategy
    base = type(strategy)()
    calibrated = CalibratedStrategy(base, method=method, cv=folds)
    calibrated.fit(features, labels)
    return calibrated


def train_all(
    data_path: str | Path,
    out_dir: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    calibrate: bool = True,
    calibration_method: str = "sigmoid",
    calibration_cv: int = 3,
    training_revision: str | None = None,
) -> dict[str, float]:
    """Train every strategy, persist each, and return held-out accuracies.

    The best strategy is calibrated (unless ``calibrate=False``) before being
    saved as ``best.joblib`` -- the model the API serves.
    """
    from sklearn.model_selection import train_test_split

    features, labels = load_dataset(data_path)
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    scores: dict[str, float] = {}
    strategies: dict[str, SklearnStrategy] = {}
    for name, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls().fit(x_train, y_train)
        accuracy = _accuracy(strategy, x_test, y_test)
        scores[name] = accuracy

        # refit on the full dataset before persisting for production use
        strategy.fit(features, labels)
        save_strategy(strategy, out / f"{name}.joblib")
        strategies[name] = strategy

    best_name = max(scores, key=scores.get)
    best = strategies[best_name]
    if calibrate:
        best = calibrate_best(
            best,
            features,
            labels,
            method=calibration_method,
            cv=calibration_cv,
        )
    save_strategy(best, out / "best.joblib")
    calibration_applied = calibrate and _calibration_folds(labels, calibration_cv) >= 2
    write_model_manifest(
        out / "best.joblib",
        data_path,
        scores=scores,
        selected_strategy=best_name,
        calibrated=calibration_applied,
        calibration_method=calibration_method,
        training_revision=training_revision,
    )

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease classifiers.")
    parser.add_argument(
        "--data", default="data/dataset.csv", help="Path to dataset.csv"
    )
    parser.add_argument("--out", default="models", help="Directory for model artifacts")
    parser.add_argument(
        "--no-calibrate",
        dest="calibrate",
        action="store_false",
        help="Persist the best model without probability calibration.",
    )
    parser.add_argument(
        "--calibration-method",
        default="sigmoid",
        choices=("sigmoid", "isotonic"),
        help="Calibration method for the served model (default: sigmoid).",
    )
    parser.add_argument(
        "--source-revision",
        default=os.environ.get("GITHUB_SHA"),
        help="AI source commit used for training (defaults to GITHUB_SHA when set).",
    )
    args = parser.parse_args()

    scores = train_all(
        args.data,
        args.out,
        calibrate=args.calibrate,
        calibration_method=args.calibration_method,
        training_revision=args.source_revision,
    )

    print("\nHeld-out accuracy:")
    for name, accuracy in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<14} {accuracy:.4f}")
    best = max(scores, key=scores.get)
    calib = (
        f"calibrated ({args.calibration_method})" if args.calibrate else "uncalibrated"
    )
    print(f"\nBest model: {best} [{calib}] (saved to {args.out}/best.joblib)")
    manifest = manifest_path_for(Path(args.out) / "best.joblib")
    print(f"Model version: {json.loads(manifest.read_text())['model_version']}")


if __name__ == "__main__":
    main()
