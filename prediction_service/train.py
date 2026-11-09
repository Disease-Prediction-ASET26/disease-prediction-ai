"""Train disease classifier strategies on the Kaggle dataset and persist them.

Usage (from the project root)::

    python -m prediction_service.train --data data/dataset.csv --out models

Trains every strategy, prints a held-out accuracy comparison, and pickles each
fitted strategy to ``<out>/<name>.pkl`` plus the best one to ``<out>/best.pkl``.
The FastAPI app loads ``<out>/best.pkl`` (or a chosen model) at startup.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from .classifier import (
    DecisionTreeStrategy,
    KNNStrategy,
    NaiveBayesStrategy,
    RandomForestStrategy,
    SklearnStrategy,
)
from .dataset import load_dataset

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


def train_all(
    data_path: str | Path,
    out_dir: str | Path,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
) -> dict[str, float]:
    """Train every strategy, persist each, and return held-out accuracies."""
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
    for name, strategy_cls in STRATEGIES.items():
        strategy = strategy_cls().fit(x_train, y_train)
        accuracy = _accuracy(strategy, x_test, y_test)
        scores[name] = accuracy

        # refit on the full dataset before persisting for production use
        strategy.fit(features, labels)
        with (out / f"{name}.pkl").open("wb") as handle:
            pickle.dump(strategy, handle)

    best_name = max(scores, key=scores.get)
    with (out / f"{best_name}.pkl").open("rb") as handle:
        best_bytes = handle.read()
    (out / "best.pkl").write_bytes(best_bytes)

    return scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Train disease classifiers.")
    parser.add_argument(
        "--data", default="data/dataset.csv", help="Path to dataset.csv"
    )
    parser.add_argument("--out", default="models", help="Directory for model artifacts")
    args = parser.parse_args()

    scores = train_all(args.data, args.out)

    print("\nHeld-out accuracy:")
    for name, accuracy in sorted(scores.items(), key=lambda kv: kv[1], reverse=True):
        print(f"  {name:<14} {accuracy:.4f}")
    best = max(scores, key=scores.get)
    print(f"\nBest model: {best} (saved to {args.out}/best.pkl)")


if __name__ == "__main__":
    main()
