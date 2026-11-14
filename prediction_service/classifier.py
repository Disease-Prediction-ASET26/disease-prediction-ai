from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

from .explanation import ExplanationGenerator
from .features import encode_entries
from .models import DiseasePrediction, SymptomEntryDTO

if TYPE_CHECKING:
    from collections.abc import Sequence


class ClassifierStrategy(ABC):
    @abstractmethod
    def classify(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]: ...


class SklearnStrategy(ClassifierStrategy):
    """Base strategy wrapping any scikit-learn ``predict_proba`` estimator.

    Subclasses only provide an estimator via :meth:`_build_estimator`. The base
    handles encoding entries into feature vectors, training, and turning the
    model's class probabilities into ranked :class:`DiseasePrediction` objects.
    """

    def __init__(self, estimator: Any | None = None, *, top_k: int = 5) -> None:
        self._estimator = estimator or self._build_estimator()
        self._top_k = top_k
        self._fitted = estimator is not None
        self._explainer: ExplanationGenerator | None = None

    def with_explanations(
        self, generator: ExplanationGenerator | None = None
    ) -> SklearnStrategy:
        """Enable per-prediction explanations, returning ``self`` for chaining.

        When enabled, each :class:`DiseasePrediction` carries an
        :class:`~prediction_service.models.Explanation` attributing it to the
        symptoms that drove it.
        """
        self._explainer = generator or ExplanationGenerator()
        return self

    @abstractmethod
    def _build_estimator(self) -> Any:
        """Return a fresh, unfitted scikit-learn estimator."""

    @property
    def algorithm(self) -> str:
        return type(self._estimator).__name__

    def fit(
        self,
        features: Sequence[Sequence[int]],
        labels: Sequence[str],
    ) -> SklearnStrategy:
        """Train the estimator on encoded feature vectors and disease labels."""
        self._estimator.fit(list(features), list(labels))
        self._fitted = True
        return self

    def classify(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        if not self._fitted:
            raise RuntimeError(
                f"{type(self).__name__} must be trained with .fit(...) "
                "before calling classify()."
            )
        vector = encode_entries(entries)
        probabilities = self._estimator.predict_proba([vector])[0]
        classes = self._estimator.classes_
        ranked = sorted(
            zip(classes, probabilities), key=lambda pair: pair[1], reverse=True
        )
        return [
            DiseasePrediction(
                rank=rank,
                probability=float(probability),
                disease_name=str(disease),
                explanation=(
                    self._explainer.explain(self._estimator, vector, str(disease))
                    if self._explainer is not None
                    else None
                ),
            )
            for rank, (disease, probability) in enumerate(
                ranked[: self._top_k], start=1
            )
        ]


class RandomForestStrategy(SklearnStrategy):
    def __init__(
        self,
        estimator: Any | None = None,
        *,
        top_k: int = 5,
        n_estimators: int = 200,
        random_state: int = 42,
    ) -> None:
        self._n_estimators = n_estimators
        self._random_state = random_state
        super().__init__(estimator, top_k=top_k)

    def _build_estimator(self) -> Any:
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=self._n_estimators,
            random_state=self._random_state,
        )


class KNNStrategy(SklearnStrategy):
    def __init__(
        self,
        estimator: Any | None = None,
        *,
        top_k: int = 5,
        n_neighbors: int = 5,
    ) -> None:
        self._n_neighbors = n_neighbors
        super().__init__(estimator, top_k=top_k)

    def _build_estimator(self) -> Any:
        from sklearn.neighbors import KNeighborsClassifier

        return KNeighborsClassifier(n_neighbors=self._n_neighbors)


class DecisionTreeStrategy(SklearnStrategy):
    def __init__(
        self,
        estimator: Any | None = None,
        *,
        top_k: int = 5,
        random_state: int = 42,
    ) -> None:
        self._random_state = random_state
        super().__init__(estimator, top_k=top_k)

    def _build_estimator(self) -> Any:
        from sklearn.tree import DecisionTreeClassifier

        return DecisionTreeClassifier(random_state=self._random_state)


class NaiveBayesStrategy(SklearnStrategy):
    def _build_estimator(self) -> Any:
        from sklearn.naive_bayes import GaussianNB

        return GaussianNB()


class CalibratedStrategy(SklearnStrategy):
    """Wrap another strategy's estimator in ``CalibratedClassifierCV``.

    The raw ``predict_proba`` of tree ensembles and naive Bayes is often poorly
    calibrated -- the numbers rank correctly but don't behave like true
    probabilities. This strategy fits the base estimator and then calibrates it
    (Platt scaling via ``"sigmoid"`` or ``"isotonic"``), so the probabilities
    attached to each :class:`DiseasePrediction` are trustworthy.

    It presents the same interface as any other strategy, so it drops straight
    into :class:`DiseaseClassifier`, the pipeline, and persistence. ``cv`` is
    the number of calibration folds; keep it small on datasets with few samples
    per class.
    """

    def __init__(
        self,
        base: SklearnStrategy,
        *,
        method: str = "sigmoid",
        cv: int = 3,
        top_k: int | None = None,
    ) -> None:
        if method not in ("sigmoid", "isotonic"):
            raise ValueError(f"method must be 'sigmoid' or 'isotonic', got {method!r}.")
        self._base = base
        self.method = method
        self._cv = cv
        super().__init__(top_k=top_k if top_k is not None else base._top_k)

    def _build_estimator(self) -> Any:
        from sklearn.calibration import CalibratedClassifierCV

        return CalibratedClassifierCV(
            estimator=self._base._estimator,
            method=self.method,
            cv=self._cv,
        )

    @property
    def algorithm(self) -> str:
        return f"Calibrated({self._base.algorithm}, {self.method})"

    @property
    def calibration_method(self) -> str:
        return self.method


class DiseaseClassifier:
    def __init__(self, strategy: ClassifierStrategy) -> None:
        self._strategy = strategy

    def set_strategy(self, strategy: ClassifierStrategy) -> None:
        self._strategy = strategy

    def predict(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        return self._strategy.classify(entries)
