"""Generate human-readable explanations for a disease prediction.

Given the feature vector that produced a prediction and the fitted estimator,
:class:`ExplanationGenerator` attributes the prediction to the symptoms that
drove it, producing an :class:`Explanation` (a short summary plus per-symptom
contribution weights).

Two attribution sources are supported, in order of preference:

* **Feature importances** (``feature_importances_``) -- tree-based models
  (RandomForest, DecisionTree) expose a global importance per feature.
* **Linear coefficients** (``coef_``) -- linear models expose per-class weights.

Either way only the symptoms that are *active* in the input vector can
contribute, so the explanation reflects what the patient actually reported. If
the estimator exposes neither, a presence-based fallback is used: every active
symptom shares the weight equally.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .models import Explanation
from .vocabulary import CANONICAL_SYMPTOMS

if TYPE_CHECKING:
    from collections.abc import Sequence


def _unwrap(estimator: Any) -> Any:
    """Return the underlying estimator, peeling a calibration wrapper if any.

    ``CalibratedClassifierCV`` hides the real model; its fitted sub-estimators
    expose the base estimator we want the importances from.
    """
    calibrated = getattr(estimator, "calibrated_classifiers_", None)
    if calibrated:
        inner = getattr(calibrated[0], "estimator", None)
        if inner is not None:
            return inner
    return estimator


def _importances(estimator: Any, class_index: int | None) -> Sequence[float] | None:
    """Pull a per-feature importance vector from the estimator, if it has one."""
    model = _unwrap(estimator)

    importances = getattr(model, "feature_importances_", None)
    if importances is not None:
        return list(importances)

    coef = getattr(model, "coef_", None)
    if coef is not None:
        row = (
            coef[class_index] if class_index is not None and coef.ndim > 1 else coef[0]
        )
        return [abs(float(w)) for w in row]

    return None


class ExplanationGenerator:
    """Attribute a prediction to the symptoms that drove it."""

    def __init__(self, vocabulary: tuple[str, ...] = CANONICAL_SYMPTOMS) -> None:
        self._vocabulary = vocabulary

    def explain(
        self,
        estimator: Any,
        feature_vector: Sequence[int],
        disease_name: str,
    ) -> Explanation:
        """Build an :class:`Explanation` for ``disease_name``.

        Only active symptoms (non-zero entries in ``feature_vector``) can
        contribute. Weights are normalised to sum to 1 across the contributing
        symptoms so they read as proportions.
        """
        active = [i for i, value in enumerate(feature_vector) if value]
        if not active:
            return Explanation(summary="No symptoms were available to explain.")

        class_index = self._class_index(estimator, disease_name)
        importances = _importances(estimator, class_index)

        if importances is not None:
            raw = {i: importances[i] for i in active if importances[i] > 0}
        else:
            raw = {}

        if not raw:
            # Fallback: every active symptom contributes equally.
            raw = {i: 1.0 for i in active}

        total = sum(raw.values())
        weights = {
            self._vocabulary[i]: round(value / total, 4)
            for i, value in sorted(raw.items(), key=lambda kv: kv[1], reverse=True)
        }

        return Explanation(
            summary=self._summarise(disease_name, weights),
            contribution_weights=weights,
        )

    @staticmethod
    def _class_index(estimator: Any, disease_name: str) -> int | None:
        classes = getattr(estimator, "classes_", None)
        if classes is None:
            return None
        for index, label in enumerate(classes):
            if str(label) == disease_name:
                return index
        return None

    @staticmethod
    def _summarise(disease_name: str, weights: dict[str, float]) -> str:
        if not weights:
            return f"{disease_name} predicted, but no driving symptoms were found."
        top = list(weights)[:3]
        pretty = ", ".join(symptom.replace("_", " ").strip() for symptom in top)
        return f"{disease_name} is driven mainly by: {pretty}."
