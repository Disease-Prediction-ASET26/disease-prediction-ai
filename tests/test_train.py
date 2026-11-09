from __future__ import annotations

from uuid import uuid4

from prediction_service.classifier import (
    CalibratedStrategy,
    RandomForestStrategy,
    SklearnStrategy,
)
from prediction_service.features import encode_entries
from prediction_service.models import Severity, SymptomEntryDTO
from prediction_service.train import _calibration_folds, calibrate_best


def _entry(symptom: str) -> SymptomEntryDTO:
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=symptom,
        severity=Severity.MODERATE,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=symptom,
    )


def _dataset(copies: int):
    flu = encode_entries([_entry("high_fever"), _entry("cough")])
    rash = encode_entries([_entry("itching"), _entry("skin_rash")])
    features = [flu] * copies + [rash] * copies
    labels = ["Flu"] * copies + ["Rash"] * copies
    return features, labels


# ---------------------------------------------------------------------------
# _calibration_folds
# ---------------------------------------------------------------------------


def test_folds_clamped_to_smallest_class():
    labels = ["A", "A", "A", "B", "B"]  # smallest class has 2
    assert _calibration_folds(labels, requested=5) == 2


def test_folds_use_requested_when_classes_large_enough():
    labels = ["A"] * 10 + ["B"] * 10
    assert _calibration_folds(labels, requested=3) == 3


def test_folds_zero_for_empty():
    assert _calibration_folds([], requested=3) == 0


# ---------------------------------------------------------------------------
# calibrate_best
# ---------------------------------------------------------------------------


def test_calibrate_best_returns_calibrated_strategy():
    features, labels = _dataset(6)
    base = RandomForestStrategy().fit(features, labels)

    result = calibrate_best(base, features, labels, cv=3)

    assert isinstance(result, CalibratedStrategy)
    assert result.calibration_method == "sigmoid"
    predictions = result.classify([_entry("high_fever"), _entry("cough")])
    assert predictions[0].disease_name == "Flu"


def test_calibrate_best_falls_back_when_too_few_samples():
    # 1 sample per class -> calibration not viable -> original returned.
    features, labels = _dataset(1)
    base = RandomForestStrategy().fit(features, labels)

    result = calibrate_best(base, features, labels, cv=3)

    assert result is base
    assert isinstance(result, SklearnStrategy)


def test_calibrate_best_respects_isotonic_method():
    features, labels = _dataset(8)
    base = RandomForestStrategy().fit(features, labels)

    result = calibrate_best(base, features, labels, method="isotonic", cv=3)
    assert isinstance(result, CalibratedStrategy)
    assert result.calibration_method == "isotonic"
