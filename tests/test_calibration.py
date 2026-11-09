from __future__ import annotations

from uuid import uuid4

import pytest

from prediction_service.classifier import (
    CalibratedStrategy,
    DiseaseClassifier,
    RandomForestStrategy,
    SklearnStrategy,
)
from prediction_service.features import encode_entries
from prediction_service.models import Severity, SymptomEntryDTO


def _entry(symptom: str) -> SymptomEntryDTO:
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=symptom,
        severity=Severity.MODERATE,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=symptom,
    )


def _training_set(copies: int = 6):
    """Two separable diseases, repeated so CalibratedClassifierCV has folds."""
    flu = encode_entries([_entry("high_fever"), _entry("cough")])
    rash = encode_entries([_entry("itching"), _entry("skin_rash")])
    features = [flu] * copies + [rash] * copies
    labels = ["Flu"] * copies + ["Rash"] * copies
    return features, labels


def test_calibrated_strategy_predicts_correct_disease():
    features, labels = _training_set()
    strategy = CalibratedStrategy(RandomForestStrategy(), cv=3).fit(features, labels)

    predictions = strategy.classify([_entry("high_fever"), _entry("cough")])

    assert predictions[0].disease_name == "Flu"
    assert 0.0 <= predictions[0].probability <= 1.0


def test_calibrated_probabilities_sum_to_about_one():
    features, labels = _training_set()
    strategy = CalibratedStrategy(RandomForestStrategy(), cv=3).fit(features, labels)

    predictions = strategy.classify([_entry("itching")])
    total = sum(p.probability for p in predictions)
    assert total == pytest.approx(1.0, abs=1e-6)


def test_calibrated_is_a_strategy_and_drops_into_classifier():
    features, labels = _training_set()
    strategy = CalibratedStrategy(RandomForestStrategy(), cv=3).fit(features, labels)
    assert isinstance(strategy, SklearnStrategy)

    classifier = DiseaseClassifier(strategy)
    assert (
        classifier.predict([_entry("itching"), _entry("skin_rash")])[0].disease_name
        == "Rash"
    )


def test_calibration_method_and_algorithm_exposed():
    strategy = CalibratedStrategy(RandomForestStrategy(), method="isotonic")
    assert strategy.calibration_method == "isotonic"
    assert "isotonic" in strategy.algorithm
    assert "RandomForest" in strategy.algorithm


def test_default_method_is_sigmoid():
    strategy = CalibratedStrategy(RandomForestStrategy())
    assert strategy.calibration_method == "sigmoid"


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="sigmoid.*isotonic"):
        CalibratedStrategy(RandomForestStrategy(), method="bogus")


def test_calibrate_before_fit_raises():
    strategy = CalibratedStrategy(RandomForestStrategy())
    with pytest.raises(RuntimeError, match="must be trained"):
        strategy.classify([_entry("itching")])


def test_calibrated_inherits_base_top_k():
    base = RandomForestStrategy(top_k=1)
    strategy = CalibratedStrategy(base, cv=3)
    features, labels = _training_set()
    strategy.fit(features, labels)
    assert len(strategy.classify([_entry("itching")])) == 1
