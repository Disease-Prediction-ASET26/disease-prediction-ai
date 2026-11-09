from __future__ import annotations

import typing
from uuid import uuid4

import pytest

from prediction_service.classifier import DecisionTreeStrategy, RandomForestStrategy
from prediction_service.explanation import ExplanationGenerator
from prediction_service.features import encode_entries
from prediction_service.models import Explanation, Severity, SymptomEntryDTO
from prediction_service.vocabulary import CANONICAL_SYMPTOMS


def _entry(symptom: str) -> SymptomEntryDTO:
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=symptom,
        severity=Severity.MODERATE,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=symptom,
    )


def _trained():
    flu = encode_entries([_entry("high_fever"), _entry("cough")])
    rash = encode_entries([_entry("itching"), _entry("skin_rash")])
    return RandomForestStrategy().fit(
        [flu, flu, rash, rash], ["Flu", "Flu", "Rash", "Rash"]
    )


# ---------------------------------------------------------------------------
# ExplanationGenerator directly
# ---------------------------------------------------------------------------


def test_explanation_uses_feature_importances_of_active_symptoms():
    strategy = _trained()
    gen = ExplanationGenerator()
    vector = encode_entries([_entry("high_fever"), _entry("cough")])

    explanation = gen.explain(strategy._estimator, vector, "Flu")

    assert isinstance(explanation, Explanation)
    assert explanation.contribution_weights  # non-empty
    # only active symptoms may appear
    for symptom in explanation.contribution_weights:
        assert symptom in {"high_fever", "cough"}


def test_contribution_weights_are_normalised():
    strategy = _trained()
    vector = encode_entries([_entry("itching"), _entry("skin_rash")])

    explanation = ExplanationGenerator().explain(strategy._estimator, vector, "Rash")
    assert sum(explanation.contribution_weights.values()) == pytest.approx(
        1.0, abs=1e-3
    )


def test_summary_mentions_disease():
    strategy = _trained()
    vector = encode_entries([_entry("high_fever")])
    explanation = ExplanationGenerator().explain(strategy._estimator, vector, "Flu")
    assert "Flu" in explanation.summary


def test_no_active_symptoms_gives_empty_explanation():
    strategy = _trained()
    empty = [0] * len(CANONICAL_SYMPTOMS)
    explanation = ExplanationGenerator().explain(strategy._estimator, empty, "Flu")
    assert explanation.contribution_weights == {}
    assert "No symptoms" in explanation.summary


def test_fallback_splits_evenly_without_importances():
    class DummyNoImportances:
        classes_: typing.ClassVar = ["Flu", "Rash"]

    vector = encode_entries([_entry("high_fever"), _entry("cough")])
    explanation = ExplanationGenerator().explain(DummyNoImportances(), vector, "Flu")

    weights = explanation.contribution_weights
    assert set(weights) == {"high_fever", "cough"}
    assert all(w == pytest.approx(0.5) for w in weights.values())


# ---------------------------------------------------------------------------
# wired into the strategy
# ---------------------------------------------------------------------------


def test_strategy_without_explanations_has_none():
    strategy = _trained()
    predictions = strategy.classify([_entry("high_fever"), _entry("cough")])
    assert predictions[0].explanation is None


def test_strategy_with_explanations_attaches_them():
    strategy = _trained().with_explanations()
    predictions = strategy.classify([_entry("high_fever"), _entry("cough")])

    top = predictions[0]
    assert top.disease_name == "Flu"
    assert top.explanation is not None
    assert top.explanation.contribution_weights


def test_with_explanations_is_chainable():
    strategy = DecisionTreeStrategy()
    assert strategy.with_explanations() is strategy
