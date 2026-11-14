from __future__ import annotations

from prediction_service.classifier import DiseaseClassifier, RandomForestStrategy
from prediction_service.features import encode_entries
from prediction_service.models import PredictionResult, Severity, SymptomEntryDTO
from prediction_service.parser import SymptomParser
from prediction_service.pipeline import PredictionPipeline

from .conftest import make_entity, make_link


def _entry(symptom: str) -> SymptomEntryDTO:
    from uuid import uuid4

    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=symptom,
        severity=Severity.MODERATE,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=symptom,
    )


def _trained_strategy() -> RandomForestStrategy:
    flu = encode_entries([_entry("high_fever"), _entry("cough")])
    rash = encode_entries([_entry("itching"), _entry("skin_rash")])
    features = [flu, flu, rash, rash]
    labels = ["Flu", "Flu", "Rash", "Rash"]
    return RandomForestStrategy().fit(features, labels)


def _parser_for(symptom_text: str, canonical: str, weight: int) -> SymptomParser:
    """A parser whose stages are faked to emit a single linked symptom."""
    from .conftest import FakeExtractor, FakeLinker, FakeNegationDetector

    extractor = FakeExtractor([make_entity(symptom_text)])
    negation = FakeNegationDetector()
    linker = FakeLinker(
        {symptom_text: make_link(symptom_text, canonical, severity_weight=weight)}
    )
    return SymptomParser(extractor, negation, linker)


def test_pipeline_routes_text_through_to_predictions():
    parser = _parser_for("high fever", "high_fever", weight=7)
    pipeline = PredictionPipeline(_trained_strategy(), parser)

    result = pipeline.predict("I have a high fever")

    assert isinstance(result, PredictionResult)
    assert result.predictions[0].disease_name == "Flu"
    assert result.predictions[0].rank == 1
    assert result.disclaimer == "This is not a medical diagnosis."


def test_pipeline_accepts_bare_strategy_and_wraps_it():
    parser = _parser_for("itching", "itching", weight=1)
    # pass the strategy directly, not a DiseaseClassifier
    pipeline = PredictionPipeline(_trained_strategy(), parser)

    result = pipeline.predict("my skin is itching")
    assert result.predictions[0].disease_name == "Rash"


def test_pipeline_accepts_disease_classifier():
    parser = _parser_for("high fever", "high_fever", weight=7)
    classifier = DiseaseClassifier(_trained_strategy())
    pipeline = PredictionPipeline(classifier, parser)

    result = pipeline.predict("fever")
    assert result.predictions[0].disease_name == "Flu"


def test_pipeline_parse_exposes_nlp_stage():
    parser = _parser_for("high fever", "high_fever", weight=7)
    pipeline = PredictionPipeline(_trained_strategy(), parser)

    parse_result = pipeline.parse("I have a high fever")
    assert parse_result.active_symptoms() == ["high_fever"]
