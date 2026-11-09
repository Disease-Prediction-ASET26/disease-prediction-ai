from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from prediction_service.api import app, get_knowledge, get_pipeline
from prediction_service.classifier import RandomForestStrategy
from prediction_service.features import encode_entries
from prediction_service.knowledge import DiseaseKnowledgeBase
from prediction_service.models import Severity, SymptomEntryDTO
from prediction_service.parser import SymptomParser
from prediction_service.pipeline import PredictionPipeline

from .conftest import (
    FakeExtractor,
    FakeLinker,
    FakeNegationDetector,
    make_entity,
    make_link,
)


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


def _fake_pipeline(
    symptom_text: str, canonical: str, weight: int
) -> PredictionPipeline:
    extractor = FakeExtractor([make_entity(symptom_text)])
    linker = FakeLinker(
        {symptom_text: make_link(symptom_text, canonical, severity_weight=weight)}
    )
    parser = SymptomParser(extractor, FakeNegationDetector(), linker)
    return PredictionPipeline(_trained_strategy(), parser)


def _empty_pipeline() -> PredictionPipeline:
    parser = SymptomParser(FakeExtractor([]), FakeNegationDetector(), FakeLinker({}))
    return PredictionPipeline(_trained_strategy(), parser)


@pytest.fixture
def client():
    c = TestClient(app)
    yield c
    app.dependency_overrides.clear()


def _override(
    pipeline: PredictionPipeline, knowledge: DiseaseKnowledgeBase | None = None
) -> None:
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    app.dependency_overrides[get_knowledge] = lambda: (
        knowledge or DiseaseKnowledgeBase()
    )


def _override_knowledge(knowledge: DiseaseKnowledgeBase) -> None:
    app.dependency_overrides[get_knowledge] = lambda: knowledge


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_ranked_diseases(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/predict", json={"text": "I have a high fever"})

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"][0]["disease_name"] == "Flu"
    assert body["predictions"][0]["rank"] == 1
    assert body["active_symptoms"] == ["high_fever"]
    assert body["disclaimer"] == "This is not a medical diagnosis."


def test_predict_enriches_with_description_and_precautions(client):
    knowledge = DiseaseKnowledgeBase(
        descriptions={"flu": "A viral infection."},
        precautions={"flu": ["rest", "drink fluids"]},
    )
    _override(_fake_pipeline("high fever", "high_fever", weight=7), knowledge)

    response = client.post("/predict", json={"text": "I have a high fever"})

    assert response.status_code == 200
    top = response.json()["predictions"][0]
    assert top["disease_name"] == "Flu"
    assert top["description"] == "A viral infection."
    assert top["precautions"] == ["rest", "drink fluids"]


def test_predict_without_knowledge_returns_null_description(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/predict", json={"text": "I have a high fever"})

    top = response.json()["predictions"][0]
    assert top["description"] is None
    assert top["precautions"] == []


def test_predict_respects_top_k(client):
    _override(_fake_pipeline("itching", "itching", weight=1))

    response = client.post("/predict", json={"text": "itchy skin", "top_k": 1})

    assert response.status_code == 200
    assert len(response.json()["predictions"]) == 1


def test_predict_no_symptoms_returns_422(client):
    _override(_empty_pipeline())

    response = client.post("/predict", json={"text": "hello there"})

    assert response.status_code == 422
    assert "No known symptoms" in response.json()["detail"]


def test_predict_rejects_empty_text(client):
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_disease_info_returns_description_and_precautions(client):
    _override_knowledge(
        DiseaseKnowledgeBase(
            descriptions={"malaria": "A mosquito-borne disease."},
            precautions={"malaria": ["consult hospital", "avoid oily food"]},
        )
    )

    response = client.get("/diseases/Malaria")

    assert response.status_code == 200
    body = response.json()
    assert body["disease_name"] == "Malaria"
    assert body["description"] == "A mosquito-borne disease."
    assert body["precautions"] == ["consult hospital", "avoid oily food"]


def test_disease_info_matches_name_case_insensitively(client):
    _override_knowledge(
        DiseaseKnowledgeBase(descriptions={"fungal infection": "A fungal disease."})
    )

    response = client.get("/diseases/fungal infection")

    assert response.status_code == 200
    assert response.json()["description"] == "A fungal disease."
    assert response.json()["precautions"] == []


def test_disease_info_unknown_returns_404(client):
    _override_knowledge(DiseaseKnowledgeBase(descriptions={"malaria": "x"}))

    response = client.get("/diseases/NotADisease")

    assert response.status_code == 404
    assert "Unknown disease" in response.json()["detail"]
