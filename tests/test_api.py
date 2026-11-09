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
    # Explanations are enabled on the served model, so mirror that here.
    return RandomForestStrategy().fit(features, labels).with_explanations()


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


def test_predict_includes_specialty(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/predict", json={"text": "I have a high fever"})

    top = response.json()["predictions"][0]
    # "Flu" is not in the Kaggle table, so it falls back to the default.
    assert top["specialty"] == "General Medicine"


def test_predict_includes_explanation(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/predict", json={"text": "I have a high fever"})

    top = response.json()["predictions"][0]
    assert top["explanation"] is not None
    assert "summary" in top["explanation"]
    assert top["explanation"]["contribution_weights"]


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
    _override(_empty_pipeline())
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422  # pydantic min_length validation


def test_predict_from_checklist_symptoms_skips_nlp(client):
    # The parser would find nothing; the checklist symptoms alone drive it.
    _override(_empty_pipeline())

    response = client.post(
        "/predict", json={"symptoms": ["high_fever", "cough"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["predictions"][0]["disease_name"] == "Flu"
    assert body["active_symptoms"] == ["high_fever", "cough"]


def test_predict_combines_text_and_checklist_symptoms(client):
    _override(_fake_pipeline("itching", "itching", weight=1))

    response = client.post(
        "/predict",
        json={"text": "it itches", "symptoms": ["Skin Rash", "itching"]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["active_symptoms"] == ["itching", "skin_rash"]
    assert body["predictions"][0]["disease_name"] == "Rash"


def test_predict_unknown_checklist_symptoms_returns_422(client):
    _override(_empty_pipeline())

    response = client.post("/predict", json={"symptoms": ["not_a_symptom"]})

    assert response.status_code == 422
    assert "No known symptoms" in response.json()["detail"]


def test_predict_requires_text_or_symptoms(client):
    _override(_empty_pipeline())
    response = client.post("/predict", json={})
    assert response.status_code == 422


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


# ---------------------------------------------------------------------------
# /map-symptoms
# ---------------------------------------------------------------------------


def test_map_symptoms_returns_linked(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/map-symptoms", json={"text": "I have a high fever"})

    assert response.status_code == 200
    body = response.json()
    assert body["active_symptoms"] == ["high_fever"]
    assert body["linked"][0]["canonical_symptom"] == "high_fever"
    assert body["linked"][0]["source_text"] == "high fever"


def test_map_symptoms_no_symptoms_returns_422(client):
    _override(_empty_pipeline())

    response = client.post("/map-symptoms", json={"text": "hello there"})

    assert response.status_code == 422
    assert "No known symptoms" in response.json()["detail"]


# ---------------------------------------------------------------------------
# /red-flags/check
# ---------------------------------------------------------------------------


def test_red_flags_check_with_explicit_symptoms(client):
    _override(_empty_pipeline())  # pipeline unused when symptoms given

    response = client.post(
        "/red-flags/check", json={"symptoms": ["chest_pain", "breathlessness"]}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["highest_urgency"] == "EMERGENCY"
    names = {f["name"] for f in body["red_flags"]}
    assert "possible_cardiac_event" in names


def test_red_flags_check_from_text(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/red-flags/check", json={"text": "I have a high fever"})

    assert response.status_code == 200
    body = response.json()
    assert body["active_symptoms"] == ["high_fever"]
    assert body["highest_urgency"] == "URGENT"


def test_red_flags_check_no_flags(client):
    _override(_empty_pipeline())

    response = client.post("/red-flags/check", json={"symptoms": ["itching"]})

    assert response.status_code == 200
    body = response.json()
    assert body["red_flags"] == []
    assert body["highest_urgency"] is None


def test_red_flags_check_requires_input(client):
    _override(_empty_pipeline())

    response = client.post("/red-flags/check", json={})

    assert response.status_code == 422
    assert "Provide either" in response.json()["detail"]
