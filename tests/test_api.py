from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from fastapi.testclient import TestClient

from prediction_service.api import (
    app,
    get_associations,
    get_follow_up_service,
    get_knowledge,
    get_pipeline,
)
from prediction_service.associations import SymptomAssociationIndex
from prediction_service.classifier import RandomForestStrategy
from prediction_service.features import encode_entries
from prediction_service.followup import FollowUpService
from prediction_service.knowledge import DiseaseKnowledgeBase
from prediction_service.models import DiseasePrediction, Severity, SymptomEntryDTO
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
    # Keep this schema-validation test independent of the trained model artifact.
    _override(_empty_pipeline())
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


# ---------------------------------------------------------------------------
# Adaptive follow-up questions (/predict low-confidence hint + /predict/followup)
# ---------------------------------------------------------------------------


@dataclass
class _ScriptedClassifier:
    by_entry_count: dict[int, list[tuple[str, float]]]
    default: list[tuple[str, float]] = field(default_factory=list)

    def predict(self, entries) -> list[DiseasePrediction]:
        ranking = self.by_entry_count.get(len(entries), self.default)
        return [
            DiseasePrediction(rank=i, probability=p, disease_name=name)
            for i, (name, p) in enumerate(ranking, start=1)
        ]


def _followup_index() -> SymptomAssociationIndex:
    return SymptomAssociationIndex(
        {
            "flu": {"cough": 1.0, "high_fever": 1.0, "chills": 0.8},
            "common cold": {"cough": 1.0, "continuous_sneezing": 0.6},
        }
    )


def _override_followup(service: FollowUpService) -> None:
    app.dependency_overrides[get_follow_up_service] = lambda: service
    # Keep get_associations harmless if anything else resolves it.
    app.dependency_overrides[get_associations] = _followup_index


def test_predict_attaches_follow_up_when_confidence_low(client):
    # Pipeline parses one real symptom ("cough"); the scripted follow-up service
    # makes the top disease (Flu) land below the 0.80 threshold.
    _override(_fake_pipeline("cough", "cough", weight=4))
    service = FollowUpService(
        _ScriptedClassifier({1: [("Flu", 0.55), ("Common Cold", 0.40)]}),
        _followup_index(),
    )
    _override_followup(service)

    response = client.post("/predict", json={"text": "I have a cough"})

    assert response.status_code == 200
    body = response.json()
    assert body["needs_follow_up"] is True
    assert body["follow_up"]["symptom"] == "high_fever"
    assert body["follow_up"]["question"] == "Do you also have high fever?"
    assert body["follow_up"]["top_disease"] == "Flu"


def test_predict_omits_follow_up_when_confident(client):
    _override(_fake_pipeline("cough", "cough", weight=4))
    service = FollowUpService(
        _ScriptedClassifier({1: [("Flu", 0.95), ("Common Cold", 0.02)]}),
        _followup_index(),
    )
    _override_followup(service)

    response = client.post("/predict", json={"text": "I have a cough"})

    body = response.json()
    assert body["needs_follow_up"] is False
    assert body["follow_up"] is None


def test_followup_yes_answer_raises_confidence(client):
    # 1 confirmed -> vague; 2 confirmed (after yes) -> confident.
    service = FollowUpService(
        _ScriptedClassifier(
            {
                1: [("Flu", 0.55), ("Common Cold", 0.40)],
                2: [("Flu", 0.90), ("Common Cold", 0.05)],
            }
        ),
        _followup_index(),
    )
    _override_followup(service)

    response = client.post(
        "/predict/followup",
        json={
            "confirmed_symptoms": ["cough"],
            "denied_symptoms": [],
            "answer": {"symptom": "high_fever", "present": True},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["needs_follow_up"] is False
    assert body["follow_up"] is None
    assert body["predictions"][0]["disease_name"] == "Flu"
    assert "high_fever" in body["active_symptoms"]


def test_followup_no_answer_asks_a_different_symptom(client):
    service = FollowUpService(
        _ScriptedClassifier({1: [("Flu", 0.55), ("Common Cold", 0.40)]}),
        _followup_index(),
    )
    _override_followup(service)

    response = client.post(
        "/predict/followup",
        json={
            "confirmed_symptoms": ["cough"],
            "denied_symptoms": [],
            "answer": {"symptom": "high_fever", "present": False},
        },
    )

    body = response.json()
    assert body["needs_follow_up"] is True
    # high_fever denied this turn -> next best is chills.
    assert body["follow_up"]["symptom"] == "chills"


def test_followup_requires_confirmed_symptoms(client):
    service = FollowUpService(
        _ScriptedClassifier({1: [("Flu", 0.55)]}),
        _followup_index(),
    )
    _override_followup(service)

    response = client.post(
        "/predict/followup",
        json={"confirmed_symptoms": [], "denied_symptoms": []},
    )

    assert response.status_code == 422
    assert "No confirmed symptoms" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Red flags on /predict and /predict/followup
# ---------------------------------------------------------------------------


def _pipeline_for(symptoms: dict[str, str]) -> PredictionPipeline:
    """A pipeline whose parser links each phrase to the given canonical symptom."""
    extractor = FakeExtractor([make_entity(text) for text in symptoms])
    linker = FakeLinker(
        {
            text: make_link(text, canonical, severity_weight=4)
            for text, canonical in symptoms.items()
        }
    )
    parser = SymptomParser(extractor, FakeNegationDetector(), linker)
    return PredictionPipeline(_trained_strategy(), parser)


def test_predict_flags_an_emergency_symptom_set(client):
    _override(
        _pipeline_for({"chest pain": "chest_pain", "short of breath": "breathlessness"})
    )

    response = client.post(
        "/predict", json={"text": "chest pain and I am short of breath"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["highest_urgency"] == "EMERGENCY"
    assert body["red_flags"][0]["name"] == "possible_cardiac_event"
    assert "112" in body["red_flags"][0]["message"]
    assert body["red_flags"][0]["matched_symptoms"] == ["breathlessness", "chest_pain"]
    assert body["disclaimer"] == "This is not a medical diagnosis."


def test_predict_flags_a_high_fever_as_urgent(client):
    _override(_fake_pipeline("high fever", "high_fever", weight=7))

    response = client.post("/predict", json={"text": "I have a high fever"})

    body = response.json()
    assert body["highest_urgency"] == "URGENT"
    assert [flag["name"] for flag in body["red_flags"]] == ["high_fever"]


def test_predict_benign_input_has_no_urgency(client):
    _override(_fake_pipeline("itching", "itching", weight=1))

    response = client.post("/predict", json={"text": "it itches"})

    body = response.json()
    assert body["highest_urgency"] is None
    assert body["red_flags"] == []


def test_followup_flags_confirmed_emergency_symptoms(client):
    service = FollowUpService(
        _ScriptedClassifier({}, default=[("Heart attack", 0.9)]),
        _followup_index(),
    )
    _override_followup(service)
    _override_knowledge(DiseaseKnowledgeBase())

    response = client.post(
        "/predict/followup",
        json={
            "confirmed_symptoms": ["chest_pain"],
            "denied_symptoms": [],
            "rounds_used": 0,
            "answer": {"symptom": "sweating", "present": True},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["highest_urgency"] == "EMERGENCY"
    assert body["red_flags"][0]["matched_symptoms"] == ["chest_pain", "sweating"]


def test_build_associations_prefers_baked_json(tmp_path, monkeypatch):
    from prediction_service import api

    frequencies = {"Flu": {"high_fever": 1.0, "cough": 1.0}}
    artifact = tmp_path / "models" / "associations.json"
    SymptomAssociationIndex.from_frequencies(frequencies).save_json(artifact)

    monkeypatch.setenv("DISEASE_ASSOCIATIONS_PATH", str(artifact))
    monkeypatch.delenv("DISEASE_DATASET_PATH", raising=False)
    api._build_associations.cache_clear()
    try:
        index = api._build_associations()
    finally:
        api._build_associations.cache_clear()

    assert index.has("Flu")
    assert index.symptoms_for("Flu")["high_fever"] == 1.0


def test_build_associations_defaults_to_json_beside_model(tmp_path, monkeypatch):
    from prediction_service import api

    models = tmp_path / "models"
    SymptomAssociationIndex.from_frequencies({"Flu": {"high_fever": 1.0}}).save_json(
        models / "associations.json"
    )

    monkeypatch.delenv("DISEASE_ASSOCIATIONS_PATH", raising=False)
    monkeypatch.delenv("DISEASE_DATASET_PATH", raising=False)
    monkeypatch.setenv("DISEASE_MODEL_PATH", str(models / "best.joblib"))
    api._build_associations.cache_clear()
    try:
        index = api._build_associations()
    finally:
        api._build_associations.cache_clear()

    assert index.has("Flu")
