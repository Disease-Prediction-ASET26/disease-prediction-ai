"""FastAPI service exposing the disease-prediction pipeline over HTTP.

Routes free-text symptom descriptions through the NLP parser and the trained
classifier, and enriches each predicted disease with a human-readable
description and recommended precautions::

    POST /predict   {"text": "I have a high fever and a cough"}
        -> ranked diseases + description + precautions

Run it (from the project root, after training a model)::

    python -m prediction_service.train --data data/dataset.csv --out models
    uvicorn prediction_service.api:app --reload

The model path can be overridden with the ``DISEASE_MODEL_PATH`` env var; the
description/precaution CSVs with ``DISEASE_DESCRIPTION_PATH`` and
``DISEASE_PRECAUTION_PATH``.
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .knowledge import (
    DEFAULT_DESCRIPTION_PATH,
    DEFAULT_PRECAUTION_PATH,
    DiseaseKnowledgeBase,
)
from .persistence import DEFAULT_MODEL_PATH, load_strategy
from .pipeline import PredictionPipeline
from .red_flags import RedFlagEngine, Urgency
from .specialty import SpecialtyMapper

app = FastAPI(
    title="Disease Prediction AI",
    description="Predict likely diseases from a free-text symptom description.",
    version="1.0.0",
)


class PredictRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the patient's symptoms.",
        examples=["I have a high fever and a cough, but no headache."],
    )
    top_k: int | None = Field(
        default=None,
        ge=1,
        le=41,
        description="How many ranked diseases to return (defaults to the model's).",
    )


class ExplanationItem(BaseModel):
    summary: str
    contribution_weights: dict[str, float] = Field(default_factory=dict)


class PredictionItem(BaseModel):
    rank: int
    disease_name: str
    probability: float
    specialty: str
    description: str | None = None
    precautions: list[str] = Field(default_factory=list)
    explanation: ExplanationItem | None = None


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    active_symptoms: list[str]
    disclaimer: str


class DiseaseInfoResponse(BaseModel):
    disease_name: str
    description: str | None = None
    precautions: list[str] = Field(default_factory=list)


class MapSymptomsRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        description="Free-text description of the patient's symptoms.",
        examples=["I have a high fever and a cough, but no headache."],
    )


class LinkedSymptomItem(BaseModel):
    source_text: str
    canonical_symptom: str
    similarity: float
    method: str
    severity_weight: int


class MapSymptomsResponse(BaseModel):
    active_symptoms: list[str]
    linked: list[LinkedSymptomItem]
    negated: list[str]


class RedFlagCheckRequest(BaseModel):
    text: str | None = Field(
        default=None,
        description="Free text to parse for symptoms (if symptoms not given).",
    )
    symptoms: list[str] | None = Field(
        default=None,
        description="Explicit canonical symptom names (skips parsing).",
    )


class RedFlagItem(BaseModel):
    name: str
    urgency: Urgency
    message: str
    matched_symptoms: list[str]


class RedFlagCheckResponse(BaseModel):
    active_symptoms: list[str]
    red_flags: list[RedFlagItem]
    highest_urgency: Urgency | None = None


@lru_cache(maxsize=1)
def _build_pipeline() -> PredictionPipeline:
    """Build the pipeline once (model + NLP parser) and cache it.

    Explanations are enabled on the loaded strategy so every prediction carries
    the symptoms that drove it.
    """
    model_path = os.environ.get("DISEASE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    strategy = load_strategy(model_path)
    strategy.with_explanations()
    return PredictionPipeline(strategy)


@lru_cache(maxsize=1)
def _build_knowledge() -> DiseaseKnowledgeBase:
    """Load description/precaution CSVs once (empty KB if files are absent)."""
    description_path = os.environ.get(
        "DISEASE_DESCRIPTION_PATH", str(DEFAULT_DESCRIPTION_PATH)
    )
    precaution_path = os.environ.get(
        "DISEASE_PRECAUTION_PATH", str(DEFAULT_PRECAUTION_PATH)
    )
    return DiseaseKnowledgeBase.from_csv(description_path, precaution_path)


def get_pipeline() -> PredictionPipeline:
    """FastAPI dependency: return the cached pipeline or a 503 if untrained."""
    try:
        return _build_pipeline()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


def get_knowledge() -> DiseaseKnowledgeBase:
    """FastAPI dependency: return the cached disease knowledge base."""
    return _build_knowledge()


@lru_cache(maxsize=1)
def _build_specialty_mapper() -> SpecialtyMapper:
    """Build the disease -> specialty mapper once and cache it."""
    return SpecialtyMapper()


def get_specialty_mapper() -> SpecialtyMapper:
    """FastAPI dependency: return the cached specialty mapper."""
    return _build_specialty_mapper()


@lru_cache(maxsize=1)
def _build_red_flag_engine() -> RedFlagEngine:
    """Build the red-flag rule engine once and cache it."""
    return RedFlagEngine()


def get_red_flag_engine() -> RedFlagEngine:
    """FastAPI dependency: return the cached red-flag engine."""
    return _build_red_flag_engine()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready(
    pipeline: PredictionPipeline = Depends(get_pipeline),  # noqa: B008
) -> dict[str, str]:
    """Warm the NLP models and report ready only when predictions can run."""
    try:
        pipeline.warm_up()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail="Prediction models are not ready.",
        ) from exc
    return {"status": "ready"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    pipeline: PredictionPipeline = Depends(get_pipeline),  # noqa: B008
    knowledge: DiseaseKnowledgeBase = Depends(get_knowledge),  # noqa: B008
    specialty: SpecialtyMapper = Depends(get_specialty_mapper),  # noqa: B008
) -> PredictResponse:
    parse_result, predictions = pipeline.predict_detailed(request.text)

    if not parse_result.active_symptoms():
        raise HTTPException(
            status_code=422,
            detail="No known symptoms could be extracted from the input text.",
        )

    if request.top_k is not None:
        predictions = predictions[: request.top_k]

    return PredictResponse(
        predictions=[
            PredictionItem(
                rank=p.rank,
                disease_name=p.disease_name,
                probability=p.probability,
                specialty=specialty.specialty_for(p.disease_name),
                description=knowledge.describe(p.disease_name),
                precautions=knowledge.precautions_for(p.disease_name),
                explanation=(
                    ExplanationItem(
                        summary=p.explanation.summary,
                        contribution_weights=p.explanation.contribution_weights,
                    )
                    if p.explanation is not None
                    else None
                ),
            )
            for p in predictions
        ],
        active_symptoms=parse_result.active_symptoms(),
        disclaimer="This is not a medical diagnosis.",
    )


@app.post("/map-symptoms", response_model=MapSymptomsResponse)
def map_symptoms(
    request: MapSymptomsRequest,
    pipeline: PredictionPipeline = Depends(get_pipeline),  # noqa: B008
) -> MapSymptomsResponse:
    """Run only the NLP stage: map free text to canonical symptoms.

    Wraps the existing parser so callers can inspect what symptoms were
    extracted (and which were negated) without running the classifier.
    """
    parse_result = pipeline.parse(request.text)

    if not parse_result.active_symptoms():
        raise HTTPException(
            status_code=422,
            detail="No known symptoms could be extracted from the input text.",
        )

    return MapSymptomsResponse(
        active_symptoms=parse_result.active_symptoms(),
        linked=[
            LinkedSymptomItem(
                source_text=link.source_text,
                canonical_symptom=link.canonical_symptom,
                similarity=link.similarity,
                method=link.method,
                severity_weight=link.severity_weight,
            )
            for link in parse_result.linked
        ],
        negated=[entity.text for entity in parse_result.negated],
    )


@app.post("/red-flags/check", response_model=RedFlagCheckResponse)
def red_flags_check(
    request: RedFlagCheckRequest,
    pipeline: PredictionPipeline = Depends(get_pipeline),  # noqa: B008
    engine: RedFlagEngine = Depends(get_red_flag_engine),  # noqa: B008
) -> RedFlagCheckResponse:
    """Check reported symptoms against the red-flag rule engine.

    Accepts either an explicit list of canonical ``symptoms`` or free ``text``
    to parse. Returns every rule that fired and the highest urgency among them.
    """
    if request.symptoms is not None:
        active = request.symptoms
    elif request.text:
        active = pipeline.parse(request.text).active_symptoms()
    else:
        raise HTTPException(
            status_code=422,
            detail="Provide either 'symptoms' or 'text'.",
        )

    fired = engine.check(active)
    return RedFlagCheckResponse(
        active_symptoms=active,
        red_flags=[
            RedFlagItem(
                name=flag.name,
                urgency=flag.urgency,
                message=flag.message,
                matched_symptoms=flag.matched_symptoms,
            )
            for flag in fired
        ],
        highest_urgency=fired[0].urgency if fired else None,
    )


@app.get("/diseases/{name}", response_model=DiseaseInfoResponse)
def disease_info(
    name: str,
    knowledge: DiseaseKnowledgeBase = Depends(get_knowledge),  # noqa: B008
) -> DiseaseInfoResponse:
    """Fetch the description and precautions for a single disease by name.

    The name is matched case-insensitively and ignoring extra whitespace.
    """
    if not knowledge.has(name):
        raise HTTPException(status_code=404, detail=f"Unknown disease: {name!r}")

    info = knowledge.info_for(name)
    return DiseaseInfoResponse(
        disease_name=info.disease_name,
        description=info.description,
        precautions=info.precautions,
    )
