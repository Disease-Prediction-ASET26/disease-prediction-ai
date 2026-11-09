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


class PredictionItem(BaseModel):
    rank: int
    disease_name: str
    probability: float
    description: str | None = None
    precautions: list[str] = Field(default_factory=list)


class PredictResponse(BaseModel):
    predictions: list[PredictionItem]
    active_symptoms: list[str]
    disclaimer: str


class DiseaseInfoResponse(BaseModel):
    disease_name: str
    description: str | None = None
    precautions: list[str] = Field(default_factory=list)


@lru_cache(maxsize=1)
def _build_pipeline() -> PredictionPipeline:
    """Build the pipeline once (model + NLP parser) and cache it."""
    model_path = os.environ.get("DISEASE_MODEL_PATH", str(DEFAULT_MODEL_PATH))
    strategy = load_strategy(model_path)
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


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
def predict(
    request: PredictRequest,
    pipeline: PredictionPipeline = Depends(get_pipeline),  # noqa: B008
    knowledge: DiseaseKnowledgeBase = Depends(get_knowledge),  # noqa: B008
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
                description=knowledge.describe(p.disease_name),
                precautions=knowledge.precautions_for(p.disease_name),
            )
            for p in predictions
        ],
        active_symptoms=parse_result.active_symptoms(),
        disclaimer="This is not a medical diagnosis.",
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
