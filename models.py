from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import UUID


class Severity(Enum):
    MILD = "MILD"
    MODERATE = "MODERATE"
    SEVERE = "SEVERE"


@dataclass
class SymptomEntryDTO:
    entry_id: UUID
    raw_text: str | None
    severity: Severity
    duration_days: int
    is_mapped: bool


@dataclass
class Explanation:
    summary: str
    contribution_weights: dict[str, float] = field(default_factory=dict)


@dataclass
class DiseasePrediction:
    rank: int
    probability: float
    disease_name: str
    explanation: Explanation | None = None


@dataclass
class PredictionResult:
    result_id: UUID
    generated_at: datetime
    predictions: list[DiseasePrediction] = field(default_factory=list)
    disclaimer: str = "This is not a medical diagnosis."


@dataclass
class MLModelMetadata:
    model_id: UUID
    version: str
    algorithm: str
    calibration_method: str
    trained_at: datetime
    is_active: bool
