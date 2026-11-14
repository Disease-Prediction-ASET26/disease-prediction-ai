from __future__ import annotations

from .models import Severity, SymptomEntryDTO
from .severity import weight_for
from .vocabulary import CANONICAL_SYMPTOMS

_SEVERITY_FALLBACK: dict[Severity, int] = {
    Severity.MILD: 2,
    Severity.MODERATE: 4,
    Severity.SEVERE: 6,
}

_INDEX_BY_SYMPTOM: dict[str, int] = {
    symptom: index for index, symptom in enumerate(CANONICAL_SYMPTOMS)
}


def _weight_for_entry(entry: SymptomEntryDTO) -> int:
    if entry.canonical_symptom is not None:
        weight = weight_for(entry.canonical_symptom)
        if weight:
            return weight
    return _SEVERITY_FALLBACK.get(entry.severity, 0)


def encode_entries(
    entries: list[SymptomEntryDTO],
    vocabulary: tuple[str, ...] = CANONICAL_SYMPTOMS,
) -> list[int]:
    """Turn symptom entries into a severity-weighted feature vector."""
    index_by_symptom = (
        _INDEX_BY_SYMPTOM
        if vocabulary is CANONICAL_SYMPTOMS
        else {symptom: i for i, symptom in enumerate(vocabulary)}
    )
    vector = [0] * len(vocabulary)
    for entry in entries:
        symptom = entry.canonical_symptom
        if symptom is None:
            continue
        index = index_by_symptom.get(symptom)
        if index is None:
            continue
        vector[index] = _weight_for_entry(entry)
    return vector
