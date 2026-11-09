"""Load the Kaggle *Disease Symptom Description* dataset into a training matrix.

The raw ``dataset.csv`` has one disease per row plus up to 17 ``Symptom_N``
columns holding symptom names (padded with blanks). This module turns those
rows into the same severity-weighted feature vectors the parser emits, so a
classifier trained here consumes exactly what it will see at inference time.

The CSV is not bundled with the repo. Download it from Kaggle
(itachi9604/disease-symptom-description-dataset) and point :func:`load_dataset`
at ``dataset.csv``.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .severity import weight_for
from .vocabulary import CANONICAL_SYMPTOMS, normalise_symptom

_INDEX_BY_NORMALISED: dict[str, int] = {
    normalise_symptom(symptom): index
    for index, symptom in enumerate(CANONICAL_SYMPTOMS)
}


def _row_to_vector(symptoms: list[str]) -> list[int]:
    vector = [0] * len(CANONICAL_SYMPTOMS)
    for raw in symptoms:
        name = raw.strip()
        if not name:
            continue
        index = _INDEX_BY_NORMALISED.get(normalise_symptom(name))
        if index is None:
            continue
        vector[index] = weight_for(CANONICAL_SYMPTOMS[index])
    return vector


def load_dataset(csv_path: str | Path) -> tuple[list[list[int]], list[str]]:
    """Read ``dataset.csv`` and return ``(features, labels)``.

    ``features`` is a list of severity-weighted vectors (aligned to
    ``CANONICAL_SYMPTOMS``); ``labels`` is the matching list of disease names.
    """
    path = Path(csv_path)
    features: list[list[int]] = []
    labels: list[str] = []

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        symptom_columns = [c for c in (reader.fieldnames or []) if c != "Disease"]
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            if not disease:
                continue
            symptoms = [row.get(column) or "" for column in symptom_columns]
            features.append(_row_to_vector(symptoms))
            labels.append(disease)

    return features, labels
