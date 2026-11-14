"""Disease descriptions and precautions from the Kaggle dataset.

Loads ``symptom_Description.csv`` (``Disease,Description``) and
``symptom_precaution.csv`` (``Disease,Precaution_1..4``) into a lookup keyed by
disease name. Disease names are matched on a whitespace-normalised, lowercased
key so minor spelling/spacing differences between the CSVs and the classifier
labels still resolve.

The CSVs are optional: if a file is missing the corresponding lookup is simply
empty, and callers get ``None`` / ``[]`` back.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DESCRIPTION_PATH = Path("data/symptom_Description.csv")
DEFAULT_PRECAUTION_PATH = Path("data/symptom_precaution.csv")


def _key(disease: str) -> str:
    return " ".join(disease.split()).strip().lower()


@dataclass
class DiseaseInfo:
    disease_name: str
    description: str | None = None
    precautions: list[str] = field(default_factory=list)


class DiseaseKnowledgeBase:
    """Lookup of human-readable description + precautions per disease."""

    def __init__(
        self,
        descriptions: dict[str, str] | None = None,
        precautions: dict[str, list[str]] | None = None,
    ) -> None:
        self._descriptions = descriptions or {}
        self._precautions = precautions or {}

    @classmethod
    def from_csv(
        cls,
        description_path: str | Path = DEFAULT_DESCRIPTION_PATH,
        precaution_path: str | Path = DEFAULT_PRECAUTION_PATH,
    ) -> DiseaseKnowledgeBase:
        return cls(
            descriptions=_load_descriptions(description_path),
            precautions=_load_precautions(precaution_path),
        )

    def describe(self, disease_name: str) -> str | None:
        return self._descriptions.get(_key(disease_name))

    def precautions_for(self, disease_name: str) -> list[str]:
        return self._precautions.get(_key(disease_name), [])

    def has(self, disease_name: str) -> bool:
        """True if the disease appears in either the description or precautions."""
        key = _key(disease_name)
        return key in self._descriptions or key in self._precautions

    def info_for(self, disease_name: str) -> DiseaseInfo:
        return DiseaseInfo(
            disease_name=disease_name,
            description=self.describe(disease_name),
            precautions=self.precautions_for(disease_name),
        )


def _load_descriptions(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.exists():
        return {}
    out: dict[str, str] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            description = (row.get("Description") or "").strip()
            if disease:
                out[_key(disease)] = description
    return out


def _load_precautions(path: str | Path) -> dict[str, list[str]]:
    source = Path(path)
    if not source.exists():
        return {}
    out: dict[str, list[str]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        precaution_columns = [
            c for c in (reader.fieldnames or []) if c and c != "Disease"
        ]
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            if not disease:
                continue
            steps = [
                (row.get(column) or "").strip()
                for column in precaution_columns
                if (row.get(column) or "").strip()
            ]
            out[_key(disease)] = steps
    return out
