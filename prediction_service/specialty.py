"""Map a predicted disease to the medical specialty that treats it (FR-P4).

:class:`SpecialtyMapper` turns a disease label from the classifier into the
clinical specialty a patient would typically be referred to (e.g. *Malaria ->
Infectious Disease*, *Hypertension -> Cardiology*). The built-in table covers
the 41 diseases in the Kaggle dataset; unknown names fall back to
``General Medicine``.

Names are matched on the same whitespace-normalised, lowercased key the
knowledge base uses, so minor spacing/case differences between CSVs and
classifier labels still resolve.
"""

from __future__ import annotations

DEFAULT_SPECIALTY = "General Medicine"


def _key(disease: str) -> str:
    return " ".join(disease.split()).strip().lower()


# Disease -> specialty for the Kaggle "Disease Symptom Description" dataset.
_SPECIALTY_TABLE: dict[str, str] = {
    "fungal infection": "Dermatology",
    "allergy": "Allergy & Immunology",
    "gerd": "Gastroenterology",
    "chronic cholestasis": "Gastroenterology",
    "drug reaction": "Allergy & Immunology",
    "peptic ulcer diseae": "Gastroenterology",
    "aids": "Infectious Disease",
    "diabetes": "Endocrinology",
    "gastroenteritis": "Gastroenterology",
    "bronchial asthma": "Pulmonology",
    "hypertension": "Cardiology",
    "migraine": "Neurology",
    "cervical spondylosis": "Orthopedics",
    "paralysis (brain hemorrhage)": "Neurology",
    "jaundice": "Gastroenterology",
    "malaria": "Infectious Disease",
    "chicken pox": "Infectious Disease",
    "dengue": "Infectious Disease",
    "typhoid": "Infectious Disease",
    "hepatitis a": "Hepatology",
    "hepatitis b": "Hepatology",
    "hepatitis c": "Hepatology",
    "hepatitis d": "Hepatology",
    "hepatitis e": "Hepatology",
    "alcoholic hepatitis": "Hepatology",
    "tuberculosis": "Pulmonology",
    "common cold": "General Medicine",
    "pneumonia": "Pulmonology",
    "dimorphic hemmorhoids(piles)": "Gastroenterology",
    "heart attack": "Cardiology",
    "varicose veins": "Vascular Surgery",
    "hypothyroidism": "Endocrinology",
    "hyperthyroidism": "Endocrinology",
    "hypoglycemia": "Endocrinology",
    "osteoarthristis": "Orthopedics",
    "arthritis": "Rheumatology",
    "(vertigo) paroymsal  positional vertigo": "Neurology",
    "acne": "Dermatology",
    "urinary tract infection": "Urology",
    "psoriasis": "Dermatology",
    "impetigo": "Dermatology",
}


class SpecialtyMapper:
    """Map disease names to the specialty that treats them."""

    def __init__(
        self,
        table: dict[str, str] | None = None,
        *,
        default: str = DEFAULT_SPECIALTY,
    ) -> None:
        source = table if table is not None else _SPECIALTY_TABLE
        self._table = {_key(name): specialty for name, specialty in source.items()}
        self._default = default

    def specialty_for(self, disease_name: str) -> str:
        """Return the specialty for ``disease_name`` (default if unknown)."""
        return self._table.get(_key(disease_name), self._default)

    def has(self, disease_name: str) -> bool:
        """True if ``disease_name`` has an explicit mapping (not the default)."""
        return _key(disease_name) in self._table
