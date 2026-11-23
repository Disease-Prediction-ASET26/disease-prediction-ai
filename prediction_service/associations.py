"""Disease -> characteristic-symptom associations, learned from the dataset.

The trained classifier tells us *which* disease is most likely, but not which
unasked symptom would best confirm or rule it out. That information lives in the
raw training data (``dataset.csv``: one disease per row plus up to 17
``Symptom_N`` columns), which the training path collapses into feature vectors
and throws away. This module keeps it.

:class:`SymptomAssociationIndex` reads the same CSV and, per disease, records how
often each symptom appears across that disease's rows. From there it answers the
question the adaptive follow-up loop needs:

    given the current most-likely disease (and its closest rival), which symptom
    that the patient has not yet confirmed or denied would best separate the two?

A symptom that is common in the top disease but *rare* in the runner-up is the
most discriminating: the yes/no answer moves probability mass decisively. A
symptom shared by both barely helps. The selector scores candidates on exactly
that difference, so each follow-up question is worth asking.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .vocabulary import CANONICAL_SYMPTOMS, normalise_symptom

DEFAULT_DATASET_PATH = Path("data/dataset.csv")

ASSOCIATIONS_SCHEMA_VERSION = 1

_CANONICAL_BY_NORMALISED: dict[str, str] = {
    normalise_symptom(symptom): symptom for symptom in CANONICAL_SYMPTOMS
}


def _key(disease: str) -> str:
    return " ".join(disease.split()).strip().lower()


@dataclass(frozen=True)
class SymptomCandidate:
    """A symptom proposed as the next follow-up question.

    ``score`` is the discrimination score (higher is better): how much more
    frequent the symptom is in the top disease than in the runner-up, weighted
    by how strongly it marks the top disease.
    """

    symptom: str
    top_frequency: float
    rival_frequency: float
    score: float


class SymptomAssociationIndex:
    """Per-disease symptom frequencies, with discriminating-symptom selection."""

    def __init__(self, frequencies: dict[str, dict[str, float]]) -> None:
        # frequencies: normalised-disease-key -> {canonical_symptom: 0..1 share}
        self._frequencies = frequencies

    @classmethod
    def from_csv(
        cls, dataset_path: str | Path = DEFAULT_DATASET_PATH
    ) -> SymptomAssociationIndex:
        """Build the index from the Kaggle ``dataset.csv``.

        A missing file yields an empty index (the follow-up loop then simply
        never finds a question and returns the model's best guess as-is).
        """
        source = Path(dataset_path)
        if not source.exists():
            return cls({})

        row_counts: dict[str, int] = defaultdict(int)
        symptom_counts: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        with source.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            symptom_columns = [
                c for c in (reader.fieldnames or []) if c and c != "Disease"
            ]
            for row in reader:
                disease = (row.get("Disease") or "").strip()
                if not disease:
                    continue
                key = _key(disease)
                row_counts[key] += 1
                seen: set[str] = set()
                for column in symptom_columns:
                    canonical = _canonical_for(row.get(column))
                    if canonical is None or canonical in seen:
                        continue
                    seen.add(canonical)
                    symptom_counts[key][canonical] += 1

        frequencies: dict[str, dict[str, float]] = {}
        for key, total in row_counts.items():
            if total == 0:
                continue
            frequencies[key] = {
                symptom: count / total for symptom, count in symptom_counts[key].items()
            }
        return cls(frequencies)

    @classmethod
    def from_frequencies(
        cls, frequencies: dict[str, dict[str, float]]
    ) -> SymptomAssociationIndex:
        """Build an index from a precomputed per-disease frequency table.

        Keys are normalised to the same disease and symptom forms the CSV path
        produces, so an index loaded from a baked artifact behaves identically
        to one built from ``dataset.csv``.
        """
        normalised: dict[str, dict[str, float]] = {}
        for disease, symptoms in frequencies.items():
            disease_key = _key(disease)
            if not disease_key:
                continue
            bucket = normalised.setdefault(disease_key, {})
            for symptom, frequency in symptoms.items():
                canonical = _canonical_for(symptom)
                if canonical is None:
                    continue
                bucket[canonical] = float(frequency)
        return cls(normalised)

    @classmethod
    def from_json(cls, path: str | Path) -> SymptomAssociationIndex:
        """Load a baked associations artifact written by :meth:`save_json`.

        A missing file yields an empty index, matching :meth:`from_csv`: the
        follow-up loop then finds no question and returns the best guess as-is.
        """
        source = Path(path)
        if not source.exists():
            return cls({})
        document = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError(f"Invalid associations artifact: {source}")
        schema = document.get("schema_version")
        if schema != ASSOCIATIONS_SCHEMA_VERSION:
            raise ValueError(f"Unsupported associations schema: {schema}")
        frequencies = document.get("frequencies")
        if not isinstance(frequencies, dict):
            raise TypeError(f"Associations artifact has no frequencies: {source}")
        return cls.from_frequencies(frequencies)

    def to_frequencies(self) -> dict[str, dict[str, float]]:
        """Return a copy of the per-disease ``{symptom: frequency}`` table."""
        return {
            disease: dict(symptoms) for disease, symptoms in self._frequencies.items()
        }

    def save_json(self, path: str | Path) -> None:
        """Write the frequency table to ``path`` as a versioned JSON artifact.

        This is the deployable form of the index: baked beside the classifier so
        the follow-up loop works in images that do not ship the raw dataset.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": ASSOCIATIONS_SCHEMA_VERSION,
            "frequencies": {
                disease: dict(symptoms)
                for disease, symptoms in sorted(self._frequencies.items())
            },
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(target)

    def has(self, disease_name: str) -> bool:
        """True if the index carries symptom frequencies for the disease."""
        return _key(disease_name) in self._frequencies

    def symptoms_for(self, disease_name: str) -> dict[str, float]:
        """Return ``{canonical_symptom: frequency}`` for a disease (0..1)."""
        return dict(self._frequencies.get(_key(disease_name), {}))

    def next_question_symptom(
        self,
        top_disease: str,
        *,
        rival_disease: str | None = None,
        exclude: Iterable[str] = (),
        min_top_frequency: float = 0.3,
    ) -> SymptomCandidate | None:
        """Pick the most discriminating unasked symptom for the top disease.

        ``exclude`` holds every symptom already confirmed or denied (and any the
        patient volunteered), so the same question is never asked twice.
        ``min_top_frequency`` ignores symptoms that only weakly mark the top
        disease, keeping questions relevant. Returns ``None`` when no useful
        symptom remains.
        """
        top = self._frequencies.get(_key(top_disease))
        if not top:
            return None

        rival = (
            self._frequencies.get(_key(rival_disease), {})
            if rival_disease is not None
            else {}
        )
        excluded = {normalise_symptom(name) for name in exclude}

        best: SymptomCandidate | None = None
        for symptom, top_frequency in top.items():
            if top_frequency < min_top_frequency:
                continue
            if normalise_symptom(symptom) in excluded:
                continue
            rival_frequency = rival.get(symptom, 0.0)
            # Discrimination: present in the top disease, absent in the rival.
            # Weight by how strongly it marks the top disease so a near-certain
            # marker wins ties over a merely-different one.
            score = (top_frequency - rival_frequency) * top_frequency
            if best is None or score > best.score:
                best = SymptomCandidate(
                    symptom=symptom,
                    top_frequency=top_frequency,
                    rival_frequency=rival_frequency,
                    score=score,
                )
        return best


def _canonical_for(raw: str | None) -> str | None:
    if not raw:
        return None
    name = raw.strip()
    if not name:
        return None
    return _CANONICAL_BY_NORMALISED.get(normalise_symptom(name))
