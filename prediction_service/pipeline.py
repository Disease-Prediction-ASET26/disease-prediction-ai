"""End-to-end pipeline: patient free text -> ranked disease predictions.

Wires the NLP :class:`SymptomParser` into a trained
:class:`DiseaseClassifier`:

    free text
        -> SymptomParser.parse(text)          (NER + negation + linking)
        -> ParseResult.to_symptom_entries()   (SymptomEntryDTO[])
        -> DiseaseClassifier.predict(entries) (ranked DiseasePrediction[])
        -> PredictionResult

Both stages are injectable so the heavy ML pieces stay swappable and testable.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from .classifier import ClassifierStrategy, DiseaseClassifier
from .models import PredictionResult, Severity, SymptomEntryDTO
from .parser import ParseResult, SymptomParser
from .vocabulary import CANONICAL_SYMPTOMS, normalise_symptom

if TYPE_CHECKING:
    from collections.abc import Sequence

    from .models import DiseasePrediction

_CANONICAL_BY_NORMALISED: dict[str, str] = {
    normalise_symptom(symptom): symptom for symptom in CANONICAL_SYMPTOMS
}


def to_canonical(symptom: str) -> str | None:
    """Map a symptom name (e.g. ``"high_fever"`` or ``"High Fever"``) to the
    exact canonical vocabulary entry, or ``None`` if it is unknown."""
    return _CANONICAL_BY_NORMALISED.get(normalise_symptom(symptom))


class PredictionPipeline:
    """Route free-text symptom descriptions through NLP and the classifier."""

    def __init__(
        self,
        classifier: DiseaseClassifier | ClassifierStrategy,
        parser: SymptomParser | None = None,
    ) -> None:
        self._parser = parser or SymptomParser()
        self._classifier = (
            classifier
            if isinstance(classifier, DiseaseClassifier)
            else DiseaseClassifier(classifier)
        )

    def parse(self, text: str) -> ParseResult:
        """Run only the NLP stage (useful for inspection/debugging)."""
        return self._parser.parse(text)

    def predict(self, text: str) -> PredictionResult:
        """Parse free text and return ranked disease predictions."""
        _, predictions = self.predict_detailed(text)
        return PredictionResult(
            result_id=uuid4(),
            generated_at=datetime.now(UTC),
            predictions=predictions,
        )

    def predict_detailed(
        self, text: str
    ) -> tuple[ParseResult, list[DiseasePrediction]]:
        """Return both the NLP parse result and the ranked predictions.

        Useful when the caller also wants the extracted symptoms (e.g. an API
        that echoes them back alongside the predictions).
        """
        parse_result = self._parser.parse(text)
        entries = parse_result.to_symptom_entries()
        predictions = self._classifier.predict(entries)
        return parse_result, predictions

    def predict_combined(
        self,
        text: str | None = None,
        symptoms: Sequence[str] = (),
    ) -> tuple[list[str], list[DiseasePrediction]]:
        """Predict from free text and/or explicit symptom names together.

        ``symptoms`` are canonical names picked from a checklist, so they skip
        the NLP stage; unknown names are ignored. Free text, when present, goes
        through the parser as usual. Returns the active canonical symptoms
        (text first, then checklist, without duplicates) and the predictions;
        the predictions are empty when no symptom was recognised.
        """
        entries: list[SymptomEntryDTO] = []
        if text and text.strip():
            entries.extend(self._parser.parse(text).to_symptom_entries())

        seen = {entry.canonical_symptom for entry in entries}
        for name in symptoms:
            canonical = to_canonical(name)
            if canonical is None or canonical in seen:
                continue
            seen.add(canonical)
            entries.append(
                SymptomEntryDTO(
                    entry_id=uuid4(),
                    raw_text=name,
                    severity=Severity.MODERATE,
                    duration_days=0,
                    is_mapped=True,
                    canonical_symptom=canonical,
                )
            )

        active = [
            entry.canonical_symptom
            for entry in entries
            if entry.canonical_symptom is not None
        ]
        if not active:
            return [], []
        return active, self._classifier.predict(entries)
