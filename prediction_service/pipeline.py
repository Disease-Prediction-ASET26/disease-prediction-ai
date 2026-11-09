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
from .models import PredictionResult
from .parser import ParseResult, SymptomParser

if TYPE_CHECKING:
    from .models import DiseasePrediction


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
