"""Adaptive follow-up questions for vague symptom descriptions.

When the most-likely disease comes back below the confidence threshold, a single
prediction is not enough to be useful. Instead of stopping there, the service
asks the patient one targeted yes/no question about a symptom characteristic of
that disease, folds the answer into the symptom set, and re-predicts -- looping
until the top disease clears the threshold or no worthwhile question remains.

The prediction service is stateless, so the loop is *client-carried*: each turn
the caller sends back the symptoms confirmed and denied so far plus the latest
answer, and the service rebuilds the symptom set from scratch. Nothing is stored
between requests.

Design:

- confirmed symptoms (from the original parse or answered "yes") become
  :class:`SymptomEntryDTO` positives, encoded at their severity weight exactly as
  the parser would;
- denied symptoms (answered "no") are recorded only so they are never asked
  again -- an absent symptom is already how the model represents "not present",
  so there is nothing to add to the vector;
- :class:`SymptomAssociationIndex` chooses the next question as the symptom that
  best separates the top disease from its closest rival;
- a hard cap on rounds guarantees termination even for a patient whose answers
  never push any disease past the threshold.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4

from .associations import SymptomAssociationIndex
from .classifier import DiseaseClassifier
from .models import DiseasePrediction, Severity, SymptomEntryDTO
from .vocabulary import normalise_symptom

DEFAULT_CONFIDENCE_THRESHOLD = 0.80
DEFAULT_MAX_ROUNDS = 5


@dataclass(frozen=True)
class FollowUpQuestion:
    """A yes/no question to put to the patient about one symptom."""

    symptom: str
    question: str
    top_disease: str
    confidence: float


@dataclass
class FollowUpOutcome:
    """Result of one follow-up turn.

    ``predictions`` is the freshly re-ranked list. When ``question`` is not
    ``None`` the top disease is still below the threshold and the caller should
    put the question to the patient and call back with the answer. When
    ``question`` is ``None`` the loop is finished: either the threshold was
    reached or no useful question remains, and ``predictions`` is the final
    answer.
    """

    predictions: list[DiseasePrediction]
    confident: bool
    rounds_used: int
    question: FollowUpQuestion | None = None
    confirmed_symptoms: list[str] = field(default_factory=list)
    denied_symptoms: list[str] = field(default_factory=list)


def symptom_to_question(symptom: str) -> str:
    """Turn a canonical symptom name into a natural yes/no question.

    ``"high_fever"`` -> ``"Do you also have high fever?"``. The canonical
    vocabulary carries a few odd spellings (stray spaces, underscores); these are
    smoothed into readable words.
    """
    phrase = normalise_symptom(symptom)
    return f"Do you also have {phrase}?"


class FollowUpService:
    """Run the adaptive follow-up loop, one question per turn."""

    def __init__(
        self,
        classifier: DiseaseClassifier,
        associations: SymptomAssociationIndex,
        *,
        confidence_threshold: float = DEFAULT_CONFIDENCE_THRESHOLD,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
    ) -> None:
        self._classifier = classifier
        self._associations = associations
        self._threshold = confidence_threshold
        self._max_rounds = max_rounds

    def evaluate(
        self,
        *,
        confirmed_symptoms: list[str],
        denied_symptoms: list[str] | None = None,
        rounds_used: int = 0,
    ) -> FollowUpOutcome:
        """Re-rank from the current symptom set and decide the next step.

        ``confirmed_symptoms`` is every symptom known present so far (the
        original parse plus every "yes"); ``denied_symptoms`` every "no".
        ``rounds_used`` is how many follow-up questions have already been
        answered, used to enforce the cap.
        """
        denied = denied_symptoms or []
        confirmed = _dedupe(confirmed_symptoms)
        entries = [_positive_entry(symptom) for symptom in confirmed]
        predictions = self._classifier.predict(entries)

        outcome = FollowUpOutcome(
            predictions=predictions,
            confident=False,
            rounds_used=rounds_used,
            confirmed_symptoms=confirmed,
            denied_symptoms=list(denied),
        )

        if not predictions:
            return outcome

        top = predictions[0]
        if top.probability >= self._threshold:
            outcome.confident = True
            return outcome

        if rounds_used >= self._max_rounds:
            return outcome

        rival = predictions[1].disease_name if len(predictions) > 1 else None
        already_asked = set(confirmed) | set(denied)
        candidate = self._associations.next_question_symptom(
            top.disease_name,
            rival_disease=rival,
            exclude=already_asked,
        )
        if candidate is None:
            return outcome

        outcome.question = FollowUpQuestion(
            symptom=candidate.symptom,
            question=symptom_to_question(candidate.symptom),
            top_disease=top.disease_name,
            confidence=top.probability,
        )
        return outcome


def _positive_entry(symptom: str) -> SymptomEntryDTO:
    """Build a mapped, present symptom entry for the classifier to encode."""
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=None,
        severity=Severity.MODERATE,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=symptom,
    )


def _dedupe(symptoms: list[str]) -> list[str]:
    """Drop duplicates (by normalised name) while keeping first-seen order."""
    seen: set[str] = set()
    out: list[str] = []
    for symptom in symptoms:
        marker = normalise_symptom(symptom)
        if marker in seen:
            continue
        seen.add(marker)
        out.append(symptom)
    return out
