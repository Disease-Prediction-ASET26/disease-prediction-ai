from __future__ import annotations

from dataclasses import dataclass, field

from prediction_service.associations import SymptomAssociationIndex
from prediction_service.followup import (
    FollowUpService,
    symptom_to_question,
)
from prediction_service.models import DiseasePrediction, SymptomEntryDTO


@dataclass
class ScriptedClassifier:
    """Return a canned ranking per call, driven by the number of entries.

    The service rebuilds the entry list each turn, so the entry count rises as
    symptoms are confirmed. We key the scripted response on that count, which
    lets a test describe "confidence climbs as more symptoms are confirmed".
    """

    by_entry_count: dict[int, list[tuple[str, float]]]
    default: list[tuple[str, float]] = field(default_factory=list)
    seen_counts: list[int] = field(default_factory=list)

    def predict(self, entries: list[SymptomEntryDTO]) -> list[DiseasePrediction]:
        count = len(entries)
        self.seen_counts.append(count)
        ranking = self.by_entry_count.get(count, self.default)
        return [
            DiseasePrediction(rank=i, probability=p, disease_name=name)
            for i, (name, p) in enumerate(ranking, start=1)
        ]


def _index() -> SymptomAssociationIndex:
    # Flu marked by high_fever and chills; Common Cold shares cough only.
    return SymptomAssociationIndex(
        {
            "flu": {"cough": 1.0, "high_fever": 1.0, "chills": 0.8},
            "common cold": {"cough": 1.0, "continuous_sneezing": 0.6},
        }
    )


def test_symptom_to_question_is_readable():
    assert symptom_to_question("high_fever") == "Do you also have high fever?"
    assert (
        symptom_to_question("continuous_sneezing")
        == "Do you also have continuous sneezing?"
    )


def test_low_confidence_offers_discriminating_question():
    classifier = ScriptedClassifier({1: [("Flu", 0.55), ("Common Cold", 0.40)]})
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(confirmed_symptoms=["cough"])

    assert outcome.confident is False
    assert outcome.question is not None
    # high_fever is in Flu, absent from the cold -> most discriminating.
    assert outcome.question.symptom == "high_fever"
    assert outcome.question.top_disease == "Flu"
    assert outcome.question.confidence == 0.55


def test_high_confidence_offers_no_question():
    classifier = ScriptedClassifier({1: [("Flu", 0.92), ("Common Cold", 0.05)]})
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(confirmed_symptoms=["cough"])

    assert outcome.confident is True
    assert outcome.question is None


def test_confirming_a_symptom_raises_confidence_and_stops():
    # 1 entry -> vague; 2 entries (after a 'yes') -> confident.
    classifier = ScriptedClassifier(
        {
            1: [("Flu", 0.55), ("Common Cold", 0.40)],
            2: [("Flu", 0.88), ("Common Cold", 0.10)],
        }
    )
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(
        confirmed_symptoms=["cough", "high_fever"], rounds_used=1
    )

    assert outcome.confident is True
    assert outcome.question is None
    assert outcome.confirmed_symptoms == ["cough", "high_fever"]


def test_denied_symptom_is_not_asked_again():
    classifier = ScriptedClassifier({1: [("Flu", 0.55), ("Common Cold", 0.40)]})
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(
        confirmed_symptoms=["cough"],
        denied_symptoms=["high_fever"],
    )

    assert outcome.question is not None
    # high_fever denied -> next best discriminating symptom is chills.
    assert outcome.question.symptom == "chills"


def test_round_cap_stops_the_loop():
    classifier = ScriptedClassifier({1: [("Flu", 0.55), ("Common Cold", 0.40)]})
    service = FollowUpService(classifier, _index(), max_rounds=2)

    outcome = service.evaluate(confirmed_symptoms=["cough"], rounds_used=2)

    # Still below threshold, but the cap is reached: return best guess, no Q.
    assert outcome.confident is False
    assert outcome.question is None
    assert outcome.rounds_used == 2


def test_no_discriminating_symptom_left_stops_gracefully():
    classifier = ScriptedClassifier({1: [("Flu", 0.55), ("Common Cold", 0.40)]})
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(
        confirmed_symptoms=["cough"],
        denied_symptoms=["high_fever", "chills"],
    )

    assert outcome.confident is False
    assert outcome.question is None


def test_empty_predictions_are_handled():
    classifier = ScriptedClassifier({}, default=[])
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(confirmed_symptoms=["cough"])

    assert outcome.predictions == []
    assert outcome.confident is False
    assert outcome.question is None


def test_confirmed_symptoms_are_deduped():
    classifier = ScriptedClassifier({1: [("Flu", 0.92)]})
    service = FollowUpService(classifier, _index())

    outcome = service.evaluate(confirmed_symptoms=["cough", "cough", "Cough"])

    assert outcome.confirmed_symptoms == ["cough"]
    assert classifier.seen_counts == [1]
