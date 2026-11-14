"""Red-flag rule engine: detect symptom patterns needing urgent attention.

The classifier ranks *likely* diseases; this module answers a different,
safety-oriented question: do the reported symptoms include a pattern that should
trigger an urgent-care warning regardless of the predicted disease?

A :class:`RedFlagRule` is a named, severity-tagged predicate over the set of
active canonical symptoms. :class:`RedFlagEngine` evaluates every rule and
returns the ones that fired. Rules are plain data + a trigger set, so the
built-in set can be extended or replaced without touching the engine.

This is a deterministic rule engine, **not** a diagnosis -- it errs toward
caution and is meant to sit alongside the probabilistic prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Urgency(str, Enum):
    EMERGENCY = "EMERGENCY"  # seek emergency care now
    URGENT = "URGENT"  # see a clinician promptly
    ADVISORY = "ADVISORY"  # worth monitoring / mentioning


@dataclass(frozen=True)
class RedFlagRule:
    """A named rule that fires when enough of its trigger symptoms are present.

    ``min_matches`` lets a rule require a *combination* (e.g. any 2 of several
    cardiac symptoms) rather than a single symptom.
    """

    name: str
    urgency: Urgency
    message: str
    triggers: frozenset[str]
    min_matches: int = 1

    def evaluate(self, active: set[str]) -> RedFlag | None:
        matched = sorted(active & self.triggers)
        if len(matched) >= self.min_matches:
            return RedFlag(
                name=self.name,
                urgency=self.urgency,
                message=self.message,
                matched_symptoms=matched,
            )
        return None


@dataclass
class RedFlag:
    """A fired rule, with the symptoms that triggered it."""

    name: str
    urgency: Urgency
    message: str
    matched_symptoms: list[str] = field(default_factory=list)


# Built-in rules over canonical Kaggle symptom names.
DEFAULT_RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        name="possible_cardiac_event",
        urgency=Urgency.EMERGENCY,
        message="Chest pain with breathlessness or sweating may indicate a "
        "cardiac emergency. Seek emergency care immediately.",
        triggers=frozenset({"chest_pain", "breathlessness", "sweating"}),
        min_matches=2,
    ),
    RedFlagRule(
        name="possible_stroke",
        urgency=Urgency.EMERGENCY,
        message="Sudden weakness or altered sensorium can indicate a stroke. "
        "Seek emergency care immediately.",
        triggers=frozenset(
            {"weakness_of_one_body_side", "altered_sensorium", "slurred_speech"}
        ),
    ),
    RedFlagRule(
        name="severe_dehydration",
        urgency=Urgency.URGENT,
        message="Persistent vomiting with dehydration needs prompt review.",
        triggers=frozenset({"vomiting", "dehydration"}),
        min_matches=2,
    ),
    RedFlagRule(
        name="high_fever",
        urgency=Urgency.URGENT,
        message="A high fever warrants prompt clinical assessment.",
        triggers=frozenset({"high_fever"}),
    ),
    RedFlagRule(
        name="acute_breathing_difficulty",
        urgency=Urgency.URGENT,
        message="Breathlessness should be assessed promptly.",
        triggers=frozenset({"breathlessness"}),
    ),
    RedFlagRule(
        name="gi_bleeding",
        urgency=Urgency.URGENT,
        message="Blood in stool or vomit needs prompt medical review.",
        triggers=frozenset({"bloody_stool"}),
    ),
)


class RedFlagEngine:
    """Evaluate red-flag rules against a set of active symptoms."""

    def __init__(self, rules: tuple[RedFlagRule, ...] = DEFAULT_RULES) -> None:
        self._rules = rules

    def check(self, active_symptoms: list[str]) -> list[RedFlag]:
        """Return every rule that fired, most urgent first."""
        active = set(active_symptoms)
        fired = [
            flag for rule in self._rules if (flag := rule.evaluate(active)) is not None
        ]
        order = {Urgency.EMERGENCY: 0, Urgency.URGENT: 1, Urgency.ADVISORY: 2}
        fired.sort(key=lambda flag: order[flag.urgency])
        return fired

    def highest_urgency(self, active_symptoms: list[str]) -> Urgency | None:
        """Return the most severe urgency among fired rules, or ``None``."""
        fired = self.check(active_symptoms)
        return fired[0].urgency if fired else None
