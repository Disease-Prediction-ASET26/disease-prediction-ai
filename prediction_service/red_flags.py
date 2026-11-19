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
    EMERGENCY = "EMERGENCY"  # call an ambulance (112) / go to the ER now
    URGENT = "URGENT"  # see a doctor promptly today
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


_CALL_112 = (
    "Call emergency services (112) or go to the nearest emergency department now."
)
_SEE_DOCTOR = "See a doctor promptly today; seek emergency care if it gets worse."

# Built-in rules over canonical Kaggle symptom names.
#
# Urgency is driven by symptoms only, never by a disease's precaution text
# (e.g. "consult nearest hospital" is attached to Osteoarthritis). EMERGENCY is
# kept to a short list of dangerous symptoms or combinations; URGENT covers
# single serious symptoms that need a doctor the same day.
DEFAULT_RULES: tuple[RedFlagRule, ...] = (
    RedFlagRule(
        name="possible_cardiac_event",
        urgency=Urgency.EMERGENCY,
        message="Chest pain with breathlessness or sweating may be a heart "
        f"emergency. {_CALL_112}",
        triggers=frozenset({"chest_pain", "breathlessness", "sweating"}),
        min_matches=2,
    ),
    RedFlagRule(
        name="possible_stroke",
        urgency=Urgency.EMERGENCY,
        message="Sudden weakness on one side, confusion or slurred speech can "
        f"be a stroke. {_CALL_112}",
        triggers=frozenset(
            {"weakness_of_one_body_side", "altered_sensorium", "slurred_speech"}
        ),
    ),
    RedFlagRule(
        name="loss_of_consciousness",
        urgency=Urgency.EMERGENCY,
        message=f"Loss of consciousness is a medical emergency. {_CALL_112}",
        triggers=frozenset({"coma"}),
    ),
    RedFlagRule(
        name="coughing_blood_with_breathlessness",
        urgency=Urgency.EMERGENCY,
        message="Coughing up blood while short of breath needs emergency care. "
        f"{_CALL_112}",
        triggers=frozenset({"blood_in_sputum", "breathlessness"}),
        min_matches=2,
    ),
    RedFlagRule(
        name="severe_dehydration",
        urgency=Urgency.URGENT,
        message=f"Persistent vomiting with dehydration. {_SEE_DOCTOR}",
        triggers=frozenset({"vomiting", "dehydration"}),
        min_matches=2,
    ),
    RedFlagRule(
        name="high_fever",
        urgency=Urgency.URGENT,
        message=f"A high fever. {_SEE_DOCTOR}",
        triggers=frozenset({"high_fever"}),
    ),
    RedFlagRule(
        name="acute_breathing_difficulty",
        urgency=Urgency.URGENT,
        message=f"Breathlessness. {_SEE_DOCTOR}",
        triggers=frozenset({"breathlessness"}),
    ),
    RedFlagRule(
        name="gi_bleeding",
        urgency=Urgency.URGENT,
        message=f"Blood in the stool. {_SEE_DOCTOR}",
        triggers=frozenset({"bloody_stool"}),
    ),
    RedFlagRule(
        name="stomach_bleeding",
        urgency=Urgency.URGENT,
        message=f"Signs of bleeding in the stomach. {_SEE_DOCTOR}",
        triggers=frozenset({"stomach_bleeding"}),
    ),
    RedFlagRule(
        name="acute_liver_failure",
        urgency=Urgency.URGENT,
        message=f"Signs of acute liver failure. {_SEE_DOCTOR}",
        triggers=frozenset({"acute_liver_failure"}),
    ),
    RedFlagRule(
        name="fluid_overload",
        urgency=Urgency.URGENT,
        message=f"Fluid build-up in the body (swelling). {_SEE_DOCTOR}",
        triggers=frozenset({"fluid_overload"}),
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
