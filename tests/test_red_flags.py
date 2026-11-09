from __future__ import annotations

from prediction_service.red_flags import (
    RedFlagEngine,
    RedFlagRule,
    Urgency,
)


def test_single_trigger_fires():
    engine = RedFlagEngine()
    fired = engine.check(["high_fever"])
    names = {flag.name for flag in fired}
    assert "high_fever" in names


def test_combination_rule_requires_min_matches():
    engine = RedFlagEngine()
    # One cardiac symptom alone should NOT fire the 2-of rule.
    one = {flag.name for flag in engine.check(["chest_pain"])}
    assert "possible_cardiac_event" not in one

    # Two should.
    two = {flag.name for flag in engine.check(["chest_pain", "breathlessness"])}
    assert "possible_cardiac_event" in two


def test_matched_symptoms_recorded():
    engine = RedFlagEngine()
    fired = engine.check(["chest_pain", "sweating"])
    cardiac = next(f for f in fired if f.name == "possible_cardiac_event")
    assert cardiac.matched_symptoms == ["chest_pain", "sweating"]


def test_no_symptoms_fires_nothing():
    engine = RedFlagEngine()
    assert engine.check([]) == []
    assert engine.highest_urgency([]) is None


def test_results_sorted_emergency_first():
    engine = RedFlagEngine()
    # high_fever is URGENT; cardiac combo is EMERGENCY.
    fired = engine.check(["high_fever", "chest_pain", "breathlessness"])
    assert fired[0].urgency == Urgency.EMERGENCY


def test_highest_urgency():
    engine = RedFlagEngine()
    assert engine.highest_urgency(["high_fever"]) == Urgency.URGENT
    assert engine.highest_urgency(["chest_pain", "breathlessness"]) == Urgency.EMERGENCY


def test_custom_rules():
    rule = RedFlagRule(
        name="custom",
        urgency=Urgency.ADVISORY,
        message="watch this",
        triggers=frozenset({"itching"}),
    )
    engine = RedFlagEngine(rules=(rule,))
    fired = engine.check(["itching"])
    assert len(fired) == 1
    assert fired[0].name == "custom"
    assert fired[0].urgency == Urgency.ADVISORY
