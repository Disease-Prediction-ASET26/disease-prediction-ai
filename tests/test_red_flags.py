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


def _names(symptoms: list[str]) -> set[str]:
    return {flag.name for flag in RedFlagEngine().check(symptoms)}


def test_coma_is_an_emergency_on_its_own():
    assert "loss_of_consciousness" in _names(["coma"])
    assert RedFlagEngine().highest_urgency(["coma"]) == Urgency.EMERGENCY


def test_blood_in_sputum_needs_breathlessness_to_be_an_emergency():
    assert "coughing_blood_with_breathlessness" not in _names(["blood_in_sputum"])

    both = ["blood_in_sputum", "breathlessness"]
    assert "coughing_blood_with_breathlessness" in _names(both)
    assert RedFlagEngine().highest_urgency(both) == Urgency.EMERGENCY


def test_new_single_symptom_urgent_rules():
    for symptom in ("stomach_bleeding", "acute_liver_failure", "fluid_overload"):
        assert symptom in _names([symptom])
        assert RedFlagEngine().highest_urgency([symptom]) == Urgency.URGENT


def test_benign_symptoms_fire_nothing():
    assert RedFlagEngine().check(["itching", "skin_rash", "cough"]) == []


def test_messages_tell_the_patient_what_to_do():
    for flag in RedFlagEngine().check(
        ["coma", "chest_pain", "breathlessness", "high_fever", "fluid_overload"]
    ):
        if flag.urgency == Urgency.EMERGENCY:
            assert "112" in flag.message
        else:
            assert "See a doctor promptly" in flag.message


def test_urgency_ordering_is_emergency_then_urgent():
    fired = RedFlagEngine().check(
        ["fluid_overload", "high_fever", "coma", "stomach_bleeding"]
    )
    urgencies = [flag.urgency for flag in fired]
    assert urgencies[0] == Urgency.EMERGENCY
    assert all(u == Urgency.URGENT for u in urgencies[1:])
