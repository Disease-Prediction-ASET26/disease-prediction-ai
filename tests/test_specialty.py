from __future__ import annotations

from prediction_service.specialty import DEFAULT_SPECIALTY, SpecialtyMapper


def test_maps_known_diseases():
    mapper = SpecialtyMapper()
    assert mapper.specialty_for("Malaria") == "Infectious Disease"
    assert mapper.specialty_for("Hypertension") == "Cardiology"
    assert mapper.specialty_for("Fungal infection") == "Dermatology"
    assert mapper.specialty_for("Diabetes ") == "Endocrinology"


def test_unknown_disease_falls_back_to_default():
    mapper = SpecialtyMapper()
    assert mapper.specialty_for("Not A Disease") == DEFAULT_SPECIALTY


def test_matching_is_case_and_whitespace_insensitive():
    mapper = SpecialtyMapper()
    assert mapper.specialty_for("  heart   attack ") == "Cardiology"
    assert mapper.specialty_for("MALARIA") == "Infectious Disease"


def test_has_distinguishes_known_from_default():
    mapper = SpecialtyMapper()
    assert mapper.has("Malaria") is True
    assert mapper.has("Nonexistent") is False


def test_custom_table_and_default():
    mapper = SpecialtyMapper({"Flu": "Virology"}, default="Triage")
    assert mapper.specialty_for("flu") == "Virology"
    assert mapper.specialty_for("anything else") == "Triage"


def test_covers_all_41_kaggle_diseases():
    from prediction_service.specialty import _SPECIALTY_TABLE

    # The Kaggle dataset has 41 distinct diseases.
    assert len(_SPECIALTY_TABLE) == 41
