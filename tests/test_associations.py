from __future__ import annotations

import textwrap
from pathlib import Path

from prediction_service.associations import SymptomAssociationIndex


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return path


def _dataset(path: Path) -> Path:
    # Flu: always high_fever + cough, chills in half the rows.
    # Common Cold: always cough, continuous_sneezing in half; never high_fever.
    return _write(
        path / "dataset.csv",
        """
        Disease,Symptom_1,Symptom_2,Symptom_3
        Flu,high_fever,cough,chills
        Flu,high_fever,cough,
        Common Cold,cough,continuous_sneezing,
        Common Cold,cough,,
        """,
    )


def test_from_csv_computes_per_disease_frequencies(tmp_path):
    index = SymptomAssociationIndex.from_csv(_dataset(tmp_path))

    flu = index.symptoms_for("Flu")
    assert flu["high_fever"] == 1.0
    assert flu["cough"] == 1.0
    assert flu["chills"] == 0.5

    cold = index.symptoms_for("Common Cold")
    assert cold["cough"] == 1.0
    assert cold["continuous_sneezing"] == 0.5
    assert "high_fever" not in cold


def test_missing_file_yields_empty_index(tmp_path):
    index = SymptomAssociationIndex.from_csv(tmp_path / "nope.csv")

    assert not index.has("Flu")
    assert index.symptoms_for("Flu") == {}
    assert index.next_question_symptom("Flu") is None


def test_next_question_prefers_discriminating_symptom(tmp_path):
    index = SymptomAssociationIndex.from_csv(_dataset(tmp_path))

    # Top=Flu, rival=Common Cold. cough is shared (no help); high_fever is
    # frequent in Flu and absent in the cold -> most discriminating.
    candidate = index.next_question_symptom("Flu", rival_disease="Common Cold")

    assert candidate is not None
    assert candidate.symptom == "high_fever"
    assert candidate.top_frequency == 1.0
    assert candidate.rival_frequency == 0.0


def test_next_question_excludes_already_asked(tmp_path):
    index = SymptomAssociationIndex.from_csv(_dataset(tmp_path))

    candidate = index.next_question_symptom(
        "Flu",
        rival_disease="Common Cold",
        exclude=["high_fever", "cough"],
    )

    # high_fever and cough excluded; chills (0.5 in Flu) is the next best.
    assert candidate is not None
    assert candidate.symptom == "chills"


def test_next_question_skips_weakly_associated_symptoms(tmp_path):
    index = SymptomAssociationIndex.from_csv(_dataset(tmp_path))

    # With a high floor, only symptoms present in all Flu rows qualify.
    candidate = index.next_question_symptom(
        "Flu",
        rival_disease="Common Cold",
        exclude=["high_fever", "cough"],
        min_top_frequency=0.9,
    )

    assert candidate is None


def test_next_question_unknown_disease_returns_none(tmp_path):
    index = SymptomAssociationIndex.from_csv(_dataset(tmp_path))

    assert index.next_question_symptom("Nonexistent") is None
