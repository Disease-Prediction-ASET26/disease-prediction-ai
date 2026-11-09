from __future__ import annotations

import textwrap
from pathlib import Path

from prediction_service.knowledge import DiseaseInfo, DiseaseKnowledgeBase


def _write(path: Path, content: str) -> Path:
    path.write_text(textwrap.dedent(content).lstrip(), encoding="utf-8")
    return path


def test_from_csv_loads_description_and_precautions(tmp_path):
    desc = _write(
        tmp_path / "desc.csv",
        """
        Disease,Description
        Malaria,A mosquito-borne disease.
        """,
    )
    prec = _write(
        tmp_path / "prec.csv",
        """
        Disease,Precaution_1,Precaution_2,Precaution_3,Precaution_4
        Malaria,consult hospital,avoid oily food,,keep mosquitos out
        """,
    )

    kb = DiseaseKnowledgeBase.from_csv(desc, prec)

    assert kb.describe("Malaria") == "A mosquito-borne disease."
    # blank precaution cells are dropped
    assert kb.precautions_for("Malaria") == [
        "consult hospital",
        "avoid oily food",
        "keep mosquitos out",
    ]


def test_lookup_normalises_whitespace_and_case(tmp_path):
    desc = _write(
        tmp_path / "desc.csv",
        """
        Disease,Description
        (vertigo) Paroymsal  Positional Vertigo,Dizziness disorder.
        """,
    )
    kb = DiseaseKnowledgeBase.from_csv(desc, tmp_path / "missing.csv")

    # single-spaced, different case still resolves
    assert (
        kb.describe("(vertigo) paroymsal positional vertigo") == "Dizziness disorder."
    )


def test_missing_files_yield_empty_lookups(tmp_path):
    kb = DiseaseKnowledgeBase.from_csv(
        tmp_path / "nope.csv", tmp_path / "also-nope.csv"
    )
    assert kb.describe("Malaria") is None
    assert kb.precautions_for("Malaria") == []


def test_info_for_bundles_description_and_precautions(tmp_path):
    desc = _write(
        tmp_path / "desc.csv",
        """
        Disease,Description
        Acne,A skin condition.
        """,
    )
    prec = _write(
        tmp_path / "prec.csv",
        """
        Disease,Precaution_1,Precaution_2,Precaution_3,Precaution_4
        Acne,bath twice,avoid fatty food,drink plenty of water,avoid too many products
        """,
    )
    kb = DiseaseKnowledgeBase.from_csv(desc, prec)

    info = kb.info_for("Acne")
    assert isinstance(info, DiseaseInfo)
    assert info.disease_name == "Acne"
    assert info.description == "A skin condition."
    assert len(info.precautions) == 4


def test_unknown_disease_returns_defaults():
    kb = DiseaseKnowledgeBase(
        descriptions={"malaria": "x"}, precautions={"malaria": ["y"]}
    )
    assert kb.describe("Unknown Disease") is None
    assert kb.precautions_for("Unknown Disease") == []


def test_has_checks_both_lookups():
    kb = DiseaseKnowledgeBase(
        descriptions={"malaria": "x"}, precautions={"acne": ["y"]}
    )
    assert kb.has("Malaria") is True  # description only
    assert kb.has("Acne") is True  # precautions only
    assert kb.has("  mAlArIa ") is True  # normalised
    assert kb.has("Unknown") is False
