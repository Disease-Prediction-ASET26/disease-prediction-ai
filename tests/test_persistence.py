from __future__ import annotations

from uuid import uuid4

import pytest

from prediction_service.classifier import RandomForestStrategy, SklearnStrategy
from prediction_service.features import encode_entries
from prediction_service.models import Severity, SymptomEntryDTO
from prediction_service.persistence import (
    ModelStore,
    load_strategy,
    save_strategy,
)


def _entry(symptom: str) -> SymptomEntryDTO:
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=symptom,
        severity=Severity.MODERATE,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=symptom,
    )


def _trained_strategy() -> RandomForestStrategy:
    flu = encode_entries([_entry("high_fever"), _entry("cough")])
    rash = encode_entries([_entry("itching"), _entry("skin_rash")])
    return RandomForestStrategy().fit(
        [flu, flu, rash, rash], ["Flu", "Flu", "Rash", "Rash"]
    )


# ---------------------------------------------------------------------------
# save_strategy / load_strategy (joblib round-trip)
# ---------------------------------------------------------------------------


def test_save_and_load_strategy_roundtrip(tmp_path):
    strategy = _trained_strategy()
    path = tmp_path / "model.joblib"

    save_strategy(strategy, path)
    assert path.exists()

    loaded = load_strategy(path)
    assert isinstance(loaded, SklearnStrategy)
    predictions = loaded.classify([_entry("high_fever"), _entry("cough")])
    assert predictions[0].disease_name == "Flu"


def test_save_strategy_creates_parent_dirs(tmp_path):
    path = tmp_path / "nested" / "dir" / "model.joblib"
    save_strategy(_trained_strategy(), path)
    assert path.exists()


def test_load_missing_model_raises_with_hint(tmp_path):
    with pytest.raises(FileNotFoundError, match="Train one first"):
        load_strategy(tmp_path / "absent.joblib")


def test_load_rejects_non_strategy_artifact(tmp_path):
    import joblib

    path = tmp_path / "bad.joblib"
    joblib.dump({"not": "a strategy"}, path)
    with pytest.raises(TypeError, match="classifier strategy"):
        load_strategy(path)


def test_load_falls_back_to_legacy_pickle_suffix(tmp_path):
    import joblib

    legacy = tmp_path / "model.pkl"
    joblib.dump(_trained_strategy(), legacy)

    # Ask for the .joblib name; loader should find the .pkl sibling.
    loaded = load_strategy(tmp_path / "model.joblib")
    assert isinstance(loaded, SklearnStrategy)


# ---------------------------------------------------------------------------
# ModelStore
# ---------------------------------------------------------------------------


def test_store_save_sets_active_and_loads_back(tmp_path):
    store = ModelStore(tmp_path)
    metadata = store.save(_trained_strategy(), version="v1")

    assert metadata.version == "v1"
    assert metadata.is_active is True
    assert store.active_version() == "v1"
    assert store.versions() == ["v1"]

    loaded = store.load_active()
    assert (
        loaded.classify([_entry("itching"), _entry("skin_rash")])[0].disease_name
        == "Rash"
    )


def test_store_metadata_records_algorithm(tmp_path):
    store = ModelStore(tmp_path)
    store.save(_trained_strategy(), version="v1", calibration_method="sigmoid")

    meta = store.metadata("v1")
    assert meta.algorithm == "RandomForestClassifier"
    assert meta.calibration_method == "sigmoid"
    assert meta.is_active is True


def test_store_keeps_multiple_versions_and_promotes(tmp_path):
    store = ModelStore(tmp_path)
    store.save(_trained_strategy(), version="v1")
    store.save(_trained_strategy(), version="v2", make_active=False)

    assert sorted(store.versions()) == ["v1", "v2"]
    assert store.active_version() == "v1"

    store.set_active("v2")
    assert store.active_version() == "v2"
    # v1 is no longer flagged active in its metadata view
    assert store.metadata("v1").is_active is False
    assert store.metadata("v2").is_active is True


def test_store_set_active_unknown_version_raises(tmp_path):
    store = ModelStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="No model version"):
        store.set_active("ghost")


def test_store_load_active_without_any_model_raises(tmp_path):
    store = ModelStore(tmp_path)
    with pytest.raises(FileNotFoundError, match="Train one first"):
        store.load_active()


def test_store_versions_empty_when_root_absent(tmp_path):
    store = ModelStore(tmp_path / "does_not_exist")
    assert store.versions() == []
    assert store.active_version() is None
