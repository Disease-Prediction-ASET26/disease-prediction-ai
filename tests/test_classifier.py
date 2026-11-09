from __future__ import annotations

from uuid import uuid4

import pytest

from prediction_service.classifier import (
    DecisionTreeStrategy,
    DiseaseClassifier,
    KNNStrategy,
    NaiveBayesStrategy,
    RandomForestStrategy,
    SklearnStrategy,
)
from prediction_service.factory import (
    DecisionTreeFactory,
    KNNFactory,
    NaiveBayesFactory,
    RandomForestFactory,
)
from prediction_service.features import encode_entries
from prediction_service.models import Severity, SymptomEntryDTO
from prediction_service.vocabulary import CANONICAL_SYMPTOMS

STRATEGIES = [
    RandomForestStrategy,
    KNNStrategy,
    DecisionTreeStrategy,
    NaiveBayesStrategy,
]


def make_entry(canonical_symptom: str, severity: Severity = Severity.MODERATE):
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=canonical_symptom,
        severity=severity,
        duration_days=0,
        is_mapped=True,
        canonical_symptom=canonical_symptom,
    )


def tiny_training_set():
    """Two separable diseases built from real canonical symptoms."""
    flu = encode_entries([make_entry("high_fever"), make_entry("cough")])
    rash = encode_entries([make_entry("itching"), make_entry("skin_rash")])
    features = [flu, flu, rash, rash]
    labels = ["Flu", "Flu", "Rash", "Rash"]
    return features, labels


def make_strategy(strategy_cls, **kwargs):
    """Instantiate a strategy, sizing KNN's neighbours to the tiny fixture."""
    if strategy_cls is KNNStrategy:
        kwargs.setdefault("n_neighbors", 1)
    return strategy_cls(**kwargs)


# ---------------------------------------------------------------------------
# features.encode_entries
# ---------------------------------------------------------------------------


def test_encode_entries_length_matches_vocabulary():
    vector = encode_entries([make_entry("itching")])
    assert len(vector) == len(CANONICAL_SYMPTOMS)


def test_encode_entries_uses_severity_weight_at_right_index():
    vector = encode_entries([make_entry("high_fever")])
    index = CANONICAL_SYMPTOMS.index("high_fever")
    assert vector[index] == 7  # Kaggle severity weight for high_fever
    assert sum(vector) == 7


def test_encode_entries_skips_unknown_and_none_symptoms():
    unknown = make_entry("not_a_real_symptom")
    no_canonical = make_entry("itching")
    no_canonical.canonical_symptom = None
    assert encode_entries([unknown, no_canonical]) == [0] * len(CANONICAL_SYMPTOMS)


def test_encode_entries_falls_back_to_severity_without_weight():
    entry = make_entry("not_a_real_symptom", severity=Severity.SEVERE)
    # unknown symptom -> not placed in the vector at all
    assert sum(encode_entries([entry])) == 0


# ---------------------------------------------------------------------------
# strategies
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
def test_strategy_predicts_correct_disease(strategy_cls):
    features, labels = tiny_training_set()
    strategy = make_strategy(strategy_cls).fit(features, labels)

    predictions = strategy.classify([make_entry("high_fever"), make_entry("cough")])

    assert predictions[0].disease_name == "Flu"
    assert predictions[0].rank == 1
    assert 0.0 <= predictions[0].probability <= 1.0


@pytest.mark.parametrize("strategy_cls", STRATEGIES)
def test_predictions_are_ranked_and_capped(strategy_cls):
    features, labels = tiny_training_set()
    strategy = make_strategy(strategy_cls, top_k=1).fit(features, labels)

    predictions = strategy.classify([make_entry("itching")])

    assert len(predictions) == 1
    assert predictions[0].rank == 1


def test_classify_before_fit_raises():
    with pytest.raises(RuntimeError, match="must be trained"):
        RandomForestStrategy().classify([make_entry("itching")])


def test_injected_estimator_is_considered_fitted():
    features, labels = tiny_training_set()
    from sklearn.tree import DecisionTreeClassifier

    estimator = DecisionTreeClassifier(random_state=0).fit(features, labels)
    strategy = DecisionTreeStrategy(estimator=estimator)

    predictions = strategy.classify([make_entry("itching"), make_entry("skin_rash")])
    assert predictions[0].disease_name == "Rash"


def test_algorithm_name_exposed():
    assert RandomForestStrategy().algorithm == "RandomForestClassifier"
    assert KNNStrategy().algorithm == "KNeighborsClassifier"


# ---------------------------------------------------------------------------
# DiseaseClassifier context + factories
# ---------------------------------------------------------------------------


def test_disease_classifier_delegates_and_swaps_strategy():
    features, labels = tiny_training_set()
    rf = RandomForestStrategy().fit(features, labels)
    knn = KNNStrategy(n_neighbors=1).fit(features, labels)

    classifier = DiseaseClassifier(rf)
    assert classifier.predict([make_entry("high_fever")])[0].disease_name == "Flu"

    classifier.set_strategy(knn)
    assert classifier.predict([make_entry("itching")])[0].disease_name == "Rash"


@pytest.mark.parametrize(
    "factory_cls,expected",
    [
        (RandomForestFactory, RandomForestStrategy),
        (KNNFactory, KNNStrategy),
        (DecisionTreeFactory, DecisionTreeStrategy),
        (NaiveBayesFactory, NaiveBayesStrategy),
    ],
)
def test_factories_create_expected_strategy(factory_cls, expected):
    strategy = factory_cls().create_classifier()
    assert isinstance(strategy, expected)
    assert isinstance(strategy, SklearnStrategy)
