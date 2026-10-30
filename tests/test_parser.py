from __future__ import annotations

import numpy as np
import pytest

from prediction_service.models import Severity
from prediction_service.parser import (
    ClinicalBertExtractor,
    KeywordNegationDetector,
    NegspacyNegationDetector,
    ParseResult,
    SemanticEntityLinker,
    SymptomParser,
    _severity_from_weight,
)

from .conftest import (
    FakeExtractor,
    FakeLinker,
    FakeNegationDetector,
    make_entity,
    make_link,
)


def _parser(entities, negated, mapping, vocab):
    return SymptomParser(
        extractor=FakeExtractor(entities),
        negation_detector=FakeNegationDetector(frozenset(negated)),
        linker=FakeLinker(mapping),
        vocabulary=vocab,
    )


def test_parse_happy_path_builds_weighted_vector(tiny_vocab):
    entities = [
        make_entity("skin rash", start=0, end=9),
        make_entity("sneezing", start=14, end=22),
    ]
    mapping = {
        "skin rash": make_link("skin rash", "skin_rash", severity_weight=3),
        "sneezing": make_link("sneezing", "continuous_sneezing", severity_weight=4),
    }
    parser = _parser(entities, negated=[], mapping=mapping, vocab=tiny_vocab)

    result = parser.parse("skin rash and sneezing")

    assert result.feature_vector == [3, 4, 0, 0]
    assert result.active_symptoms() == ["skin_rash", "continuous_sneezing"]
    assert len(result.affirmed) == 2
    assert result.negated == []


def test_parse_filters_negated_before_linking(tiny_vocab):
    entities = [
        make_entity("skin rash", start=0, end=9),
        make_entity("headache", start=20, end=28),
    ]
    mapping = {
        "skin rash": make_link("skin rash", "skin_rash", severity_weight=3),
        "headache": make_link("headache", "headache", severity_weight=3),
    }
    parser = _parser(entities, negated=["headache"], mapping=mapping, vocab=tiny_vocab)

    result = parser.parse("skin rash but no headache")

    assert [e.text for e in result.negated] == ["headache"]
    assert result.active_symptoms() == ["skin_rash"]
    assert result.feature_vector == [3, 0, 0, 0]
    assert parser.linker.seen == ["skin rash"]


def test_parse_passes_raw_text_to_each_stage(tiny_vocab):
    parser = _parser([], negated=[], mapping={}, vocab=tiny_vocab)
    parser.parse("original text")
    assert parser.extractor.seen_text == "original text"
    assert parser.negation_detector.seen_text == "original text"


def test_parser_default_vocabulary_is_canonical():
    parser = SymptomParser(
        extractor=FakeExtractor([]),
        negation_detector=FakeNegationDetector(),
        linker=FakeLinker({}),
    )
    assert len(parser.vocabulary) == 131


def test_empty_string_yields_zero_vector(tiny_vocab):
    parser = _parser([], negated=[], mapping={}, vocab=tiny_vocab)
    result = parser.parse("")
    assert result.feature_vector == [0, 0, 0, 0]
    assert result.active_symptoms() == []
    assert result.linked == []


def test_text_with_no_medical_entities(tiny_vocab):
    parser = _parser([], negated=[], mapping={}, vocab=tiny_vocab)
    result = parser.parse("the weather is nice today")
    assert sum(result.feature_vector) == 0
    assert result.entities == []


def test_only_negated_entities_produces_empty_vector(tiny_vocab):
    entities = [
        make_entity("headache", start=0, end=8),
        make_entity("fever", start=13, end=18),
    ]
    mapping = {
        "headache": make_link("headache", "headache"),
        "fever": make_link("fever", "high_fever"),
    }
    parser = _parser(
        entities, negated=["headache", "fever"], mapping=mapping, vocab=tiny_vocab
    )
    result = parser.parse("no headache and no fever")
    assert result.affirmed == []
    assert len(result.negated) == 2
    assert result.feature_vector == [0, 0, 0, 0]


def test_numpy_vector_is_1d_int8_matching_vocab_length(tiny_vocab):
    entities = [make_entity("skin rash", start=0, end=9)]
    mapping = {"skin rash": make_link("skin rash", "skin_rash", severity_weight=3)}
    parser = _parser(entities, negated=[], mapping=mapping, vocab=tiny_vocab)
    result = parser.parse("skin rash")

    vec = result.numpy_vector()
    assert isinstance(vec, np.ndarray)
    assert vec.ndim == 1
    assert vec.dtype == np.int8
    assert vec.shape == (len(tiny_vocab),)
    assert vec.tolist() == [3, 0, 0, 0]


def test_numpy_vector_bool_dtype(tiny_vocab):
    entities = [make_entity("skin rash", start=0, end=9)]
    mapping = {"skin rash": make_link("skin rash", "skin_rash", severity_weight=3)}
    parser = _parser(entities, negated=[], mapping=mapping, vocab=tiny_vocab)
    result = parser.parse("skin rash")

    vec = result.numpy_vector(dtype="bool")
    assert vec.dtype == np.bool_
    assert vec.ndim == 1
    assert vec.tolist() == [True, False, False, False]


def test_binary_vector_is_zero_one_list(tiny_vocab):
    entities = [make_entity("skin rash", start=0, end=9)]
    mapping = {"skin rash": make_link("skin rash", "skin_rash", severity_weight=7)}
    parser = _parser(entities, negated=[], mapping=mapping, vocab=tiny_vocab)
    result = parser.parse("skin rash")
    assert result.binary_vector() == [1, 0, 0, 0]
    assert set(result.binary_vector()) <= {0, 1}


@pytest.mark.parametrize(
    ("weight", "expected"),
    [
        (0, Severity.MILD),
        (1, Severity.MILD),
        (2, Severity.MILD),
        (3, Severity.MODERATE),
        (4, Severity.MODERATE),
        (5, Severity.SEVERE),
        (7, Severity.SEVERE),
    ],
)
def test_severity_from_weight_buckets(weight, expected):
    assert _severity_from_weight(weight) == expected


def test_to_symptom_entries_maps_fields(tiny_vocab):
    link = make_link("skin rash", "skin_rash", severity_weight=7)
    result = ParseResult(raw_text="x", linked=[link], vocabulary=tiny_vocab)
    entries = result.to_symptom_entries()
    assert len(entries) == 1
    entry = entries[0]
    assert entry.raw_text == "skin rash"
    assert entry.severity == Severity.SEVERE
    assert entry.is_mapped is True
    assert entry.duration_days == 0


def test_extractor_keeps_problem_spans_above_threshold(fake_transformers):
    fake_transformers.spans = [
        {
            "word": "skin rash",
            "entity_group": "problem",
            "score": 0.95,
            "start": 0,
            "end": 9,
        },
        {
            "word": "aspirin",
            "entity_group": "treatment",
            "score": 0.99,
            "start": 10,
            "end": 17,
        },
        {
            "word": "cough",
            "entity_group": "problem",
            "score": 0.10,
            "start": 18,
            "end": 23,
        },
    ]
    extractor = ClinicalBertExtractor()
    entities = extractor.extract("skin rash aspirin cough")

    assert [e.text for e in entities] == ["skin rash"]
    assert entities[0].label == "problem"
    assert fake_transformers.calls == ["skin rash aspirin cough"]


def test_extractor_pipeline_is_built_once(fake_transformers):
    fake_transformers.spans = []
    extractor = ClinicalBertExtractor()
    extractor.extract("a")
    extractor.extract("b")
    assert fake_transformers.calls == ["a", "b"]
    assert fake_transformers.pipeline_kwargs["aggregation_strategy"] == "first"


def test_extractor_injected_pipeline_skips_import():
    def fake_pipeline(text):
        return [
            {"word": "fever", "entity": "problem", "score": 0.8, "start": 0, "end": 5}
        ]

    extractor = ClinicalBertExtractor(pipeline=fake_pipeline)
    entities = extractor.extract("fever")
    assert entities[0].text == "fever"
    assert entities[0].label == "problem"


def test_extractor_empty_keep_labels_accepts_all():
    def fake_pipeline(text):
        return [
            {
                "word": "x",
                "entity_group": "treatment",
                "score": 0.9,
                "start": 0,
                "end": 1,
            }
        ]

    extractor = ClinicalBertExtractor(keep_labels=(), pipeline=fake_pipeline)
    assert len(extractor.extract("x")) == 1


def test_negspacy_flags_overlapping_negated_entity(fake_spacy):
    from tests.conftest import _FakeSpan

    fake_spacy.ents = [_FakeSpan(10, 18, negex=True)]
    detector = NegspacyNegationDetector()
    entities = [
        make_entity("skin rash", start=0, end=9),
        make_entity("headache", start=10, end=18),
    ]
    out = detector.annotate("skin rash headache", entities)
    assert out[0].is_negated is False
    assert out[1].is_negated is True


def test_negspacy_ignores_affirmed_spans(fake_spacy):
    from tests.conftest import _FakeSpan

    fake_spacy.ents = [_FakeSpan(0, 9, negex=False)]
    detector = NegspacyNegationDetector()
    entities = [make_entity("skin rash", start=0, end=9)]
    out = detector.annotate("skin rash", entities)
    assert out[0].is_negated is False


def test_negspacy_adds_negex_pipe_when_missing(fake_spacy):
    fake_spacy.pipe_names = []
    fake_spacy.ents = []
    detector = NegspacyNegationDetector()
    detector.annotate("text", [])
    assert "negex" in fake_spacy.added


def test_negspacy_overlaps_boundaries():
    assert NegspacyNegationDetector._overlaps(0, 5, 4, 10) is True
    assert NegspacyNegationDetector._overlaps(0, 5, 5, 10) is False


def test_keyword_detector_negates_after_cue():
    detector = KeywordNegationDetector()
    text = "patient has no headache"
    start = text.index("headache")
    entity = make_entity("headache", start=start, end=start + 8)
    out = detector.annotate(text, [entity])
    assert out[0].is_negated is True


def test_keyword_detector_affirms_without_cue():
    detector = KeywordNegationDetector()
    text = "patient has a headache"
    start = text.index("headache")
    entity = make_entity("headache", start=start, end=start + 8)
    out = detector.annotate(text, [entity])
    assert out[0].is_negated is False


def test_keyword_detector_terminator_cancels_negation():
    detector = KeywordNegationDetector(window=60)
    text = "no fever but headache"
    start = text.index("headache")
    entity = make_entity("headache", start=start, end=start + 8)
    out = detector.annotate(text, [entity])
    assert out[0].is_negated is False


def test_linker_semantic_match_above_threshold(fake_sentence_transformers, tiny_vocab):
    fake_sentence_transformers.scores = {"head hurts": [0.2, 0.1, 0.9, 0.0]}
    linker = SemanticEntityLinker(tiny_vocab, semantic_threshold=0.45)
    links = linker.link([make_entity("head hurts")])
    assert len(links) == 1
    assert links[0].canonical_symptom == "headache"
    assert links[0].method == "semantic"
    assert links[0].severity_weight == 3


def test_linker_semantic_below_threshold_falls_back_to_fuzzy(
    fake_sentence_transformers, tiny_vocab
):
    fake_sentence_transformers.scores = {"skin_rash": [0.1, 0.1, 0.1, 0.1]}
    linker = SemanticEntityLinker(tiny_vocab, semantic_threshold=0.45)
    links = linker.link([make_entity("skin_rash")])
    assert len(links) == 1
    assert links[0].canonical_symptom == "skin_rash"
    assert links[0].method == "fuzzy"


def test_linker_barely_passes_threshold(fake_sentence_transformers, tiny_vocab):
    fake_sentence_transformers.scores = {"q": [0.45, 0.0, 0.0, 0.0]}
    linker = SemanticEntityLinker(tiny_vocab, semantic_threshold=0.45)
    links = linker.link([make_entity("q")])
    assert links[0].method == "semantic"
    assert links[0].canonical_symptom == "skin_rash"


def test_linker_fuzzy_below_threshold_drops_entity(
    fake_sentence_transformers, tiny_vocab
):
    fake_sentence_transformers.scores = {"zzzzz qqqqq": [0.0, 0.0, 0.0, 0.0]}
    linker = SemanticEntityLinker(
        tiny_vocab, semantic_threshold=0.45, fuzzy_threshold=99.0
    )
    links = linker.link([make_entity("zzzzz qqqqq")])
    assert links == []


def test_linker_deduplicates_keeping_best(fake_sentence_transformers, tiny_vocab):
    fake_sentence_transformers.scores = {
        "rash one": [0.6, 0.0, 0.0, 0.0],
        "rash two": [0.9, 0.0, 0.0, 0.0],
    }
    linker = SemanticEntityLinker(tiny_vocab, semantic_threshold=0.45)
    links = linker.link([make_entity("rash one"), make_entity("rash two")])
    assert len(links) == 1
    assert links[0].source_text == "rash two"
    assert links[0].similarity == pytest.approx(0.9)


def test_linker_without_sentence_transformers_uses_fuzzy(
    no_sentence_transformers, tiny_vocab
):
    linker = SemanticEntityLinker(tiny_vocab)
    links = linker.link([make_entity("skin_rash")])
    assert links[0].method == "fuzzy"
    assert links[0].canonical_symptom == "skin_rash"


def test_linker_injected_model_skips_import(tiny_vocab):
    class Model:
        def encode(self, data, **kwargs):
            class V:
                def __init__(self, t):
                    self.t = t

            if isinstance(data, list):
                return [V(d) for d in data]
            return V(data)

    linker = SemanticEntityLinker(tiny_vocab, model=Model())
    assert linker._get_model() is not None


def test_linker_fuzzy_returns_none_when_rapidfuzz_missing(
    no_sentence_transformers, monkeypatch, tiny_vocab
):
    import sys

    monkeypatch.setitem(sys.modules, "rapidfuzz", None)
    linker = SemanticEntityLinker(tiny_vocab)
    assert linker.link([make_entity("skin_rash")]) == []


def test_linker_fuzzy_handles_no_match(no_sentence_transformers, tiny_vocab):
    linker = SemanticEntityLinker((), semantic_threshold=0.45)
    assert linker.link([make_entity("anything")]) == []


def test_extractor_drops_low_score_with_empty_keep_labels():
    def fake_pipeline(text):
        return [
            {"word": "x", "entity_group": "problem", "score": 0.1, "start": 0, "end": 1}
        ]

    extractor = ClinicalBertExtractor(
        keep_labels=(), min_score=0.5, pipeline=fake_pipeline
    )
    assert extractor.extract("x") == []


def test_keyword_detector_no_entities_returns_empty():
    detector = KeywordNegationDetector()
    assert detector.annotate("no fever", []) == []
