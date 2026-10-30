from __future__ import annotations

import sys
import types
from dataclasses import dataclass

import pytest

from prediction_service.parser import ExtractedEntity, LinkedSymptom


@dataclass
class FakeExtractor:
    entities: list[ExtractedEntity]

    def extract(self, text: str) -> list[ExtractedEntity]:
        self.seen_text = text
        return list(self.entities)


@dataclass
class FakeNegationDetector:
    negated_texts: frozenset[str] = frozenset()

    def annotate(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        self.seen_text = text
        for entity in entities:
            if entity.text in self.negated_texts:
                entity.is_negated = True
        return entities


@dataclass
class FakeLinker:
    mapping: dict[str, LinkedSymptom]

    def link(self, entities: list[ExtractedEntity]) -> list[LinkedSymptom]:
        self.seen = [e.text for e in entities]
        out: list[LinkedSymptom] = []
        for entity in entities:
            link = self.mapping.get(entity.text)
            if link is not None:
                out.append(link)
        return out


def make_entity(
    text: str,
    *,
    start: int = 0,
    end: int | None = None,
    label: str = "problem",
    score: float = 0.99,
    is_negated: bool = False,
) -> ExtractedEntity:
    return ExtractedEntity(
        text=text,
        label=label,
        start=start,
        end=end if end is not None else len(text),
        score=score,
        is_negated=is_negated,
    )


def make_link(
    source_text: str,
    canonical_symptom: str,
    *,
    similarity: float = 0.9,
    method: str = "semantic",
    severity_weight: int = 3,
) -> LinkedSymptom:
    return LinkedSymptom(
        source_text=source_text,
        canonical_symptom=canonical_symptom,
        similarity=similarity,
        method=method,
        severity_weight=severity_weight,
    )


@pytest.fixture
def tiny_vocab() -> tuple[str, ...]:
    return ("skin_rash", "continuous_sneezing", "headache", "high_fever")


class _FakeSpan:
    def __init__(self, start_char: int, end_char: int, negex: bool) -> None:
        self.start_char = start_char
        self.end_char = end_char
        self._ = types.SimpleNamespace(negex=negex)


class _FakeDoc:
    def __init__(self, ents: list[_FakeSpan]) -> None:
        self.ents = ents


@pytest.fixture
def fake_transformers(monkeypatch):
    module = types.ModuleType("transformers")
    state = types.SimpleNamespace(spans=[], calls=[])

    class _FakeTokenizer:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    class _FakeModel:
        @classmethod
        def from_pretrained(cls, name):
            return cls()

    def _pipeline(task, *, model, tokenizer, aggregation_strategy):
        state.pipeline_kwargs = {
            "task": task,
            "aggregation_strategy": aggregation_strategy,
        }

        def _run(text):
            state.calls.append(text)
            return state.spans

        return _run

    module.AutoTokenizer = _FakeTokenizer
    module.AutoModelForTokenClassification = _FakeModel
    module.pipeline = _pipeline
    monkeypatch.setitem(sys.modules, "transformers", module)
    return state


@pytest.fixture
def fake_spacy(monkeypatch):
    spacy_module = types.ModuleType("spacy")
    state = types.SimpleNamespace(ents=[], pipe_names=[], added=[])

    class _FakeNlp:
        def __init__(self) -> None:
            self.pipe_names = list(state.pipe_names)

        def add_pipe(self, name, *, last=False):
            self.added.append(name)
            self.pipe_names.append(name)

        def __call__(self, text):
            state.seen_text = text
            return _FakeDoc(state.ents)

    def _load(name):
        state.loaded = name
        nlp = _FakeNlp()
        nlp.added = state.added
        return nlp

    spacy_module.load = _load

    negspacy_module = types.ModuleType("negspacy")
    negation_module = types.ModuleType("negspacy.negation")
    negation_module.Negex = object
    negspacy_module.negation = negation_module

    monkeypatch.setitem(sys.modules, "spacy", spacy_module)
    monkeypatch.setitem(sys.modules, "negspacy", negspacy_module)
    monkeypatch.setitem(sys.modules, "negspacy.negation", negation_module)
    return state


@pytest.fixture
def fake_sentence_transformers(monkeypatch):
    module = types.ModuleType("sentence_transformers")
    util_module = types.ModuleType("sentence_transformers.util")
    state = types.SimpleNamespace(scores={}, default_row=None, encoded=[])

    class _Vec:
        def __init__(self, text):
            self.text = text

    class _FakeModel:
        def __init__(self, name):
            state.model_name = name

        def encode(self, data, *, convert_to_tensor=True, normalize_embeddings=True):
            state.encoded.append(data)
            if isinstance(data, list):
                return [_Vec(d) for d in data]
            return _Vec(data)

    class _Row:
        def __init__(self, values):
            self._values = values

        def __getitem__(self, idx):
            return self._values[idx]

        def argmax(self):
            best = 0
            for i, v in enumerate(self._values):
                if v > self._values[best]:
                    best = i
            return best

    def _cos_sim(query, vocab_vecs):
        row = state.scores.get(query.text)
        if row is None:
            row = state.default_row
        if row is None:
            row = [0.0] * len(vocab_vecs)
        return [_Row(row)]

    module.SentenceTransformer = _FakeModel
    util_module.cos_sim = _cos_sim
    module.util = util_module
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    monkeypatch.setitem(sys.modules, "sentence_transformers.util", util_module)
    return state


@pytest.fixture
def no_sentence_transformers(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    return monkeypatch
