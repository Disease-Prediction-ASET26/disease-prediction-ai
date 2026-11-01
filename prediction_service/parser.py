from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

from .models import Severity, SymptomEntryDTO
from .severity import weight_for
from .vocabulary import CANONICAL_SYMPTOMS, normalise_symptom

if TYPE_CHECKING:
    import numpy as np


@dataclass
class ExtractedEntity:
    text: str
    label: str
    start: int
    end: int
    score: float
    is_negated: bool = False


@dataclass
class LinkedSymptom:
    source_text: str
    canonical_symptom: str
    similarity: float
    method: str
    severity_weight: int = 0


@dataclass
class ParseResult:
    raw_text: str
    entities: list[ExtractedEntity] = field(default_factory=list)
    affirmed: list[ExtractedEntity] = field(default_factory=list)
    negated: list[ExtractedEntity] = field(default_factory=list)
    linked: list[LinkedSymptom] = field(default_factory=list)
    feature_vector: list[int] = field(default_factory=list)
    vocabulary: tuple[str, ...] = CANONICAL_SYMPTOMS

    def active_symptoms(self) -> list[str]:
        return [link.canonical_symptom for link in self.linked]

    def binary_vector(self) -> list[int]:
        active = {link.canonical_symptom for link in self.linked}
        return [1 if symptom in active else 0 for symptom in self.vocabulary]

    def numpy_vector(self, dtype: str = "int8") -> np.ndarray:
        import numpy as np

        return np.asarray(self.feature_vector, dtype=dtype)

    def to_symptom_entries(self) -> list[SymptomEntryDTO]:
        return [
            SymptomEntryDTO(
                entry_id=uuid4(),
                raw_text=link.source_text,
                severity=_severity_from_weight(link.severity_weight),
                duration_days=0,
                is_mapped=True,
            )
            for link in self.linked
        ]


def _severity_from_weight(weight: int) -> Severity:
    if weight <= 2:
        return Severity.MILD
    if weight <= 4:
        return Severity.MODERATE
    return Severity.SEVERE


@runtime_checkable
class EntityExtractor(Protocol):
    def extract(self, text: str) -> list[ExtractedEntity]: ...


@runtime_checkable
class NegationDetector(Protocol):
    def annotate(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]: ...


@runtime_checkable
class EntityLinker(Protocol):
    def link(self, entities: list[ExtractedEntity]) -> list[LinkedSymptom]: ...


class ClinicalBertExtractor:
    def __init__(
        self,
        model_name: str = "samrawal/bert-base-uncased_clinical-ner",
        *,
        keep_labels: Sequence[str] = ("problem",),
        min_score: float = 0.5,
        pipeline: Any = None,
    ) -> None:
        self.model_name = model_name
        self.keep_labels = {label.lower() for label in keep_labels}
        self.min_score = min_score
        self._pipeline = pipeline

    def _get_pipeline(self) -> Any:
        if self._pipeline is None:
            from transformers import (
                AutoModelForTokenClassification,
                AutoTokenizer,
                pipeline,
            )

            tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            model = AutoModelForTokenClassification.from_pretrained(self.model_name)
            self._pipeline = pipeline(
                "token-classification",
                model=model,
                tokenizer=tokenizer,
                aggregation_strategy="first",
            )
        return self._pipeline

    def extract(self, text: str) -> list[ExtractedEntity]:
        ner = self._get_pipeline()
        raw_spans = ner(text)

        entities: list[ExtractedEntity] = []
        for span in raw_spans:
            label = str(span.get("entity_group", span.get("entity", ""))).lower()
            score = float(span.get("score", 0.0))
            if self.keep_labels and label not in self.keep_labels:
                continue
            if score < self.min_score:
                continue
            entities.append(
                ExtractedEntity(
                    text=str(span["word"]).strip(),
                    label=label,
                    start=int(span.get("start", 0)),
                    end=int(span.get("end", 0)),
                    score=score,
                )
            )
        return entities


class NegspacyNegationDetector:
    def __init__(
        self,
        spacy_model: str = "en_ner_bc5cdr_md",
        *,
        nlp: Any = None,
    ) -> None:
        self.spacy_model = spacy_model
        self._nlp = nlp

    def _get_nlp(self) -> Any:
        if self._nlp is None:
            import spacy
            from negspacy.negation import Negex  # noqa: F401

            nlp = spacy.load(self.spacy_model)
            if "negex" not in nlp.pipe_names:
                nlp.add_pipe("negex", last=True)
            self._nlp = nlp
        return self._nlp

    def annotate(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        nlp = self._get_nlp()
        doc = nlp(text)

        negated_spans: list[tuple[int, int]] = [
            (ent.start_char, ent.end_char)
            for ent in doc.ents
            if getattr(ent._, "negex", False)
        ]

        for entity in entities:
            entity.is_negated = any(
                self._overlaps(entity.start, entity.end, ns, ne)
                for ns, ne in negated_spans
            )
        return entities

    @staticmethod
    def _overlaps(a_start: int, a_end: int, b_start: int, b_end: int) -> bool:
        return a_start < b_end and b_start < a_end


class KeywordNegationDetector:
    NEGATION_CUES = (
        "no",
        "not",
        "without",
        "denies",
        "denied",
        "deny",
        "absent",
        "except",
        "negative for",
        "free of",
        "ruled out",
    )
    TERMINATORS = (" but ", " however ", ", ", ";")

    def __init__(self, window: int = 40) -> None:
        self.window = window

    def annotate(
        self, text: str, entities: list[ExtractedEntity]
    ) -> list[ExtractedEntity]:
        lowered = text.lower()
        for entity in entities:
            preceding = lowered[max(0, entity.start - self.window) : entity.start]
            entity.is_negated = self._is_negated(preceding)
        return entities

    def _is_negated(self, preceding: str) -> bool:
        cue_pos = -1
        for cue in self.NEGATION_CUES:
            idx = preceding.rfind(cue)
            cue_pos = max(cue_pos, idx)
        if cue_pos == -1:
            return False
        tail = preceding[cue_pos:]
        return not any(term in tail for term in self.TERMINATORS)


class SemanticEntityLinker:
    def __init__(
        self,
        vocabulary: Sequence[str] = CANONICAL_SYMPTOMS,
        *,
        model_name: str = "all-MiniLM-L6-v2",
        semantic_threshold: float = 0.45,
        fuzzy_threshold: float = 70.0,
        model: Any = None,
    ) -> None:
        self.vocabulary = tuple(vocabulary)
        self.normalised = [normalise_symptom(s) for s in self.vocabulary]
        self.model_name = model_name
        self.semantic_threshold = semantic_threshold
        self.fuzzy_threshold = fuzzy_threshold
        self._model = model
        self._vocab_embeddings: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError:
                return None
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _get_vocab_embeddings(self, model: Any) -> Any:
        if self._vocab_embeddings is None:
            self._vocab_embeddings = model.encode(
                self.normalised,
                convert_to_tensor=True,
                normalize_embeddings=True,
            )
        return self._vocab_embeddings

    def link(self, entities: list[ExtractedEntity]) -> list[LinkedSymptom]:
        links: dict[str, LinkedSymptom] = {}
        for entity in entities:
            link = self._link_one(entity.text)
            if link is None:
                continue
            existing = links.get(link.canonical_symptom)
            if existing is None or link.similarity > existing.similarity:
                links[link.canonical_symptom] = link
        return list(links.values())

    def _link_one(self, phrase: str) -> LinkedSymptom | None:
        model = self._get_model()
        if model is not None:
            semantic = self._semantic_match(model, phrase)
            if semantic is not None and semantic.similarity >= self.semantic_threshold:
                return semantic
        return self._fuzzy_match(phrase)

    def _semantic_match(self, model: Any, phrase: str) -> LinkedSymptom | None:
        from sentence_transformers import util

        query = model.encode(
            normalise_symptom(phrase),
            convert_to_tensor=True,
            normalize_embeddings=True,
        )
        scores = util.cos_sim(query, self._get_vocab_embeddings(model))[0]
        best_idx = int(scores.argmax())
        best_score = float(scores[best_idx])
        canonical = self.vocabulary[best_idx]
        return LinkedSymptom(
            source_text=phrase,
            canonical_symptom=canonical,
            similarity=best_score,
            method="semantic",
            severity_weight=weight_for(canonical),
        )

    def _fuzzy_match(self, phrase: str) -> LinkedSymptom | None:
        try:
            from rapidfuzz import fuzz, process
        except ImportError:
            return None
        match = process.extractOne(
            normalise_symptom(phrase),
            self.normalised,
            scorer=fuzz.token_set_ratio,
        )
        if match is None:
            return None
        _, score, idx = match
        if score < self.fuzzy_threshold:
            return None
        canonical = self.vocabulary[idx]
        return LinkedSymptom(
            source_text=phrase,
            canonical_symptom=canonical,
            similarity=float(score) / 100.0,
            method="fuzzy",
            severity_weight=weight_for(canonical),
        )


class SymptomParser:
    def __init__(
        self,
        extractor: EntityExtractor | None = None,
        negation_detector: NegationDetector | None = None,
        linker: EntityLinker | None = None,
        *,
        vocabulary: Sequence[str] = CANONICAL_SYMPTOMS,
    ) -> None:
        self.vocabulary = tuple(vocabulary)
        self.extractor = extractor or ClinicalBertExtractor()
        self.negation_detector = negation_detector or NegspacyNegationDetector()
        self.linker = linker or SemanticEntityLinker(self.vocabulary)

    def parse(self, text: str) -> ParseResult:
        entities = self.extractor.extract(text)

        entities = self.negation_detector.annotate(text, entities)
        affirmed = [e for e in entities if not e.is_negated]
        negated = [e for e in entities if e.is_negated]

        linked = self.linker.link(affirmed)

        weight_by_symptom = {
            link.canonical_symptom: link.severity_weight for link in linked
        }
        feature_vector = [
            weight_by_symptom.get(symptom, 0) for symptom in self.vocabulary
        ]

        return ParseResult(
            raw_text=text,
            entities=entities,
            affirmed=affirmed,
            negated=negated,
            linked=linked,
            feature_vector=feature_vector,
            vocabulary=self.vocabulary,
        )
