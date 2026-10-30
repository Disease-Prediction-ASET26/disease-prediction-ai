from __future__ import annotations

from .parser import (
    ClinicalBertExtractor,
    KeywordNegationDetector,
    NegspacyNegationDetector,
    SemanticEntityLinker,
    SymptomParser,
)

TEST_TEXT = (
    "I have been dealing with a skin rash and sneezing, "
    "but I have no headache and no fever."
)


def build_parser() -> SymptomParser:
    extractor = ClinicalBertExtractor()

    negation: NegspacyNegationDetector | KeywordNegationDetector
    try:
        import negspacy  # noqa: F401
        import spacy  # noqa: F401

        negation = NegspacyNegationDetector()
    except ImportError:
        print("[info] negspacy/spaCy not found -> using KeywordNegationDetector")
        negation = KeywordNegationDetector()

    linker = SemanticEntityLinker()
    return SymptomParser(extractor, negation, linker)


def main() -> None:
    parser = build_parser()
    result = parser.parse(TEST_TEXT)

    print("Input:")
    print(f"  {result.raw_text}\n")

    print("Extracted entities (NER):")
    for ent in result.entities:
        flag = "NEGATED" if ent.is_negated else "affirmed"
        print(f"  [{flag:8}] {ent.text!r}  ({ent.label}, score={ent.score:.2f})")

    print("\nLinked active symptoms (affirmed only):")
    for link in result.linked:
        print(
            f"  {link.source_text!r} -> {link.canonical_symptom!r} "
            f"({link.method}, sim={link.similarity:.2f}, weight={link.severity_weight})"
        )

    active = [(i, v) for i, v in enumerate(result.feature_vector) if v]
    print(f"\nFeature vector length: {len(result.feature_vector)}")
    print(f"Active (index, weight): {active}")
    print(f"Active symptoms: {result.active_symptoms()}")

    entries = result.to_symptom_entries()
    print(f"\nSymptomEntryDTO objects ready for DiseaseClassifier: {len(entries)}")


if __name__ == "__main__":
    main()
