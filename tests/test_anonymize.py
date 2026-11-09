from __future__ import annotations

from uuid import uuid4

from prediction_service.anonymize import (
    anonymize_entries,
    anonymize_entry,
    pseudonymize_id,
    scrub_text,
)
from prediction_service.models import Severity, SymptomEntryDTO


def _entry(raw_text: str | None = "patient reported this") -> SymptomEntryDTO:
    return SymptomEntryDTO(
        entry_id=uuid4(),
        raw_text=raw_text,
        severity=Severity.MODERATE,
        duration_days=3,
        is_mapped=True,
        canonical_symptom="high_fever",
    )


# ---------------------------------------------------------------------------
# pseudonymize_id
# ---------------------------------------------------------------------------


def test_pseudonymize_is_deterministic():
    source = uuid4()
    assert pseudonymize_id(source) == pseudonymize_id(source)


def test_pseudonymize_changes_the_id():
    source = uuid4()
    assert pseudonymize_id(source) != source


def test_pseudonymize_distinct_inputs_give_distinct_outputs():
    a, b = uuid4(), uuid4()
    assert pseudonymize_id(a) != pseudonymize_id(b)


# ---------------------------------------------------------------------------
# scrub_text
# ---------------------------------------------------------------------------


def test_scrub_redacts_email():
    out = scrub_text("contact me at jane.doe@example.com please")
    assert "jane.doe@example.com" not in out
    assert "[redacted]" in out


def test_scrub_redacts_phone_number():
    out = scrub_text("call +1 (555) 123-4567 now")
    assert "555" not in out
    assert "[redacted]" in out


def test_scrub_redacts_long_digit_runs():
    out = scrub_text("MRN 123456789 on file")
    assert "123456789" not in out


def test_scrub_keeps_clinical_text():
    out = scrub_text("high fever and a cough for 3 days")
    assert out == "high fever and a cough for 3 days"


def test_scrub_handles_none_and_empty():
    assert scrub_text(None) is None
    assert scrub_text("") == ""


# ---------------------------------------------------------------------------
# anonymize_entry / anonymize_entries
# ---------------------------------------------------------------------------


def test_anonymize_entry_drops_raw_text_by_default():
    entry = _entry("my name is John and my email is john@x.com")
    out = anonymize_entry(entry)

    assert out.raw_text is None
    assert out.entry_id == pseudonymize_id(entry.entry_id)
    # clinical fields preserved
    assert out.canonical_symptom == "high_fever"
    assert out.severity == Severity.MODERATE
    assert out.duration_days == 3


def test_anonymize_entry_can_keep_scrubbed_text():
    entry = _entry("fever, reach me at a@b.com")
    out = anonymize_entry(entry, drop_raw_text=False)

    assert out.raw_text is not None
    assert "a@b.com" not in out.raw_text
    assert "fever" in out.raw_text


def test_anonymize_does_not_mutate_the_original():
    entry = _entry("sensitive")
    original_id = entry.entry_id
    anonymize_entry(entry)
    assert entry.entry_id == original_id
    assert entry.raw_text == "sensitive"


def test_anonymize_entries_batch():
    entries = [_entry(), _entry()]
    out = anonymize_entries(entries)
    assert len(out) == 2
    assert all(e.raw_text is None for e in out)
