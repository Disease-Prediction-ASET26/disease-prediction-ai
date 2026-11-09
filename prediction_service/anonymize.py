"""Anonymise data before it is used for training or logging (NFR-2).

Two concerns are covered:

* :func:`anonymize_entries` scrubs :class:`SymptomEntryDTO` records of the
  fields that can carry patient-identifying information -- the free-text
  ``raw_text`` and the per-record ``entry_id`` -- while preserving everything
  the classifier actually needs (the canonical symptom, severity, duration).
  The ``entry_id`` is replaced with a **salted, non-reversible** hash so records
  stay distinguishable without being traceable back to a source row.

* :func:`scrub_text` removes common direct identifiers (emails, phone numbers,
  long digit runs such as MRNs/SSNs) from a free-text string, so any text that
  *is* retained (e.g. for debugging) cannot leak contact details.

Neither step is a substitute for a full de-identification review, but together
they ensure the persisted training matrix and any retained text are free of the
obvious identifiers NFR-2 calls out.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from uuid import UUID

from .models import SymptomEntryDTO

# A fixed namespace so the same entry_id maps to the same pseudonym across a run
# while remaining non-reversible. Not a secret; it only de-correlates the hash
# from the raw UUID bytes.
_HASH_SALT = b"disease-prediction/anonymize/v1"

_EMAIL_RE = re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+")
_PHONE_RE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
# Standalone runs of 5+ digits (MRNs, SSNs, account numbers).
_LONG_DIGITS_RE = re.compile(r"\b\d{5,}\b")

_REDACTED = "[redacted]"


def pseudonymize_id(entry_id: UUID) -> UUID:
    """Return a stable, non-reversible pseudonym UUID for ``entry_id``."""
    digest = hashlib.sha256(_HASH_SALT + entry_id.bytes).digest()
    return UUID(bytes=digest[:16])


def scrub_text(text: str | None) -> str | None:
    """Redact emails, phone numbers and long digit runs from free text."""
    if not text:
        return text
    scrubbed = _EMAIL_RE.sub(_REDACTED, text)
    scrubbed = _PHONE_RE.sub(_REDACTED, scrubbed)
    scrubbed = _LONG_DIGITS_RE.sub(_REDACTED, scrubbed)
    return scrubbed


def anonymize_entry(
    entry: SymptomEntryDTO, *, drop_raw_text: bool = True
) -> SymptomEntryDTO:
    """Return a copy of ``entry`` with identifying fields removed.

    ``entry_id`` is pseudonymised; ``raw_text`` is dropped by default (set
    ``drop_raw_text=False`` to keep a *scrubbed* version instead).
    """
    raw_text = None if drop_raw_text else scrub_text(entry.raw_text)
    return replace(
        entry,
        entry_id=pseudonymize_id(entry.entry_id),
        raw_text=raw_text,
    )


def anonymize_entries(
    entries: list[SymptomEntryDTO], *, drop_raw_text: bool = True
) -> list[SymptomEntryDTO]:
    """Anonymise a batch of entries (see :func:`anonymize_entry`)."""
    return [anonymize_entry(e, drop_raw_text=drop_raw_text) for e in entries]
