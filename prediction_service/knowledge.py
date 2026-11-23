"""Disease descriptions and precautions from the Kaggle dataset.

Loads ``symptom_Description.csv`` (``Disease,Description``) and
``symptom_precaution.csv`` (``Disease,Precaution_1..4``) into a lookup keyed by
disease name. Disease names are matched on a whitespace-normalised, lowercased
key so minor spelling/spacing differences between the CSVs and the classifier
labels still resolve.

The CSVs are optional: if a file is missing the corresponding lookup is simply
empty, and callers get ``None`` / ``[]`` back.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_DESCRIPTION_PATH = Path("data/symptom_Description.csv")
DEFAULT_PRECAUTION_PATH = Path("data/symptom_precaution.csv")

KNOWLEDGE_SCHEMA_VERSION = 1

# Committed, build-time knowledge asset at the repository root. It is baked into
# runtime images and is the API's default description/precaution source.
DEFAULT_KNOWLEDGE_PATH = (
    Path(__file__).resolve().parent.parent / "disease_knowledge.json"
)


# Spelling variants between the Kaggle files: the dataset and precaution CSVs
# say "hemmorhoids" while symptom_Description.csv says "hemorrhoids". The CSVs
# are downloaded, not versioned here, so the names are reconciled on lookup.
_KEY_ALIASES: dict[str, str] = {
    "dimorphic hemorrhoids(piles)": "dimorphic hemmorhoids(piles)",
}


def _key(disease: str) -> str:
    key = " ".join(disease.split()).strip().lower()
    return _KEY_ALIASES.get(key, key)


@dataclass
class DiseaseInfo:
    disease_name: str
    description: str | None = None
    precautions: list[str] = field(default_factory=list)


class DiseaseKnowledgeBase:
    """Lookup of human-readable description + precautions per disease."""

    def __init__(
        self,
        descriptions: dict[str, str] | None = None,
        precautions: dict[str, list[str]] | None = None,
    ) -> None:
        self._descriptions = descriptions or {}
        self._precautions = precautions or {}

    @classmethod
    def from_csv(
        cls,
        description_path: str | Path = DEFAULT_DESCRIPTION_PATH,
        precaution_path: str | Path = DEFAULT_PRECAUTION_PATH,
    ) -> DiseaseKnowledgeBase:
        return cls(
            descriptions=_load_descriptions(description_path),
            precautions=_load_precautions(precaution_path),
        )

    @classmethod
    def from_dict(cls, data: dict) -> DiseaseKnowledgeBase:
        """Build from a ``{descriptions, precautions}`` mapping.

        Keys are normalised to the same lookup key the CSV path produces, so a
        base loaded from a baked artifact behaves identically to one built from
        the CSVs.
        """
        descriptions = {
            _key(name): str(text)
            for name, text in (data.get("descriptions") or {}).items()
        }
        precautions = {
            _key(name): [str(step) for step in steps]
            for name, steps in (data.get("precautions") or {}).items()
        }
        return cls(descriptions=descriptions, precautions=precautions)

    @classmethod
    def from_json(cls, path: str | Path) -> DiseaseKnowledgeBase:
        """Load a baked knowledge artifact written by :meth:`save_json`.

        A missing file yields an empty knowledge base, matching
        :meth:`from_csv`: callers then get ``None`` / ``[]`` back.
        """
        source = Path(path)
        if not source.exists():
            return cls()
        document = json.loads(source.read_text(encoding="utf-8"))
        if not isinstance(document, dict):
            raise TypeError(f"Invalid knowledge artifact: {source}")
        schema = document.get("schema_version")
        if schema != KNOWLEDGE_SCHEMA_VERSION:
            raise ValueError(f"Unsupported knowledge schema: {schema}")
        return cls.from_dict(document)

    def to_dict(self) -> dict:
        """Return a ``{descriptions, precautions}`` copy of the lookup."""
        return {
            "descriptions": dict(self._descriptions),
            "precautions": {
                disease: list(steps) for disease, steps in self._precautions.items()
            },
        }

    def save_json(self, path: str | Path) -> None:
        """Write the lookup to ``path`` as a versioned JSON artifact.

        This is the deployable form of the knowledge base: committed and baked
        into images so descriptions and precautions work without the raw CSVs.
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        document = {
            "schema_version": KNOWLEDGE_SCHEMA_VERSION,
            "descriptions": dict(sorted(self._descriptions.items())),
            "precautions": {
                disease: list(steps)
                for disease, steps in sorted(self._precautions.items())
            },
        }
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(target)

    def describe(self, disease_name: str) -> str | None:
        return self._descriptions.get(_key(disease_name))

    def precautions_for(self, disease_name: str) -> list[str]:
        return self._precautions.get(_key(disease_name), [])

    def has(self, disease_name: str) -> bool:
        """True if the disease appears in either the description or precautions."""
        key = _key(disease_name)
        return key in self._descriptions or key in self._precautions

    def info_for(self, disease_name: str) -> DiseaseInfo:
        return DiseaseInfo(
            disease_name=disease_name,
            description=self.describe(disease_name),
            precautions=self.precautions_for(disease_name),
        )


def _load_descriptions(path: str | Path) -> dict[str, str]:
    source = Path(path)
    if not source.exists():
        return {}
    out: dict[str, str] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            description = (row.get("Description") or "").strip()
            if disease:
                out[_key(disease)] = description
    return out


def _load_precautions(path: str | Path) -> dict[str, list[str]]:
    source = Path(path)
    if not source.exists():
        return {}
    out: dict[str, list[str]] = {}
    with source.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        precaution_columns = [
            c for c in (reader.fieldnames or []) if c and c != "Disease"
        ]
        for row in reader:
            disease = (row.get("Disease") or "").strip()
            if not disease:
                continue
            steps = [
                (row.get(column) or "").strip()
                for column in precaution_columns
                if (row.get(column) or "").strip()
            ]
            out[_key(disease)] = steps
    return out


def main() -> None:
    """Build the committed disease_knowledge.json from the Kaggle CSVs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Build the disease knowledge JSON asset from the CSVs."
    )
    parser.add_argument("--descriptions", default=str(DEFAULT_DESCRIPTION_PATH))
    parser.add_argument("--precautions", default=str(DEFAULT_PRECAUTION_PATH))
    parser.add_argument(
        "--out",
        default=None,
        help=f"Output path (default: {DEFAULT_KNOWLEDGE_PATH}).",
    )
    args = parser.parse_args()

    knowledge = DiseaseKnowledgeBase.from_csv(args.descriptions, args.precautions)
    out = Path(args.out) if args.out else DEFAULT_KNOWLEDGE_PATH
    knowledge.save_json(out)
    document = knowledge.to_dict()
    print(
        f"Wrote {out}: {len(document['descriptions'])} descriptions, "
        f"{len(document['precautions'])} precaution sets."
    )


if __name__ == "__main__":
    main()
