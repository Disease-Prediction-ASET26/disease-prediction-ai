# Disease Prediction AI — Symptom Parser

A Python NLP pipeline that turns a patient's free-text description into a
feature vector for a disease classifier trained on the Kaggle
[Disease Symptom Description dataset](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
(itachi9604).

It extracts symptoms with ClinicalBERT, filters out negated symptoms with
NegEx, maps the surviving phrases to the dataset's exact symptom vocabulary, and
emits a **severity-weighted** feature vector aligned to that vocabulary.

---

## Pipeline

```
patient free text
        │
        ▼
┌───────────────────────────────────────────────────────────────┐
│  SymptomParser.parse(text) -> ParseResult                      │
│                                                                │
│  Step 1  NER         ClinicalBERT token-classification         │
│                      -> candidate "problem" spans              │
│  Step 2  Negation    NegEx (negspacy over scispaCy)            │
│                      -> drop negated spans                     │
│  Step 3  Linking     SentenceTransformer cosine + RapidFuzz    │
│                      -> map affirmed spans to Kaggle vocab     │
│  Step 4  Vectorise   severity-weighted vector (Kaggle weights) │
└───────────────────────────────────────────────────────────────┘
        │                              │
        ▼                              ▼
ParseResult.feature_vector      ParseResult.to_symptom_entries()
(severity weights per symptom)  (SymptomEntryDTO[] for the classifier)
```

### Step 1 — Symptom extraction (NER)

`ClinicalBertExtractor` wraps a Hugging Face token-classification pipeline
(default `samrawal/bert-base-uncased_clinical-ner`). It keeps spans labelled
`problem` above a score threshold.

The pipeline uses `aggregation_strategy="first"` so sub-word tokens are merged
back into whole words. (`"simple"` was found to split words like *sneezing* into
`s` / `##nee` / `##zing`, which then failed to link — `"first"` reassembles
them.)

### Step 2 — Negation detection

`NegspacyNegationDetector` runs NegEx (via negspacy) over the **full sentence**,
then marks any extracted span that overlaps a negated entity. Negated spans are
removed **before** linking, so a negated symptom ("no headache") can never leak
into the feature vector.

Stop words are deliberately **not** removed — tokens like *no*, *without*, and
*except* are exactly what NegEx keys off.

A dependency-free `KeywordNegationDetector` is provided as a fallback for
environments without the scispaCy model (useful for tests and quick local runs).

### Step 3 — Entity linking

`SemanticEntityLinker` maps each affirmed phrase to the closest canonical Kaggle
symptom:

1. **Semantic** — `SentenceTransformer("all-MiniLM-L6-v2")` embeddings + cosine
   similarity. Accepted if the best score ≥ `semantic_threshold` (default 0.45).
2. **Fuzzy fallback** — RapidFuzz `token_set_ratio` when the semantic score is
   too low or sentence-transformers is unavailable. Accepted if the score ≥
   `fuzzy_threshold` (default 70).

Matching is done against a normalised view of the vocabulary, but the **exact**
canonical string is emitted so the vector stays aligned to the training columns.

### Step 4 — Feature vector

The vector holds, for each vocabulary symptom in order, the Kaggle
[`Symptom-severity.csv`](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset)
weight (1–7) if the symptom is active, else 0.

- `ParseResult.feature_vector` — `list[int]`, severity-weighted (default).
- `ParseResult.binary_vector()` — `list[int]`, plain 0/1 presence.
- `ParseResult.numpy_vector(dtype="int8")` — 1-D `numpy.ndarray` (also
  `dtype="bool"`), length == vocabulary length.

> **Training note:** the classifier must be trained with the **same** encoding
> the parser emits. If you train on severity-weighted features, keep
> `feature_vector`; if you train on one-hot presence, use `binary_vector()`.

---

## Classifier

`prediction_service/classifier.py` turns the parser's feature vector into a
ranked list of diseases. It uses the **Strategy** pattern so models are
interchangeable, all sharing one feature space (severity-weighted vectors
aligned to `CANONICAL_SYMPTOMS`).

The Kaggle dataset is almost perfectly separable, so several classic models
score ~98–100%. Four are provided:

| Strategy | Estimator | Notes |
| --- | --- | --- |
| `RandomForestStrategy` | `RandomForestClassifier` | Strong default; robust, gives calibrated-ish probabilities. |
| `KNNStrategy` | `KNeighborsClassifier` | Simple, competitive on this separable data. |
| `DecisionTreeStrategy` | `DecisionTreeClassifier` | Interpretable; near-perfect here. |
| `NaiveBayesStrategy` | `GaussianNB` | Fast baseline. |

All extend `SklearnStrategy`, which handles the shared plumbing: encoding
entries into feature vectors (`features.encode_entries`), fitting, and ranking
class probabilities into `DiseasePrediction` objects (top-k, default 5).

These four strategies landed across the two most recent changes. The first
introduced the `SklearnStrategy` base and the `RandomForest`, `KNN`,
`DecisionTree` and `NaiveBayes` strategies themselves, giving every model one
shared feature space and a `canonical_symptom` field on `SymptomEntryDTO` so
linked symptoms carry their exact vocabulary name. The second built the rest of
the machinery around them: a factory per strategy (`RandomForestFactory`,
`KNNFactory`, `DecisionTreeFactory`, `NaiveBayesFactory`), the
`features.encode_entries` encoder and `dataset.load_dataset` loader that feed
them, a `train.py` that fits and scores all four and pickles each one (plus a
`best.pkl`), and the end-to-end pipeline and HTTP API that route free text
through the chosen strategy to ranked diseases.

```python
from prediction_service.classifier import DiseaseClassifier, RandomForestStrategy
from prediction_service.dataset import load_dataset

X, y = load_dataset("dataset.csv")  # Kaggle dataset.csv
strategy = RandomForestStrategy().fit(X, y)

classifier = DiseaseClassifier(strategy)  # swap models with set_strategy(...)
predictions = classifier.predict(entries)  # entries from parser.to_symptom_entries()
```

Any scikit-learn `predict_proba` estimator can be injected
(`RandomForestStrategy(estimator=my_fitted_model)`), and factories in
`factory.py` (`RandomForestFactory`, `KNNFactory`, `DecisionTreeFactory`,
`NaiveBayesFactory`) construct the strategies.

`prediction_service/dataset.py` reads the raw Kaggle `dataset.csv` (one disease
per row, `Symptom_N` columns) into the same severity-weighted matrix the parser
emits, so training and inference share one encoding. The CSV is **not** bundled
— download it from Kaggle and pass its path to `load_dataset`.

---

## End-to-end pipeline

`prediction_service/pipeline.py` routes **free text straight through to ranked
predictions**, wiring the NLP parser into a trained classifier:

```
free text
  -> SymptomParser.parse(text)          NER + negation + linking
  -> ParseResult.to_symptom_entries()   SymptomEntryDTO[]
  -> DiseaseClassifier.predict(entries) ranked DiseasePrediction[]
  -> PredictionResult
```

```python
from prediction_service.classifier import RandomForestStrategy
from prediction_service.dataset import load_dataset
from prediction_service.pipeline import PredictionPipeline

X, y = load_dataset("dataset.csv")
strategy = RandomForestStrategy().fit(X, y)

pipeline = PredictionPipeline(strategy)  # or pass a DiseaseClassifier
result = pipeline.predict("I have a high fever and a cough, but no headache.")

for p in result.predictions:
    print(p.rank, p.disease_name, round(p.probability, 3))
```

Both stages are injectable — pass a custom `SymptomParser`
(`PredictionPipeline(strategy, parser=my_parser)`) to swap the NLP components,
and `pipeline.parse(text)` exposes the NLP stage alone for inspection.

---

## Training

Train all four strategies on the Kaggle dataset and persist them:

```powershell
python -m prediction_service.train --data data/dataset.csv --out models
```

This splits the data (stratified 80/20), prints held-out accuracy for each
model, refits the winners on the full dataset, and writes:

- `models/random_forest.pkl`, `models/knn.pkl`, `models/decision_tree.pkl`,
  `models/naive_bayes.pkl` — each a fitted strategy, ready to unpickle.
- `models/best.pkl` — a copy of the best-scoring model (what the API loads).

On this dataset all four reach **100% accuracy** — both on a stratified 80/20
hold-out and under 5-fold cross-validation (±0.00). The dataset is almost
perfectly separable (each disease has a near-unique symptom signature), so this
reflects how easy the data is rather than real-world diagnostic skill; the
harder, noisier stage in practice is the NLP parser. Random Forest is kept as
the default because it degrades most gracefully on the partial/ambiguous
symptom sets the parser actually produces.

> The dataset CSV is **not** bundled. Download `dataset.csv` from Kaggle
> ([itachi9604](https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset))
> into `data/`. `prediction_service/persistence.py` handles load/save of the
> pickled strategies.

---

## HTTP API

`prediction_service/api.py` is a FastAPI app that routes free text through the
pipeline. Train a model first (above), then run:

```powershell
uvicorn prediction_service.api:app --reload
```

Override the model path with the `DISEASE_MODEL_PATH` env var (defaults to
`models/best.pkl`). Each prediction is enriched with a description and
recommended precautions loaded from `data/symptom_Description.csv` and
`data/symptom_precaution.csv` (override paths with `DISEASE_DESCRIPTION_PATH` /
`DISEASE_PRECAUTION_PATH`). These CSVs are optional — if absent, `description`
is `null` and `precautions` is `[]`, and predictions still work.

### Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{"status": "ok"}` |
| `POST` | `/predict` | `{"text": "...", "top_k": 5}` | ranked diseases + description + precautions + active symptoms |
| `GET` | `/diseases/{name}` | — | description + precautions for one disease |

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "I have itching, a skin rash and nodal skin eruptions."}'
```

```json
{
  "predictions": [
    {
      "rank": 1,
      "disease_name": "Fungal infection",
      "probability": 1.0,
      "description": "In humans, fungal infections occur when an invading fungus takes over an area of the body ...",
      "precautions": ["bath twice", "use detol or neem in bathing water", "keep infected area dry", "use clean cloths"]
    }
  ],
  "active_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "disclaimer": "This is not a medical diagnosis."
}
```

- `422` if no known symptoms can be extracted from the text (or `text` is empty).
- `503` if no trained model is found — run the training command first.

Fetch info for a single disease directly (name matched case-insensitively,
ignoring extra whitespace):

```bash
curl http://localhost:8000/diseases/Malaria
```

```json
{
  "disease_name": "Malaria",
  "description": "An infectious disease caused by protozoan parasites ...",
  "precautions": ["Consult nearest hospital", "avoid oily food", "avoid non veg food", "keep mosquitos out"]
}
```

- `404` if the disease name is not in the knowledge base.

Interactive docs are served at `/docs` (Swagger UI) once the app is running.

---

## Vocabulary

`prediction_service/vocabulary.py` holds `CANONICAL_SYMPTOMS` — the dataset's
symptom columns in order. The raw header has 132 columns but `fluid_overload`
appears twice; the duplicate is collapsed, leaving **131** distinct symptoms.
The original dataset spellings (including quirks such as `spotting_ urination`,
`foul_smell_of urine`, `dischromic _patches`) are preserved verbatim so the
vector lines up with the trained model. Train the classifier on the
de-duplicated 131-column matrix to match.

`prediction_service/severity.py` holds the per-symptom Kaggle weights and
`weight_for(symptom)`.

---

## Installation

Requires Python 3.11–3.13.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1          # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The scispaCy NER model is not on PyPI — install it from its release URL:

```powershell
pip install "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.4/en_ner_bc5cdr_md-0.5.4.tar.gz"
```

The ClinicalBERT and MiniLM models download automatically from the Hugging Face
Hub on first use.

---

## Usage

Run the bundled demo (as a module, from the project root):

```powershell
python -m prediction_service.demo_parser
```

Or use the parser directly:

```python
from prediction_service.parser import SymptomParser

parser = SymptomParser()
result = parser.parse("I have a skin rash and sneezing, but no headache and no fever.")

result.active_symptoms()  # ['skin_rash', 'continuous_sneezing']
result.feature_vector  # severity weights, 0 elsewhere (length 131)
result.numpy_vector()  # 1-D int8 ndarray
result.to_symptom_entries()  # list[SymptomEntryDTO] for the classifier
```

### Components are injectable

`SymptomParser` accepts any extractor / negation detector / linker implementing
the `EntityExtractor`, `NegationDetector`, and `EntityLinker` protocols. This
keeps the heavy ML imports lazy and makes the pipeline easy to test or swap:

```python
from prediction_service.parser import SymptomParser, KeywordNegationDetector

parser = SymptomParser(negation_detector=KeywordNegationDetector())
```

---

## Public API (`prediction_service.parser`)

| Symbol | Kind | Purpose |
| --- | --- | --- |
| `SymptomParser` | class | Orchestrates the 4-step pipeline. |
| `ParseResult` | dataclass | Output: entities, affirmed/negated, linked, vectors. |
| `ExtractedEntity` | dataclass | A raw NER span (text, label, offsets, score, is_negated). |
| `LinkedSymptom` | dataclass | An affirmed span mapped to a canonical symptom + weight. |
| `ClinicalBertExtractor` | class | Step 1 — ClinicalBERT NER. |
| `NegspacyNegationDetector` | class | Step 2 — NegEx negation. |
| `KeywordNegationDetector` | class | Step 2 — dependency-free fallback. |
| `SemanticEntityLinker` | class | Step 3 — semantic + fuzzy linking. |
| `EntityExtractor` / `NegationDetector` / `EntityLinker` | Protocol | Injection seams. |

---

## Tests

The suite lives in `tests/` and runs in well under a second because **all ML
models are mocked** — no real inference, no network, no model downloads.

```powershell
python -m pytest tests
```

With branch coverage:

```powershell
python -m pytest tests --cov=prediction_service.parser --cov-branch --cov-report=term-missing
```

Current status: **40 tests, ~99% branch coverage** on `parser.py`.

### How mocking works

`tests/conftest.py` installs fake modules into `sys.modules` so the lazy imports
inside the parser resolve to lightweight stand-ins:

| Fixture | Replaces | Lets a test control |
| --- | --- | --- |
| `fake_transformers` | `transformers` | the NER spans the pipeline returns |
| `fake_spacy` | `spacy` + `negspacy.negation` | which entities NegEx marks negated |
| `fake_sentence_transformers` | `sentence_transformers` | cosine scores per query (to sit above/below the threshold) |
| `no_sentence_transformers` | forces import failure | the RapidFuzz fallback path |

It also provides `Fake*` implementations of the three protocols
(`FakeExtractor`, `FakeNegationDetector`, `FakeLinker`) for testing the
orchestrator in isolation, plus `make_entity` / `make_link` builders and a
`tiny_vocab` fixture.

### What is covered

- **Orchestration** — weighted vector assembly, negated spans filtered before
  linking, raw text forwarded to each stage, default 131-symptom vocabulary.
- **Edge cases** — empty string, text with no medical entities, text with only
  negated entities, phrases that barely pass vs. fail the similarity threshold.
- **Data types** — `numpy_vector()` is a 1-D `int8` (and `bool`) ndarray whose
  length matches the vocabulary; `binary_vector()` is 0/1.
- **Each component** — ClinicalBERT span filtering and sub-word merging config,
  NegEx overlap logic and pipe registration, keyword-detector cues and scope
  terminators, semantic/fuzzy thresholds, de-duplication, and the
  dependency-missing fallbacks.

---

## Project layout

```
prediction_service/
├── parser.py        # the pipeline (this README's subject)
├── vocabulary.py    # CANONICAL_SYMPTOMS (131) + normalise_symptom
├── severity.py      # Kaggle severity weights + weight_for
├── demo_parser.py   # runnable end-to-end demo
├── models.py        # domain dataclasses (SymptomEntryDTO, Severity, ...)
├── classifier.py    # classifier strategy interfaces
└── factory.py       # classifier factories
tests/
├── conftest.py      # fakes, fixtures, builders
└── test_parser.py   # 40 tests
requirements.txt
pytest.ini
```
