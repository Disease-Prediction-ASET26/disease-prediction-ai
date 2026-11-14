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

### Probability calibration

Raw `predict_proba` scores from tree ensembles rank correctly but aren't
well-calibrated — the numbers don't behave like true probabilities.
`CalibratedStrategy` (`classifier.py`) wraps any other strategy's estimator in
scikit-learn's `CalibratedClassifierCV` (`"sigmoid"` / Platt scaling by default,
or `"isotonic"`) so the probabilities attached to each `DiseasePrediction` are
trustworthy. It presents the same interface as any other strategy, so it drops
straight into `DiseaseClassifier`, the pipeline, persistence, and the training
script.

```python
from prediction_service.classifier import CalibratedStrategy, RandomForestStrategy

strategy = CalibratedStrategy(RandomForestStrategy(), method="sigmoid", cv=3)
strategy.fit(X, y)
```

**The served model is calibrated by default** — `train.py` calibrates the
best-scoring strategy before writing `best.joblib` (see [Training](#training)).

### Explanations

`prediction_service/explanation.py` attributes a prediction to the symptoms that
drove it. `ExplanationGenerator` reads the estimator's `feature_importances_`
(tree models) or `coef_` (linear models) — unwrapping the calibration wrapper
when present — and keeps only the symptoms that are *active* in the input,
producing an `Explanation` (a short `summary` plus normalised
`contribution_weights` per symptom).

Enable it per strategy with `.with_explanations()`; each `DiseasePrediction`
then carries an `explanation`. **The API enables this by default**, so every
`/predict` result includes the driving symptoms (see [HTTP API](#http-api)).

```python
strategy = RandomForestStrategy().fit(X, y).with_explanations()
prediction = DiseaseClassifier(strategy).predict(entries)[0]
prediction.explanation.summary  # "Fungal infection is driven mainly by: itching, ..."
prediction.explanation.contribution_weights  # {"itching": 0.49, "nodal_skin_eruptions": 0.31, ...}
```

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

## Model training and release

Train all four strategies on the Kaggle dataset and persist them:

```powershell
python -m prediction_service.train --data data/dataset.csv --out models
```

This splits the data (stratified 80/20), prints held-out accuracy for each
model, refits the winners on the full dataset, and writes:

- `models/random_forest.joblib`, `models/knn.joblib`,
  `models/decision_tree.joblib`, `models/naive_bayes.joblib` — each a fitted
  (uncalibrated) strategy, kept for comparison.
- `models/best.joblib` — the best-scoring model, **calibrated** (see below),
  and the artifact the API loads.
- `models/best.manifest.json` — model version, file and dataset checksums,
  training scores, source revision, and Python/scikit-learn versions.

By default the best model is wrapped in `CalibratedClassifierCV` before being
saved as `best.joblib`, so the probabilities the API returns are trustworthy.
Flags:

- `--no-calibrate` — persist the raw best model without calibration.
- `--calibration-method {sigmoid,isotonic}` — calibration method (default
  `sigmoid`). On datasets with too few samples per class for the CV folds,
  calibration is skipped automatically and the raw model is saved.

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
> strategies via **joblib**, and also provides a versioned `ModelStore` (model
> registry with metadata + an active-version pointer) when you need to keep and
> promote several trained models.

For a deployable model, train with the same pinned AI environment used by the
runtime image. On Linux or macOS, `scripts/train-model.sh` builds that
environment and writes the model and manifest under the ignored `models/`
directory. It requires a clean AI source checkout so the recorded commit is
accurate. Training runs on the workstation; it does not run in the API
container or in k3s.

### Review and publish a model

Train and review a candidate on your workstation. The release workflow never
trains a model. It copies the exact classifier from a digest-pinned runtime
image, checks its manifest and checksum, then builds an image for each CPU
architecture.

1. Commit the AI code that produced the model. Put the Kaggle CSV in the local
   `data/` directory. Do not commit the dataset or model files.
2. From the AI repository on Linux or macOS, run:

   ```bash
   ./scripts/train-model.sh
   ```

   Review the held-out scores in `models/best.manifest.json` and check the
   model with representative inputs. Record its full SHA-256:

   ```bash
   sha256sum models/best.joblib
   ```

3. Package the reviewed local files as a candidate image, following
   [Publish a trained model runtime](#publish-a-trained-model-runtime). Record
   the candidate image digest and the classifier SHA-256. The candidate image
   can contain one CPU architecture: the release workflow uses it only as an
   immutable source for the classifier.
4. Update `model-release-dev.json` with the candidate image digest and the
   classifier SHA-256. Open and merge a pull request into `develop`. After CI
   passes, the release workflow uses the base image built by that exact CI run
   and the selected classifier. It builds, scans, and runs an offline
   prediction on native AMD64 and ARM64 runners. It publishes an image index
   and opens a GitOps pull request for the Proxmox dev deployment. The workflow
   merges the update after GitOps validation and secret scanning pass. Argo CD
   then deploys the image.
5. After the team accepts the dev image for production, update
   `model-release-prod.json` with the accepted dev runtime image digest and
   classifier SHA-256. Merge the change into `develop`, then promote it to
   `main` in a pull request. After CI passes, the release workflow builds the
   production runtime and merges its GitOps update after the checks pass.

The workflow requires a GitHub App installed on `disease-prediction-gitops`
with Contents and Pull requests write access. Set `GITOPS_CLIENT_ID` to the app
Client ID and `GITOPS_APP_PRIVATE_KEY` to its complete PEM key as Actions
secrets in this repository. The classifier source image, base image, and output
image are pinned by digest; the workflow records them in its run summary.
Subsequent changes merged into `develop` or `main` rebuild the matching runtime
with the classifier pinned for that environment. To rerun a release manually,
select the matching branch and channel and supply the base image from that
branch's current CI run. Both image architectures must match the current source
revision. The source runtime image and classifier checksum must match the
environment's release file. Releases for each environment run one at a
time, and pending releases wait in a queue. A failed GitOps check leaves its
pull request open for inspection. The workflow refuses to deploy an outdated
source revision. Keep the pinned classifier source images and any images needed
for rollback in GHCR. Deleting a pinned source prevents later releases.

The runtime image contains the trained classifier and pinned ClinicalBERT and
MiniLM inference assets. `/ready` loads these local files before the pod
receives prediction traffic. The deployed image runs in Hugging Face offline
mode, so startup does not need Hub access or a writable model cache. Building
the image downloads the two model snapshots at the revisions in
`model-assets.json`; the artifact manifest and image digest identify the
classifier and model asset set. This downloads inference weights during image
build. It does not train the models.

---

## HTTP API

`prediction_service/api.py` is a FastAPI app that routes free text through the
pipeline. Train a model first (above), then run:

```powershell
uvicorn prediction_service.api:app --reload
```

Override the model path with the `DISEASE_MODEL_PATH` env var (defaults to
`models/best.joblib`). Each prediction is enriched with the medical
**specialty** that treats the disease, a human-readable **explanation** of the
driving symptoms, and a description and recommended precautions loaded from
`data/symptom_Description.csv` and `data/symptom_precaution.csv` (override paths
with `DISEASE_DESCRIPTION_PATH` / `DISEASE_PRECAUTION_PATH`). The CSVs are
optional — if absent, `description` is `null` and `precautions` is `[]`, and
predictions still work.

### Endpoints

| Method | Path | Body | Returns |
| --- | --- | --- | --- |
| `GET` | `/health` | — | `{"status": "ok"}` |
| `GET` | `/ready` | — | `{"status": "ready"}` after classifier and parser models load |
| `POST` | `/predict` | `{"text": "...", "top_k": 5}` | ranked diseases + probability + specialty + explanation + description + precautions + active symptoms |
| `POST` | `/map-symptoms` | `{"text": "..."}` | NLP-only: linked canonical symptoms + negated spans (no classifier) |
| `POST` | `/red-flags/check` | `{"text": "..."}` or `{"symptoms": [...]}` | fired red-flag rules + highest urgency |
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
      "probability": 0.975,
      "specialty": "Dermatology",
      "description": "In humans, fungal infections occur when an invading fungus takes over an area of the body ...",
      "precautions": ["bath twice", "use detol or neem in bathing water", "keep infected area dry", "use clean cloths"],
      "explanation": {
        "summary": "Fungal infection is driven mainly by: itching, nodal skin eruptions, skin rash.",
        "contribution_weights": {"itching": 0.4854, "nodal_skin_eruptions": 0.305, "skin_rash": 0.2096}
      }
    }
  ],
  "active_symptoms": ["itching", "skin_rash", "nodal_skin_eruptions"],
  "disclaimer": "This is not a medical diagnosis."
}
```

- `422` if no known symptoms can be extracted from the text (or `text` is empty).
- `503` if no trained model is found — run the training command first.

The `probability` is **calibrated** (the served `best.joblib` is wrapped in
`CalibratedClassifierCV`), and the `explanation` attributes the prediction to
the active symptoms that drove it. The `specialty` is the clinical specialty
that treats the disease (defaults to `General Medicine` for anything outside
the Kaggle set).

Map free text to canonical symptoms without running the classifier
(`POST /map-symptoms`), or screen symptoms against the red-flag rule engine
(`POST /red-flags/check`), which accepts either `text` or an explicit
`symptoms` list and returns the fired rules plus the highest urgency
(`EMERGENCY` / `URGENT` / `ADVISORY`):

```bash
curl -X POST http://localhost:8000/red-flags/check \
  -H "Content-Type: application/json" \
  -d '{"symptoms": ["chest_pain", "breathlessness"]}'
```

```json
{
  "active_symptoms": ["chest_pain", "breathlessness"],
  "red_flags": [
    {
      "name": "possible_cardiac_event",
      "urgency": "EMERGENCY",
      "message": "Chest pain with breathlessness or sweating may indicate a cardiac emergency. Seek emergency care immediately.",
      "matched_symptoms": ["breathlessness", "chest_pain"]
    }
  ],
  "highest_urgency": "EMERGENCY"
}
```

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

The released runtime image includes pinned ClinicalBERT and MiniLM weights.
Local Python runs that do not set model paths load the models from Hugging Face
on first use and use the normal local cache. Training records the dataset
checksum in the classifier manifest.

## Docker images

The shared Compose setup is in
[`disease-prediction-docker`](../disease-prediction-docker). It uses
`Dockerfile.local` for the source-mounted developer service,
`Dockerfile.dev` for a deployable dev image, and `Dockerfile.prod` for the
production runtime image. Each build excludes the local `data/` and `models/`
folders. The runtime images use CPU-only PyTorch. The published model runtime
adds the pinned inference weights to the image.

The API can start without a trained classifier, but `/predict` returns `503`
until `models/best.joblib` is present at the configured model path. `/ready`
returns success only after the classifier and manifest have been verified and
loaded, and the parser models have been loaded. `/health` remains a process
check. The published runtime image loads the parser models from local files in
offline mode. A pod cannot receive prediction traffic until this completes.

### Publish a trained model runtime

Use this command to put an already reviewed local classifier in GHCR as a
candidate. Run it on a native AMD64 or ARM64 host with Docker Buildx, Trivy,
and package write access. Replace the base digest, architecture, and unique
candidate tag with your values:

```sh
docker login ghcr.io
scripts/publish-model-runtime.sh dev \
  ghcr.io/disease-prediction-aset26/disease-prediction-ai-dev@sha256:<base-digest> \
  amd64 candidate-<unique-name> models
```

The script checks the artifact manifest, downloads pinned inference files,
scans for fixable HIGH and CRITICAL vulnerabilities, runs a prediction without
network access, and publishes the tested image. Record its printed digest for
the GitHub release workflow. The workflow builds and checks both architectures
and publishes their combined image index. Pin the resulting index digest in
GitOps. Keep the source dataset in the ignored `data/` directory and never
include it in the runtime image. Joblib files can execute code while loading,
so package only artifacts trained and reviewed by the project team.

The published model runtime contains the classifier and both pinned NLP model
directories. It does not need a Hugging Face cache volume or Hub network access
at startup. The model revisions and files included in the image are listed in
`model-assets.json`.

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

The suite lives in `tests/` (**146 tests**). The NLP parser tests mock all the
heavy ML models — no real inference, no network, no model downloads — so they
run in well under a second. The classifier, calibration and persistence tests
do train small scikit-learn models on tiny in-memory fixtures, so the full run
takes a few seconds.

```powershell
python -m pytest tests
```

With branch coverage on the parser:

```powershell
python -m pytest tests --cov=prediction_service.parser --cov-branch --cov-report=term-missing
```

The parser suite alone is **40 tests at ~99% branch coverage** on `parser.py`.

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
├── parser.py        # the NLP pipeline (free text -> canonical symptoms)
├── vocabulary.py    # CANONICAL_SYMPTOMS (131) + normalise_symptom
├── severity.py      # Kaggle severity weights + weight_for
├── features.py      # encode_entries: symptom entries -> feature vector
├── dataset.py       # load_dataset: Kaggle CSV -> feature matrix + labels
├── demo_parser.py   # runnable end-to-end demo
├── models.py        # domain dataclasses (SymptomEntryDTO, Explanation, ...)
├── classifier.py    # strategies + CalibratedStrategy + DiseaseClassifier
├── factory.py       # classifier factories
├── explanation.py   # ExplanationGenerator (per-symptom contributions)
├── specialty.py     # SpecialtyMapper (disease -> medical specialty, FR-P4)
├── red_flags.py     # RedFlagEngine (urgent-symptom rule engine)
├── anonymize.py     # training-data anonymisation (NFR-2)
├── knowledge.py     # DiseaseKnowledgeBase (descriptions + precautions)
├── persistence.py   # joblib save/load + versioned ModelStore
├── train.py         # train, calibrate best, persist
└── api.py           # FastAPI app (/predict, /map-symptoms, /red-flags/check, ...)
tests/
├── conftest.py            # fakes, fixtures, builders
├── test_parser.py         # NLP pipeline
├── test_classifier.py     # strategies + factories + encoding
├── test_calibration.py    # CalibratedStrategy
├── test_explanation.py    # ExplanationGenerator
├── test_specialty.py      # SpecialtyMapper
├── test_red_flags.py      # RedFlagEngine
├── test_anonymize.py      # anonymisation
├── test_persistence.py    # joblib round-trip + ModelStore
├── test_train.py          # calibration wiring
├── test_knowledge.py      # knowledge base
├── test_pipeline.py       # end-to-end pipeline
└── test_api.py            # HTTP endpoints
requirements.txt
pytest.ini
```
