# Diabetes Readmission Pipeline

A production-style machine learning system that predicts 30-day hospital readmission risk for diabetic patients, built end-to-end with a working MLOps pipeline: continuous integration, model versioning, automated validation gating, experiment tracking, automated retraining, and live drift monitoring.

**Live app:** [Streamlit frontend](https://diabetes-readmission-pipeline-frontend.onrender.com) · **API:** [FastAPI backend](https://diabetes-readmission-pipeline.onrender.com/docs)

---

## What this project demonstrates

Most portfolio ML projects stop at "train a model, wrap it in an API." This one goes further — it implements the operational pieces that keep a model reliable *after* it's deployed:

| Capability | Status | How it works |
|---|---|---|
| Continuous Integration | ✅ | Every push runs lint, a model-load sanity check, and the test suite before anything can merge |
| Model versioning | ✅ | Every training run is saved as a permanent, timestamped artifact with full metadata |
| Automated validation gate | ✅ | A retrained model is only promoted to production if it doesn't regress against the best model on record |
| Experiment tracking | ✅ | Every training run logs hyperparameters, metrics, and artifacts to MLflow |
| Automated retraining | ✅ | A scheduled (and manually triggerable) workflow retrains, validates, and — if promoted — auto-commits and redeploys the new model |
| Drift monitoring | ✅ | A live endpoint compares recent production traffic against the training data distribution and flags meaningful shifts |

---

## Architecture

```
                        ┌─────────────────────┐
                        │   GitHub (main)      │
                        └──────────┬───────────┘
                                   │ push
              ┌────────────────────┼────────────────────┐
              │                    │                     │
              ▼                    ▼                     ▼
     ┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
     │   CI workflow    │  │ Render (backend)  │  │ Render (frontend)     │
     │  lint + tests    │  │  FastAPI + XGBoost │  │  Streamlit UI         │
     └─────────────────┘  │  + SHAP explainer  │  └──────────┬───────────┘
                           └─────────┬──────────┘             │
                                     │ writes predictions      │ calls /predict
                                     ▼                         │
                           ┌──────────────────┐                │
                           │ Postgres (Render) │◄───────────────┘
                           │ prediction_logs   │
                           └─────────┬──────────┘
                                     │ read by
                                     ▼
                           ┌──────────────────────┐
                           │ /monitoring/drift      │
                           │ compares live traffic  │
                           │ vs. training baseline  │
                           └──────────────────────┘

     ┌───────────────────────────────────────────────────┐
     │  Automated Retraining (scheduled / manual trigger)  │
     │  1. python main.py  → train + evaluate              │
     │  2. Validation gate → compare ROC-AUC vs. best        │
     │  3. If passed → promote model + save baseline         │
     │  4. Log run to MLflow                                 │
     │  5. Commit + push models/ back to GitHub → redeploy    │
     └───────────────────────────────────────────────────┘
```

---

## Repository structure

```
diabetes-readmission-pipeline/
├── .github/workflows/
│   ├── ci.yml                  # Lint, model-load check, tests — runs on every push
│   └── retrain.yml             # Automated retraining — scheduled + manual trigger
├── data/
│   └── diabetic_data.csv       # Training dataset
├── models/
│   ├── diabetes_readmission_xgb_model.json   # Current production model
│   ├── feature_baseline.json                 # Training-data stats used for drift comparison
│   ├── registry.json                         # Log of every training run ever promoted or attempted
│   └── versions/                              # Permanent, timestamped snapshot of every model + its metadata
├── src/
│   ├── app.py                  # FastAPI backend: /predict, /health, /monitoring/drift
│   ├── frontend.py             # Streamlit UI
│   ├── preprocess.py           # Data cleaning + feature engineering
│   └── train.py                # Training pipeline: versioning, validation gate, MLflow, baseline
├── tests/
│   └── test_api.py             # API test suite (used by CI)
├── main.py                     # Entry point: python main.py runs the full training pipeline
├── requirements.txt            # Backend dependencies (also used for local training)
├── Dockerfile.backend
├── Dockerfile.frontend
└── docker-compose.yml          # Local multi-service testing
```

---

## How each MLOps piece works

### 1. Continuous Integration (`.github/workflows/ci.yml`)
On every push or pull request to `main`:
- **Lint** — `ruff` checks for real errors (undefined names, unused imports)
- **Model-load check** — confirms the committed model artifact actually loads and has valid features, catching a corrupted/missing model before it reaches deployment
- **Tests** — `pytest tests/` runs the API test suite (health check, a full prediction round-trip, and input validation)

### 2. Model versioning (`src/train.py`)
Every training run saves a permanent, uniquely named copy of the model to `models/versions/`, tagged with a timestamp and the git commit that produced it. Nothing is ever overwritten — you can always trace exactly which code and data produced any given model. Each version also gets a `.metadata.json` file recording hyperparameters, full evaluation metrics, and row counts.

### 3. Automated validation gate (`src/train.py`)
Before a newly trained model can overwrite the production model, its ROC-AUC is compared against the best score in `models/registry.json`. If it's worse by more than a small tolerance, the versioned snapshot is still saved (for the record), but the production model is left untouched and the training run exits with an error. This means a bad retrain can never silently degrade what's live.

### 4. Experiment tracking (MLflow)
Every training run logs its hyperparameters, ROC-AUC, and per-threshold precision/recall/F1 to a local MLflow instance (SQLite-backed). View it with:
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

### 5. Automated retraining (`.github/workflows/retrain.yml`)
Runs on a schedule (weekly, configurable) or can be triggered manually from the Actions tab. It runs the full training pipeline in CI, and — if the validation gate passes — automatically commits the new model and pushes it to `main`, which in turn triggers Render to redeploy the backend with the updated model. This closes the loop from "new data available" to "safely live in production" with no manual steps required.

### 6. Drift monitoring (`src/app.py` → `GET /monitoring/drift`)
Every `/predict` request is logged (patient features + predicted probability) to a Postgres table. The `/monitoring/drift` endpoint pulls recent requests and compares their average values against the training data's baseline for each feature, using a standardized mean shift. Features that drift beyond a configurable threshold are flagged — a signal that incoming traffic may no longer resemble what the model was trained on.

---

## Running it yourself

### Backend
```bash
pip install -r requirements.txt
uvicorn src.app:app --reload
```
Visit `http://127.0.0.1:8000/docs` for interactive API docs.

### Frontend
```bash
pip install streamlit requests
streamlit run src/frontend.py
```

### Full training pipeline (with versioning, validation gate, MLflow, drift baseline)
```bash
python main.py
```

### Tests
```bash
pytest tests/ -v
```

### Environment variables (backend)
| Variable | Purpose |
|---|---|
| `DATABASE_URL` | Postgres connection string for prediction logging + drift monitoring |
| `PORT` | Set automatically by Render; defaults to `10000` locally |

---

## Model details

- **Algorithm:** XGBoost classifier, tuned for the minority class (readmitted <30 days) via `scale_pos_weight`
- **Explainability:** SHAP (TreeExplainer) — every prediction returns per-feature contribution values
- **Decision thresholds:** the API returns flags at three operational thresholds (0.45 / 0.50 / 0.55), letting downstream consumers choose their own precision/recall trade-off rather than baking in one cutoff

---

## Tech stack

**ML/Backend:** Python, XGBoost, scikit-learn, SHAP, FastAPI, pandas
**Frontend:** Streamlit
**Infra:** Docker, Render (web services + managed Postgres), GitHub Actions
**MLOps:** MLflow (experiment tracking), custom versioning/registry system, automated CI/CD

---

## Known limitations

- Drift detection uses a standardized mean-shift heuristic, not a full statistical test (e.g. Population Stability Index or KS-test) — a reasonable first line of monitoring, but not a substitute for more rigorous methods at larger scale.
- No staging environment — changes go through CI on `main` before deploying directly to production. For a team setting, a staging branch/environment would sit between CI and production deploy.
- Render's free-tier services spin down when idle, so the first request after inactivity may be slow (cold start).
