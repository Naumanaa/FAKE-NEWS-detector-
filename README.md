# Fake News Detector

An end-to-end machine learning project — from dataset to a live REST API —
for classifying short political statements as **fake** or **real**.
Built summer 2026.

## Dataset

Trained on **LIAR-PLUS** (Alhindi, Petridis & Muresan, 2018), an extended
version of the LIAR benchmark (Wang, 2017): ~12.8K short statements from
PolitiFact, each hand-labeled on a 6-point truthfulness scale
(`pants-fire`, `false`, `barely-true`, `half-true`, `mostly-true`, `true`)
along with speaker/context metadata and a justification sentence.

For this v1 baseline, the 6-way label is collapsed into binary:

| Bucket | Original labels |
|---|---|
| **REAL** (1) | true, mostly-true, half-true |
| **FAKE** (0) | barely-true, false, pants-fire |

## Features

- **TF-IDF + Random Forest** baseline classifier (Scikit-learn)
- **Data pipeline** for loading/cleaning the LIAR-PLUS tsv splits (Pandas)
- **Production server** via FastAPI, serving predictions over REST
- **Docker** deployment image
- **Tests** via PyTest covering the data loader, featurizer, and API

## Project layout

```
fake_news/
  data/loader.py         # loads + cleans the LIAR-PLUS tsv splits
  features/featurizer.py # TF-IDF vectorizer wrapper
  models/tree_based.py   # random forest model wrapper
  server/main.py         # FastAPI app
scripts/
  download_data.py       # fetches LIAR-PLUS into data/raw
  train.py                # trains + evaluates + checkpoints the model
tests/
  test_pipeline.py
deploy/
  Dockerfile.serve
```

## How to use it

Install dependencies:

```
pip install -r requirements.txt
```

Download the dataset:

```
python scripts/download_data.py
```

### Train

```
python scripts/train.py
```

Sample output:

```
INFO - Loading LIAR-PLUS dataset...
INFO - train=10269 rows, val=1284 rows
INFO - Fitting TF-IDF featurizer...
INFO - Training random forest model...
INFO - Evaluating on validation set...
INFO - Val metrics: {'val_accuracy': 0.611, 'val_f1': 0.662, 'val_auc': 0.657, ...}
```

This saves `model.joblib` and `featurizer.joblib` to
`model_checkpoints/random_forest/`.

### Serve locally

```
uvicorn fake_news.server.main:app --reload
```

```
curl -X POST http://127.0.0.1:8000/api/predict-fakeness \
  -H "Content-Type: application/json" \
  -d '{"text": "The moon landing was staged in a Hollywood studio."}'
```

### Deploy with Docker

```
docker build . -f deploy/Dockerfile.serve -t fake-news-deploy
docker run -p 8000:80 fake-news-deploy
```

### Test

```
pytest tests/ -v
```

## Roadmap

- [ ] Swap in a transformer (RoBERTa) model for a v2 accuracy bump
- [ ] Add SHAP-based error/feature analysis
- [ ] Add MLFlow experiment tracking
- [ ] Add CI via GitHub Actions
- [ ] Browser extension front-end

## License

AGPL-3.0
