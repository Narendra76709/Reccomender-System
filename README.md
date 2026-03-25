# Hybrid Two-Stage Movie Recommendation System

A research-oriented recommendation system built on the **MovieLens 25M** dataset, implementing a two-stage pipeline: multi-generator candidate retrieval followed by LightGBM-based ranking.

---

## Architecture

```
Stage 1 — Multi-Generator Retrieval
├── Generator 1: NMF-ANN   (Non-negative Matrix Factorization + Approximate Nearest Neighbours)
├── Generator 2: SVD-ANN   (Truncated SVD + Approximate Nearest Neighbours)
└── Generator 3: Popularity (Bayesian Average Score)
         │
         ▼  Weighted Reciprocal Rank Fusion (wRRF)
         │
    200 Candidates
         │
Stage 2 — LightGBM Re-Ranker
         │  44 features: user stats · movie stats · retrieval scores · genome tag PCA-30
         ▼
    Top-10 Recommendations
```

### User Routing
| User Type | Rating Count | Generators Used |
|-----------|-------------|-----------------|
| Cold      | 0 – 5       | Popularity only |
| Light     | 6 – 20      | SVD-ANN + Popularity |
| Medium    | 21 – 100    | NMF-ANN + SVD-ANN + Popularity |
| Power     | > 100       | All generators  |

---

## Dataset

[MovieLens 25M](https://grouplens.org/datasets/movielens/25m/) — place files in `../ml-25m/`:
- `ratings.csv` — 25 million ratings
- `movies.csv` — 62,000 movies
- `genome-scores.csv` — tag relevance scores
- `genome-tags.csv` — tag labels

---

## Project Structure

```
movie-recommender/
├── config.py                      # All paths & hyperparameters
├── data_loader.py                 # Cached CSV loading
├── generators/
│   ├── nmf_ann_generator.py       # NMF latent space + Annoy ANN
│   ├── svd_ann_generator.py       # SVD latent space + Annoy ANN
│   ├── popularity_generator.py    # Bayesian popularity + genre boost
│   └── rank_fusion.py             # Weighted RRF
├── stage2_reranker.py             # LightGBM reranker (44 features)
├── train_models.py                # Train NMF + SVD + build Annoy indexes
├── train_reranker.py              # Train LightGBM reranker
├── recommender.py                 # HybridRecommender pipeline
├── api.py                         # FastAPI backend
├── frontend/index.html            # Web UI (dark-theme SPA)
├── evaluation.py                  # Offline metrics evaluation
├── run.py                         # Unified CLI entry point
└── requirements.txt
```

---

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Train Models
```bash
# Train NMF + SVD models and build Annoy indexes (uses 10% sample for speed)
python train_models.py --sample_frac 0.1

# Train LightGBM reranker
python train_reranker.py --n_users 500
```

### 3. Run Smoke Tests
```bash
python run.py test
```

### 4. Start Web Server
```bash
python run.py serve --port 8000
# Open http://localhost:8000
```

### 5. CLI Demo
```bash
python run.py demo --user_id 1
```

### 6. Evaluate
```bash
python run.py evaluate --n_users 200
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | Web UI frontend |
| GET | `/api/status` | System health & loaded models |
| GET | `/api/recommend?user_id=<n>` | Personalised top-10 recommendations |
| GET | `/api/search?q=<title>` | Movie search + content-based similar |
| GET | `/api/movie/<id>` | Movie details + similar movies |
| GET | `/api/popular?genre=<g>` | Trending movies (optional genre filter) |
| GET | `/api/genres` | All available genres |
| POST | `/api/train` | Trigger background reranker training |

---

## Features

- **Multi-generator retrieval** with weighted Reciprocal Rank Fusion
- **LightGBM reranker** using 44 features including genome tag PCA vectors
- **User-adaptive routing** based on rating history depth
- **Content-based fallback** using genre similarity
- **Dark-theme web UI** with autocomplete search, genre filters, and movie modals
- **Offline evaluation** with Precision@K, Recall@K, NDCG@K, HitRate@K, MRR

---

## Evaluation Metrics

The system is evaluated against two baselines:

| Metric | Description |
|--------|-------------|
| Precision@K | Fraction of top-K recommendations that are relevant |
| Recall@K | Fraction of relevant items captured in top-K |
| NDCG@K | Normalized Discounted Cumulative Gain |
| HitRate@K | 1 if any relevant item is in top-K |
| MRR | Mean Reciprocal Rank of first relevant item |

---

## Tech Stack

- **Python 3.8+**
- **scikit-learn** — NMF, TruncatedSVD
- **Annoy** — Approximate Nearest Neighbours
- **LightGBM** — Gradient boosted reranker
- **FastAPI + Uvicorn** — REST API
- **pandas / numpy / scipy** — Data processing

---

## References

- Harper, F. M., & Konstan, J. A. (2015). The MovieLens Datasets. *ACM TIIS*.
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NeurIPS*.
- Bernhardsson, E. (2018). Annoy: Approximate Nearest Neighbors in C++/Python.
