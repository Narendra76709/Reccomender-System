import os

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
DATA_DIR   = os.path.join(BASE_DIR, "..", "ml-25m")
MODELS_DIR = BASE_DIR
CACHE_DIR  = os.path.join(BASE_DIR, "cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# ── Dataset paths ──────────────────────────────────────────────────────────
RATINGS_PATH       = os.path.join(DATA_DIR, "ratings.csv")
MOVIES_PATH        = os.path.join(DATA_DIR, "movies.csv")
GENOME_SCORES_PATH = os.path.join(DATA_DIR, "genome-scores.csv")
GENOME_TAGS_PATH   = os.path.join(DATA_DIR, "genome-tags.csv")

# ── Trained model paths ────────────────────────────────────────────────────
NMF_MODEL_PATH  = os.path.join(MODELS_DIR, "nmf_model.pkl")
NMF_ANN_PATH    = os.path.join(MODELS_DIR, "nmf.ann")
SVD_MODEL_PATH  = os.path.join(MODELS_DIR, "svd_model.pkl")
SVD_ANN_PATH    = os.path.join(MODELS_DIR, "svd.ann")
LGBM_MODEL_PATH = os.path.join(MODELS_DIR, "lgbm_reranker.pkl")

# ── Stage-1 retrieval ──────────────────────────────────────────────────────
NUM_CANDIDATES = 200      # candidates after rank fusion
NMF_TOP_K      = 100      # from NMF-ANN generator
SVD_TOP_K      = 100      # from SVD-ANN generator
POP_TOP_K      = 60       # from popularity generator

GENERATOR_WEIGHTS = {
    "nmf_ann":    0.45,
    "svd_ann":    0.40,
    "popularity": 0.15,
}

# ── User classification thresholds ────────────────────────────────────────
COLD_MAX_RATINGS   = 5    # 0-5   → cold
LIGHT_MAX_RATINGS  = 20   # 6-20  → light
MEDIUM_MAX_RATINGS = 100  # 21-100 → medium
# > 100 → power

# ── Stage-2 re-ranking ─────────────────────────────────────────────────────
TOP_K_FINAL        = 10
POSITIVE_THRESHOLD = 3.5   # rating >= this → positive label

# ── NMF / SVD ─────────────────────────────────────────────────────────────
NMF_COMPONENTS = 20        # latent dimensions for NMF
SVD_COMPONENTS = 20        # latent dimensions for TruncatedSVD
ANN_METRIC     = "angular"
ANN_N_TREES    = 15

# ── LightGBM ──────────────────────────────────────────────────────────────
LGBM_PARAMS = {
    "objective":         "binary",
    "metric":            "auc",
    "n_estimators":      400,
    "learning_rate":     0.05,
    "num_leaves":        63,
    "min_child_samples": 20,
    "subsample":         0.8,
    "colsample_bytree":  0.8,
    "reg_alpha":         0.1,
    "reg_lambda":        0.1,
    "n_jobs":            -1,
    "verbose":           -1,
    "random_state":      42,
}

EVAL_K_LIST = [5, 10, 20]
