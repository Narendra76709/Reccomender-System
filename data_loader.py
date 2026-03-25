import sys
import os
import pickle
import pandas as pd
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from config import (
    RATINGS_PATH, MOVIES_PATH, GENOME_SCORES_PATH, GENOME_TAGS_PATH,
    CACHE_DIR, COLD_MAX_RATINGS, LIGHT_MAX_RATINGS, MEDIUM_MAX_RATINGS
)

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(name):
    return os.path.join(CACHE_DIR, name + ".pkl")


def _load_cache(name):
    p = _cache_path(name)
    if os.path.exists(p):
        with open(p, "rb") as f:
            return pickle.load(f)
    return None


def _save_cache(name, obj):
    with open(_cache_path(name), "wb") as f:
        pickle.dump(obj, f)


def load_movies():
    cached = _load_cache("movies")
    if cached is not None:
        return cached
    print("Loading movies.csv ...")
    df = pd.read_csv(MOVIES_PATH)
    _save_cache("movies", df)
    return df


def load_ratings(sample_frac=1.0):
    cache_name = f"ratings_{sample_frac:.4f}"
    cached = _load_cache(cache_name)
    if cached is not None:
        return cached
    print(f"Loading ratings.csv (sample_frac={sample_frac}) ...")
    df = pd.read_csv(RATINGS_PATH)
    if sample_frac < 1.0:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    _save_cache(cache_name, df)
    return df


def load_genome_scores():
    cached = _load_cache("genome_scores")
    if cached is not None:
        return cached
    print("Loading genome-scores.csv ...")
    df = pd.read_csv(GENOME_SCORES_PATH)
    _save_cache("genome_scores", df)
    return df


def load_user_stats(ratings_df=None):
    cached = _load_cache("user_stats")
    if cached is not None:
        return cached
    if ratings_df is None:
        ratings_df = load_ratings()
    print("Computing user stats ...")
    stats = ratings_df.groupby("userId").agg(
        n_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
        min_rating=("rating", "min"),
        max_rating=("rating", "max"),
    ).fillna(0)
    _save_cache("user_stats", stats)
    return stats


def load_movie_stats(ratings_df=None):
    cached = _load_cache("movie_stats")
    if cached is not None:
        return cached
    if ratings_df is None:
        ratings_df = load_ratings()
    print("Computing movie stats ...")
    stats = ratings_df.groupby("movieId").agg(
        n_ratings=("rating", "count"),
        mean_rating=("rating", "mean"),
        std_rating=("rating", "std"),
    ).fillna(0)
    _save_cache("movie_stats", stats)
    return stats


def load_encodings(ratings_df=None):
    """
    Returns (user_enc, movie_enc, user_ids, movie_ids).
    user_enc: dict userId -> row index
    movie_enc: dict movieId -> col index
    """
    cached = _load_cache("encodings")
    if cached is not None:
        return cached
    if ratings_df is None:
        ratings_df = load_ratings()
    print("Building encodings ...")
    user_ids = sorted(ratings_df["userId"].unique())
    movie_ids = sorted(ratings_df["movieId"].unique())
    user_enc = {u: i for i, u in enumerate(user_ids)}
    movie_enc = {m: i for i, m in enumerate(movie_ids)}
    result = (user_enc, movie_enc, user_ids, movie_ids)
    _save_cache("encodings", result)
    return result


def load_popularity_scores(ratings_df=None):
    """Bayesian average popularity score per movie."""
    cached = _load_cache("popularity_scores")
    if cached is not None:
        return cached
    if ratings_df is None:
        ratings_df = load_ratings()
    print("Computing popularity scores ...")
    movie_stats = load_movie_stats(ratings_df)
    C = movie_stats["n_ratings"].mean()
    m = movie_stats["mean_rating"].mean()
    movie_stats["pop_score"] = (
        (movie_stats["n_ratings"] * movie_stats["mean_rating"] + C * m)
        / (movie_stats["n_ratings"] + C)
    )
    scores = movie_stats["pop_score"].to_dict()
    _save_cache("popularity_scores", scores)
    return scores


def classify_user(user_id, user_stats_df):
    """Return 'cold', 'light', 'medium', or 'power'."""
    if user_id not in user_stats_df.index:
        return "cold"
    n = user_stats_df.loc[user_id, "n_ratings"]
    if n <= COLD_MAX_RATINGS:
        return "cold"
    elif n <= LIGHT_MAX_RATINGS:
        return "light"
    elif n <= MEDIUM_MAX_RATINGS:
        return "medium"
    else:
        return "power"


def get_user_rated_movies(user_id, ratings_df):
    """Return set of movieIds rated by user."""
    user_rows = ratings_df[ratings_df["userId"] == user_id]
    return set(user_rows["movieId"].tolist())
