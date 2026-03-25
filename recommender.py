"""
HybridRecommender: Two-stage recommendation pipeline.
Stage 1: Multi-generator retrieval + weighted RRF fusion -> 200 candidates
Stage 2: LightGBM reranker -> Top-10
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import joblib
import numpy as np
import pandas as pd
from annoy import AnnoyIndex

from config import (
    NMF_MODEL_PATH, NMF_ANN_PATH,
    SVD_MODEL_PATH, SVD_ANN_PATH,
    LGBM_MODEL_PATH,
    ANN_METRIC,
    NUM_CANDIDATES, NMF_TOP_K, SVD_TOP_K, POP_TOP_K,
    GENERATOR_WEIGHTS, TOP_K_FINAL,
)
from data_loader import (
    load_movies, load_ratings, load_user_stats, load_movie_stats,
    load_popularity_scores, classify_user, get_user_rated_movies
)
from generators.nmf_ann_generator import NMFANNGenerator
from generators.svd_ann_generator import SVDANNGenerator
from generators.popularity_generator import PopularityGenerator
from generators.rank_fusion import weighted_rrf
from stage2_reranker import Stage2Reranker


class HybridRecommender:

    def __init__(self, ratings_df=None, verbose=True):
        self.verbose = verbose
        self._log("Loading data ...")

        self.movies_df = load_movies()
        self.ratings_df = ratings_df if ratings_df is not None else load_ratings(sample_frac=0.1)
        self.user_stats_df = load_user_stats(self.ratings_df)
        self.movie_stats_df = load_movie_stats(self.ratings_df)
        self.pop_scores = load_popularity_scores(self.ratings_df)

        # Movie lookup: movieId -> row
        self.movie_lookup = self.movies_df.set_index("movieId").to_dict("index")

        # Load generators
        self._load_generators()

        # Load reranker (optional)
        self.reranker = None
        if os.path.exists(LGBM_MODEL_PATH):
            self._log("Loading LightGBM reranker ...")
            self.reranker = Stage2Reranker()
            self.reranker.load(LGBM_MODEL_PATH)
            self._log("  Reranker loaded.")
        else:
            self._log("No reranker found. Will use RRF scores only.")

        self._log("HybridRecommender ready.")

    def _log(self, msg):
        if self.verbose:
            print(msg)

    def _load_generators(self):
        self.nmf_gen = None
        self.svd_gen = None

        if os.path.exists(NMF_MODEL_PATH) and os.path.exists(NMF_ANN_PATH):
            self._log("Loading NMF generator ...")
            nmf_bundle = joblib.load(NMF_MODEL_PATH)
            nmf_ann = AnnoyIndex(nmf_bundle["n_components"], ANN_METRIC)
            nmf_ann.load(NMF_ANN_PATH)
            self.nmf_gen = NMFANNGenerator(
                nmf_bundle["model"], nmf_ann,
                nmf_bundle["n_components"],
                nmf_bundle["encodings"],
                self.ratings_df,
                nmf_bundle.get("item_factors"),
            )

        if os.path.exists(SVD_MODEL_PATH) and os.path.exists(SVD_ANN_PATH):
            self._log("Loading SVD generator ...")
            svd_bundle = joblib.load(SVD_MODEL_PATH)
            svd_ann = AnnoyIndex(svd_bundle["n_components"], ANN_METRIC)
            svd_ann.load(SVD_ANN_PATH)
            self.svd_gen = SVDANNGenerator(
                svd_bundle["model"], svd_ann,
                svd_bundle["n_components"],
                svd_bundle["encodings"],
                self.ratings_df
            )

        self.pop_gen = PopularityGenerator(self.pop_scores, self.movies_df, self.ratings_df)

    def recommend(self, user_id, top_k=TOP_K_FINAL, return_details=False):
        """
        Main recommendation pipeline.
        Returns list of dicts with movie info and scores.
        """
        user_type = classify_user(user_id, self.user_stats_df)
        exclude_ids = get_user_rated_movies(user_id, self.ratings_df)

        # --- Stage 1: Generate candidates ---
        ranked_lists = []
        weights = []
        nmf_list, svd_list, pop_list = [], [], []

        if user_type == "cold":
            pop_list = self.pop_gen.recommend(user_id, exclude_ids, top_k=POP_TOP_K)
            ranked_lists = [pop_list]
            weights = [1.0]

        elif user_type == "light":
            if self.svd_gen and self.svd_gen.is_available():
                svd_list = self.svd_gen.recommend(user_id, exclude_ids, top_k=SVD_TOP_K)
                ranked_lists.append(svd_list)
                weights.append(GENERATOR_WEIGHTS["svd_ann"])
            pop_list = self.pop_gen.recommend(user_id, exclude_ids, top_k=POP_TOP_K)
            ranked_lists.append(pop_list)
            weights.append(GENERATOR_WEIGHTS["popularity"])

        else:  # medium or power
            if self.nmf_gen and self.nmf_gen.is_available():
                nmf_list = self.nmf_gen.recommend(user_id, exclude_ids, top_k=NMF_TOP_K)
                ranked_lists.append(nmf_list)
                weights.append(GENERATOR_WEIGHTS["nmf_ann"])
            if self.svd_gen and self.svd_gen.is_available():
                svd_list = self.svd_gen.recommend(user_id, exclude_ids, top_k=SVD_TOP_K)
                ranked_lists.append(svd_list)
                weights.append(GENERATOR_WEIGHTS["svd_ann"])
            pop_list = self.pop_gen.recommend(user_id, exclude_ids, top_k=POP_TOP_K)
            ranked_lists.append(pop_list)
            weights.append(GENERATOR_WEIGHTS["popularity"])

        if not ranked_lists:
            return []

        # Normalize weights
        total_w = sum(weights)
        weights = [w / total_w for w in weights]

        # Weighted RRF fusion
        candidates = weighted_rrf(ranked_lists, weights, top_n=NUM_CANDIDATES)

        if not candidates:
            return []

        # --- Stage 2: Rerank ---
        if self.reranker is not None:
            nmf_scores = {mid: sc for mid, sc in nmf_list}
            svd_scores = {mid: sc for mid, sc in svd_list}
            pop_score_dict = {mid: sc for mid, sc in pop_list}
            final = self.reranker.rerank(
                user_id, candidates,
                self.user_stats_df, self.movie_stats_df,
                nmf_scores, svd_scores, pop_score_dict,
                top_k=top_k
            )
        else:
            final = candidates[:top_k]

        # Format results
        results = []
        for movie_id, score in final:
            info = self.movie_lookup.get(movie_id, {})
            result = {
                "movieId": movie_id,
                "title": info.get("title", f"Movie {movie_id}"),
                "genres": info.get("genres", ""),
                "score": round(float(score), 4),
                "user_type": user_type,
            }
            if return_details:
                stats = self.movie_stats_df.loc[movie_id] if movie_id in self.movie_stats_df.index else None
                result["n_ratings"] = int(stats["n_ratings"]) if stats is not None else 0
                result["mean_rating"] = round(float(stats["mean_rating"]), 2) if stats is not None else 0.0
            results.append(result)

        return results

    def recommend_by_movie(self, title, top_k=10):
        """
        Content-based: find similar movies using NMF/SVD Annoy item-item similarity.
        Falls back to genre matching if movie not in ANN index.
        """
        mask = self.movies_df["title"].str.contains(title, case=False, na=False, regex=False)
        matches = self.movies_df[mask]
        if matches.empty:
            return []

        seed_row = matches.iloc[0]
        seed_id = int(seed_row["movieId"])

        # Build genre set for the seed movie
        seed_genres_str = seed_row["genres"] if pd.notna(seed_row.get("genres")) else ""
        seed_genres = set(seed_genres_str.split("|")) if seed_genres_str else set()

        # --- NMF-ANN: get broader candidate pool, then re-rank with genre ---
        ann_candidates = {}  # mid -> ann_score
        gen = self.nmf_gen if (self.nmf_gen and self.nmf_gen.is_available()) else (
              self.svd_gen if (self.svd_gen and self.svd_gen.is_available()) else None)

        if gen is not None:
            item_idx = gen.movie_enc.get(seed_id)
            if item_idx is not None:
                try:
                    n_fetch = max(top_k * 10, 100)
                    ann_idxs, dists = gen.ann.get_nns_by_item(
                        item_idx, n_fetch, include_distances=True
                    )
                    for idx, dist in zip(ann_idxs, dists):
                        if idx >= len(gen.movie_id_arr):
                            continue
                        mid = int(gen.movie_id_arr[idx])
                        if mid != seed_id:
                            ann_candidates[mid] = 1.0 / (1.0 + dist)
                except Exception:
                    pass

        # --- Score candidates: blend ANN similarity + genre overlap ---
        results = []
        pool = ann_candidates if ann_candidates else {
            int(row["movieId"]): 0.0
            for _, row in self.movies_df.iterrows()
            if row["movieId"] != seed_id
        }

        for mid, ann_score in pool.items():
            info = self.movie_lookup.get(mid, {})
            genres_str = info.get("genres", "")
            genres = set(genres_str.split("|")) if genres_str else set()
            genre_sim = len(seed_genres & genres) / max(len(seed_genres | genres), 1) if seed_genres else 0
            score = 0.6 * ann_score + 0.4 * genre_sim
            if score > 0:
                results.append({
                    "movieId": mid,
                    "title": info.get("title", f"Movie {mid}"),
                    "genres": genres_str,
                    "score": round(score, 4),
                    "method": "NMF-ANN+genre" if ann_candidates else "genre",
                })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_movie_info(self, movie_id):
        info = self.movie_lookup.get(movie_id)
        if not info:
            return None
        stats = self.movie_stats_df.loc[movie_id] if movie_id in self.movie_stats_df.index else None
        return {
            "movieId": movie_id,
            "title": info.get("title", ""),
            "genres": info.get("genres", ""),
            "n_ratings": int(stats["n_ratings"]) if stats is not None else 0,
            "mean_rating": round(float(stats["mean_rating"]), 2) if stats is not None else 0.0,
            "pop_score": round(self.pop_scores.get(movie_id, 0.0), 4),
        }

    def get_popular_movies(self, limit=20, genre=None):
        """Return top popular movies, optionally filtered by genre."""
        results = []
        sorted_pop = sorted(self.pop_scores.items(), key=lambda x: x[1], reverse=True)

        for mid, score in sorted_pop:
            info = self.movie_lookup.get(mid, {})
            genres_str = info.get("genres", "")
            if genre and genre.lower() not in genres_str.lower():
                continue
            results.append({
                "movieId": mid,
                "title": info.get("title", f"Movie {mid}"),
                "genres": genres_str,
                "score": round(float(score), 4),
            })
            if len(results) >= limit:
                break

        return results

    def get_all_genres(self):
        """Return sorted list of all genres."""
        genres = set()
        for _, row in self.movies_df.iterrows():
            if pd.notna(row.get("genres")):
                for g in row["genres"].split("|"):
                    if g and g != "(no genres listed)":
                        genres.add(g)
        return sorted(genres)
