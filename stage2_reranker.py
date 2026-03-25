"""
Stage 2: LightGBM reranker.
Features: user_stats(5) + movie_stats(3) + retrieval_scores/ranks(6) + genome_pca(30) = 44 features.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import numpy as np
import pandas as pd
import joblib
from sklearn.decomposition import TruncatedSVD

from config import LGBM_MODEL_PATH, LGBM_PARAMS, POSITIVE_THRESHOLD


GENOME_PCA_COMPONENTS = 30


class Stage2Reranker:

    def __init__(self):
        self.model = None
        self.genome_pca = None
        self.genome_movie_ids = None
        self.genome_matrix = None
        self.feature_names = None

    # ------------------------------------------------------------------
    # Genome PCA setup
    # ------------------------------------------------------------------

    def fit_genome_pca(self, genome_scores_df):
        """Fit TruncatedSVD on genome scores, return item-factor matrix."""
        print("  Fitting genome PCA ...")
        pivot = genome_scores_df.pivot(index="movieId", columns="tagId", values="relevance").fillna(0)
        self.genome_movie_ids = list(pivot.index)

        n_comp = min(GENOME_PCA_COMPONENTS, pivot.shape[1] - 1)
        self.genome_pca = TruncatedSVD(n_components=n_comp, random_state=42)
        self.genome_matrix = self.genome_pca.fit_transform(pivot.values)
        # genome_matrix: (n_genome_movies, n_comp)
        self.genome_id_to_idx = {mid: i for i, mid in enumerate(self.genome_movie_ids)}
        print(f"  Genome PCA shape: {self.genome_matrix.shape}")

    def _get_genome_vec(self, movie_id):
        idx = self.genome_id_to_idx.get(movie_id)
        if idx is None:
            return np.zeros(GENOME_PCA_COMPONENTS)
        vec = self.genome_matrix[idx]
        # Pad if needed
        if len(vec) < GENOME_PCA_COMPONENTS:
            vec = np.pad(vec, (0, GENOME_PCA_COMPONENTS - len(vec)))
        return vec

    # ------------------------------------------------------------------
    # Feature building
    # ------------------------------------------------------------------

    def build_features(self, user_id, candidates, user_stats_df, movie_stats_df,
                        nmf_scores, svd_scores, pop_scores):
        """
        Build feature matrix for (user, candidate movie) pairs.

        candidates: list of movieIds
        nmf_scores, svd_scores, pop_scores: dict movieId -> score
        Returns: np.ndarray of shape (len(candidates), 44)
        """
        rows = []

        # User features
        if user_id in user_stats_df.index:
            u = user_stats_df.loc[user_id]
            u_feats = [
                float(u["n_ratings"]),
                float(u["mean_rating"]),
                float(u["std_rating"]),
                float(u["min_rating"]),
                float(u["max_rating"]),
            ]
        else:
            u_feats = [0.0, 0.0, 0.0, 0.0, 0.0]

        for mid in candidates:
            # Movie stats
            if mid in movie_stats_df.index:
                m = movie_stats_df.loc[mid]
                m_feats = [
                    float(m["n_ratings"]),
                    float(m["mean_rating"]),
                    float(m["std_rating"]),
                ]
            else:
                m_feats = [0.0, 0.0, 0.0]

            # Retrieval scores and ranks
            nmf_score = nmf_scores.get(mid, 0.0)
            svd_score = svd_scores.get(mid, 0.0)
            pop_score = pop_scores.get(mid, 0.0)

            # Ranks (lower is better; 0 means not in list)
            nmf_rank = list(nmf_scores.keys()).index(mid) + 1 if mid in nmf_scores else 0
            svd_rank = list(svd_scores.keys()).index(mid) + 1 if mid in svd_scores else 0
            pop_rank = list(pop_scores.keys()).index(mid) + 1 if mid in pop_scores else 0

            ret_feats = [nmf_score, svd_score, pop_score,
                         float(nmf_rank), float(svd_rank), float(pop_rank)]

            # Genome PCA features
            genome_feats = self._get_genome_vec(mid).tolist()

            row = u_feats + m_feats + ret_feats + genome_feats
            rows.append(row)

        return np.array(rows, dtype=np.float32)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, ratings_df, genome_scores_df, n_users, generators, user_stats_df, movie_stats_df):
        """
        Build training samples from n_users, then fit LightGBM.
        generators: dict with keys 'nmf', 'svd', 'popularity'
        """
        from lightgbm import LGBMClassifier

        self.fit_genome_pca(genome_scores_df)

        print(f"  Building training samples for {n_users} users ...")
        sample_users = ratings_df["userId"].unique()
        np.random.seed(42)
        if len(sample_users) > n_users:
            sample_users = np.random.choice(sample_users, n_users, replace=False)

        all_X, all_y = [], []

        for i, user_id in enumerate(sample_users):
            if (i + 1) % 100 == 0:
                print(f"    Processing user {i+1}/{len(sample_users)} ...")

            # Get ground truth: movies rated >= threshold
            user_rows = ratings_df[ratings_df["userId"] == user_id]
            if len(user_rows) < 5:
                continue

            # Hold out 20% for labels
            hold_out = user_rows.sample(frac=0.2, random_state=42)
            train_rows = user_rows.drop(hold_out.index)

            positive_ids = set(
                hold_out[hold_out["rating"] >= POSITIVE_THRESHOLD]["movieId"].tolist()
            )
            if not positive_ids:
                continue

            exclude_ids = set(train_rows["movieId"].tolist())

            # Generate candidates
            nmf_gen = generators.get("nmf")
            svd_gen = generators.get("svd")
            pop_gen = generators.get("popularity")

            nmf_list = nmf_gen.recommend(user_id, exclude_ids, top_k=80) if nmf_gen and nmf_gen.is_available() else []
            svd_list = svd_gen.recommend(user_id, exclude_ids, top_k=80) if svd_gen and svd_gen.is_available() else []
            pop_list = pop_gen.recommend(user_id, exclude_ids, top_k=40) if pop_gen and pop_gen.is_available() else []

            # Merge candidates
            candidate_set = set()
            for mid, _ in nmf_list + svd_list + pop_list:
                candidate_set.add(mid)
            candidates = list(candidate_set)

            if not candidates:
                continue

            # Build score dicts for feature creation
            nmf_scores = {mid: sc for mid, sc in nmf_list}
            svd_scores = {mid: sc for mid, sc in svd_list}
            pop_score_dict = {mid: sc for mid, sc in pop_list}

            X = self.build_features(
                user_id, candidates, user_stats_df, movie_stats_df,
                nmf_scores, svd_scores, pop_score_dict
            )
            y = np.array([1 if mid in positive_ids else 0 for mid in candidates])

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            print("  No training samples generated!")
            return

        X_train = np.vstack(all_X)
        y_train = np.concatenate(all_y)
        print(f"  Training LightGBM on {len(X_train):,} samples (pos={y_train.sum():.0f}) ...")

        self.model = LGBMClassifier(**LGBM_PARAMS)
        self.model.fit(X_train, y_train)
        print("  LightGBM training done.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def rerank(self, user_id, candidates, user_stats_df, movie_stats_df,
               nmf_scores, svd_scores, pop_scores, top_k=10):
        """
        Rerank candidates using LightGBM.
        Returns list of (movieId, score) top_k items.
        """
        if self.model is None or not candidates:
            # Fall back to RRF scores
            return candidates[:top_k]

        candidate_ids = [mid for mid, _ in candidates]
        X = self.build_features(
            user_id, candidate_ids, user_stats_df, movie_stats_df,
            nmf_scores, svd_scores, pop_scores
        )
        probs = self.model.predict_proba(X)[:, 1]
        ranked = sorted(zip(candidate_ids, probs), key=lambda x: x[1], reverse=True)
        return ranked[:top_k]

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------

    def save(self, path=None):
        if path is None:
            path = LGBM_MODEL_PATH
        bundle = {
            "model": self.model,
            "genome_pca": self.genome_pca,
            "genome_movie_ids": self.genome_movie_ids,
            "genome_matrix": self.genome_matrix,
            "genome_id_to_idx": self.genome_id_to_idx,
        }
        joblib.dump(bundle, path)
        print(f"  Reranker saved to {path}")

    def load(self, path=None):
        if path is None:
            path = LGBM_MODEL_PATH
        bundle = joblib.load(path)
        self.model = bundle["model"]
        self.genome_pca = bundle["genome_pca"]
        self.genome_movie_ids = bundle["genome_movie_ids"]
        self.genome_matrix = bundle["genome_matrix"]
        self.genome_id_to_idx = bundle["genome_id_to_idx"]
        return self
