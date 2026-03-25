"""
Train the Stage 2 LightGBM reranker.
Usage: python train_reranker.py [--n_users 2000]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import joblib
from annoy import AnnoyIndex

from config import (
    NMF_MODEL_PATH, NMF_ANN_PATH,
    SVD_MODEL_PATH, SVD_ANN_PATH,
    ANN_METRIC, LGBM_MODEL_PATH
)
from data_loader import (
    load_ratings, load_movies, load_genome_scores,
    load_user_stats, load_movie_stats, load_popularity_scores
)
from generators.nmf_ann_generator import NMFANNGenerator
from generators.svd_ann_generator import SVDANNGenerator
from generators.popularity_generator import PopularityGenerator
from stage2_reranker import Stage2Reranker


def load_generators(ratings_df, movies_df):
    generators = {}

    # --- NMF generator ---
    if os.path.exists(NMF_MODEL_PATH) and os.path.exists(NMF_ANN_PATH):
        print("Loading NMF model ...")
        nmf_bundle = joblib.load(NMF_MODEL_PATH)
        nmf_model = nmf_bundle["model"]
        nmf_enc = nmf_bundle["encodings"]
        n_comp = nmf_bundle["n_components"]
        nmf_ann = AnnoyIndex(n_comp, ANN_METRIC)
        nmf_ann.load(NMF_ANN_PATH)
        nmf_item_factors = nmf_bundle.get("item_factors")
        generators["nmf"] = NMFANNGenerator(nmf_model, nmf_ann, n_comp, nmf_enc, ratings_df, nmf_item_factors)
        print("  NMF generator loaded.")
    else:
        print("WARNING: NMF model not found. Run train_models.py first.")

    # --- SVD generator ---
    if os.path.exists(SVD_MODEL_PATH) and os.path.exists(SVD_ANN_PATH):
        print("Loading SVD model ...")
        svd_bundle = joblib.load(SVD_MODEL_PATH)
        svd_model = svd_bundle["model"]
        svd_enc = svd_bundle["encodings"]
        n_comp = svd_bundle["n_components"]
        svd_ann = AnnoyIndex(n_comp, ANN_METRIC)
        svd_ann.load(SVD_ANN_PATH)
        generators["svd"] = SVDANNGenerator(svd_model, svd_ann, n_comp, svd_enc, ratings_df)
        print("  SVD generator loaded.")
    else:
        print("WARNING: SVD model not found. Run train_models.py first.")

    # --- Popularity generator ---
    pop_scores = load_popularity_scores(ratings_df)
    generators["popularity"] = PopularityGenerator(pop_scores, movies_df, ratings_df)
    print("  Popularity generator loaded.")

    return generators


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=2000)
    args = parser.parse_args()

    print(f"=== Training Stage 2 Reranker (n_users={args.n_users}) ===")

    # Load data
    ratings_df = load_ratings(sample_frac=0.1)
    movies_df = load_movies()
    genome_scores_df = load_genome_scores()
    user_stats_df = load_user_stats(ratings_df)
    movie_stats_df = load_movie_stats(ratings_df)

    # Load generators
    generators = load_generators(ratings_df, movies_df)

    # Train reranker
    reranker = Stage2Reranker()
    reranker.train(
        ratings_df=ratings_df,
        genome_scores_df=genome_scores_df,
        n_users=args.n_users,
        generators=generators,
        user_stats_df=user_stats_df,
        movie_stats_df=movie_stats_df,
    )

    reranker.save(LGBM_MODEL_PATH)
    print(f"\n=== Reranker saved to {LGBM_MODEL_PATH} ===")


if __name__ == "__main__":
    main()
