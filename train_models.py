"""
Train NMF and SVD models, build Annoy indexes.
Usage: python train_models.py [--sample_frac 0.1] [--nmf_components 20] [--svd_components 20]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import numpy as np
import scipy.sparse as sp
import joblib
from annoy import AnnoyIndex
from sklearn.decomposition import NMF, TruncatedSVD
from tqdm import tqdm

from config import (
    NMF_MODEL_PATH, NMF_ANN_PATH,
    SVD_MODEL_PATH, SVD_ANN_PATH,
    ANN_METRIC, ANN_N_TREES,
    NMF_COMPONENTS, SVD_COMPONENTS,
    CACHE_DIR
)
from data_loader import load_ratings, load_encodings


def build_user_item_matrix(ratings_df, user_enc, movie_enc):
    """Build sparse user-item matrix (users x items)."""
    print("Building sparse user-item matrix ...")
    n_users = len(user_enc)
    n_items = len(movie_enc)

    rows, cols, data = [], [], []
    for _, r in tqdm(ratings_df.iterrows(), total=len(ratings_df), desc="Matrix"):
        uid = r["userId"]
        mid = r["movieId"]
        if uid in user_enc and mid in movie_enc:
            rows.append(user_enc[uid])
            cols.append(movie_enc[mid])
            data.append(r["rating"])

    mat = sp.csr_matrix((data, (rows, cols)), shape=(n_users, n_items))
    print(f"  Matrix shape: {mat.shape}, nnz: {mat.nnz}")
    return mat


def train_nmf(item_user_mat, n_components):
    """Train NMF on item-user matrix (items as rows). Returns (model, item_factors)."""
    print(f"Training NMF (n_components={n_components}) ...")
    nmf = NMF(
        n_components=n_components,
        init="nndsvda",
        random_state=42,
        max_iter=200,
        tol=1e-4,
    )
    # fit_transform on item-user matrix returns W = item factors (n_items, n_components)
    item_factors = nmf.fit_transform(item_user_mat)
    print("  NMF training done.")
    return nmf, item_factors


def train_svd(user_item_mat, n_components):
    """Train TruncatedSVD on user-item matrix."""
    print(f"Training TruncatedSVD (n_components={n_components}) ...")
    svd = TruncatedSVD(n_components=n_components, random_state=42, n_iter=5)
    svd.fit(user_item_mat)
    print("  SVD training done.")
    return svd


def build_annoy_index(item_factors, n_components, ann_path, metric, n_trees):
    """
    Build Annoy index from item factor matrix.
    item_factors: numpy array of shape (n_items, n_components)
    """
    print(f"Building Annoy index ({item_factors.shape[0]} items, dim={n_components}) ...")
    ann = AnnoyIndex(n_components, metric)
    for i, vec in enumerate(tqdm(item_factors, desc="Annoy")):
        ann.add_item(i, vec.tolist())
    ann.build(n_trees)
    if os.path.exists(ann_path):
        os.remove(ann_path)
    ann.save(ann_path)
    print(f"  Annoy index saved to {ann_path}")
    return ann


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample_frac", type=float, default=0.1)
    parser.add_argument("--nmf_components", type=int, default=NMF_COMPONENTS)
    parser.add_argument("--svd_components", type=int, default=SVD_COMPONENTS)
    args = parser.parse_args()

    print(f"=== Training Models (sample_frac={args.sample_frac}) ===")

    # --- Load data ---
    ratings_df = load_ratings(sample_frac=args.sample_frac)
    print(f"Ratings loaded: {len(ratings_df):,} rows")

    encodings = load_encodings(ratings_df)
    user_enc, movie_enc, user_ids, movie_ids = encodings

    # --- Build matrix ---
    user_item_mat = build_user_item_matrix(ratings_df, user_enc, movie_enc)
    item_user_mat = user_item_mat.T.tocsr()  # (n_items, n_users)

    # --- NMF ---
    nmf_model, nmf_item_factors = train_nmf(item_user_mat, args.nmf_components)
    # nmf_item_factors = W matrix from fit_transform = (n_items, n_components)
    print(f"  NMF item factors shape: {nmf_item_factors.shape}")

    # Save NMF model + encodings + item factors together
    nmf_bundle = {
        "model": nmf_model,
        "encodings": encodings,
        "n_components": args.nmf_components,
        "item_factors": nmf_item_factors,  # (n_items, n_components)
    }
    joblib.dump(nmf_bundle, NMF_MODEL_PATH)
    print(f"  NMF model saved to {NMF_MODEL_PATH}")

    # Build NMF Annoy index
    build_annoy_index(nmf_item_factors, args.nmf_components, NMF_ANN_PATH, ANN_METRIC, ANN_N_TREES)

    # --- SVD ---
    svd_model = train_svd(user_item_mat, args.svd_components)
    # Item factors via SVD: transform item-user matrix transposed... actually
    # SVD on user-item: Vt = svd.components_, shape (n_components, n_items)
    # Item factors = Vt.T => (n_items, n_components)
    svd_item_factors = svd_model.components_.T  # (n_items, n_components)
    print(f"  SVD item factors shape: {svd_item_factors.shape}")

    svd_bundle = {
        "model": svd_model,
        "encodings": encodings,
        "n_components": args.svd_components,
    }
    joblib.dump(svd_bundle, SVD_MODEL_PATH)
    print(f"  SVD model saved to {SVD_MODEL_PATH}")

    # Build SVD Annoy index
    build_annoy_index(svd_item_factors, args.svd_components, SVD_ANN_PATH, ANN_METRIC, ANN_N_TREES)

    print("\n=== All models trained and saved successfully! ===")
    print(f"  NMF model: {NMF_MODEL_PATH}")
    print(f"  NMF Annoy: {NMF_ANN_PATH}")
    print(f"  SVD model: {SVD_MODEL_PATH}")
    print(f"  SVD Annoy: {SVD_ANN_PATH}")


if __name__ == "__main__":
    main()
