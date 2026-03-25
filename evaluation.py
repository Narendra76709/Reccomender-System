"""
Offline evaluation: Precision@K, Recall@K, NDCG@K, HitRate@K, MRR.
Compares: Hybrid vs Popularity baseline vs Random baseline.
Usage: python evaluation.py [--n_users 200]
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import argparse
import numpy as np
import random
from collections import defaultdict

from config import POSITIVE_THRESHOLD, EVAL_K_LIST
from data_loader import load_ratings, load_movies, load_popularity_scores, load_user_stats


# ── Metrics ────────────────────────────────────────────────────────────

def precision_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = sum(1 for m in rec_k if m in relevant)
    return hits / k if k > 0 else 0.0


def recall_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    hits = sum(1 for m in rec_k if m in relevant)
    return hits / len(relevant) if relevant else 0.0


def ndcg_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    dcg = sum(
        1.0 / np.log2(i + 2)
        for i, m in enumerate(rec_k)
        if m in relevant
    )
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0


def hit_rate_at_k(recommended, relevant, k):
    rec_k = recommended[:k]
    return 1.0 if any(m in relevant for m in rec_k) else 0.0


def mrr(recommended, relevant):
    for i, m in enumerate(recommended, start=1):
        if m in relevant:
            return 1.0 / i
    return 0.0


# ── Baselines ──────────────────────────────────────────────────────────

def popularity_baseline(pop_scores, exclude_ids, top_k=20):
    sorted_pop = sorted(pop_scores.items(), key=lambda x: x[1], reverse=True)
    return [mid for mid, _ in sorted_pop if mid not in exclude_ids][:top_k]


def random_baseline(all_movie_ids, exclude_ids, top_k=20, seed=42):
    candidates = [m for m in all_movie_ids if m not in exclude_ids]
    random.seed(seed)
    random.shuffle(candidates)
    return candidates[:top_k]


# ── Evaluation loop ────────────────────────────────────────────────────

def evaluate_system(recommender, ratings_df, pop_scores, all_movie_ids, n_users, k_list):
    sample_users = ratings_df["userId"].unique()
    np.random.seed(42)
    if len(sample_users) > n_users:
        sample_users = np.random.choice(sample_users, n_users, replace=False)

    metrics = {
        "hybrid":     defaultdict(list),
        "popularity": defaultdict(list),
        "random":     defaultdict(list),
    }

    evaluated = 0
    for i, user_id in enumerate(sample_users):
        if (i + 1) % 50 == 0:
            print(f"  Evaluating user {i+1}/{len(sample_users)} ...")

        user_rows = ratings_df[ratings_df["userId"] == user_id]
        if len(user_rows) < 5:
            continue

        # Hold out 20% as ground truth
        hold_out = user_rows.sample(frac=0.2, random_state=42)
        train_rows = user_rows.drop(hold_out.index)

        relevant = set(
            hold_out[hold_out["rating"] >= POSITIVE_THRESHOLD]["movieId"].tolist()
        )
        if not relevant:
            continue

        exclude_ids = set(train_rows["movieId"].tolist())

        # Hybrid
        try:
            hybrid_recs = recommender.recommend(user_id, top_k=max(k_list), return_details=False)
            hybrid_ids = [r["movieId"] for r in hybrid_recs]
        except Exception:
            hybrid_ids = []

        # Popularity baseline
        pop_ids = popularity_baseline(pop_scores, exclude_ids, top_k=max(k_list))

        # Random baseline
        rand_ids = random_baseline(all_movie_ids, exclude_ids, top_k=max(k_list))

        for k in k_list:
            for name, rec_list in [("hybrid", hybrid_ids), ("popularity", pop_ids), ("random", rand_ids)]:
                metrics[name][f"P@{k}"].append(precision_at_k(rec_list, relevant, k))
                metrics[name][f"R@{k}"].append(recall_at_k(rec_list, relevant, k))
                metrics[name][f"NDCG@{k}"].append(ndcg_at_k(rec_list, relevant, k))
                metrics[name][f"HR@{k}"].append(hit_rate_at_k(rec_list, relevant, k))
            for name, rec_list in [("hybrid", hybrid_ids), ("popularity", pop_ids), ("random", rand_ids)]:
                metrics[name]["MRR"].append(mrr(rec_list, relevant))

        evaluated += 1

    return metrics, evaluated


def print_results(metrics, k_list, n_evaluated):
    print(f"\n{'='*65}")
    print(f"  Evaluation Results (n_users={n_evaluated})")
    print(f"{'='*65}")

    systems = ["hybrid", "popularity", "random"]
    col_w = 14

    # Header
    metric_keys = [f"P@{k}" for k in k_list] + [f"R@{k}" for k in k_list] + \
                  [f"NDCG@{k}" for k in k_list] + [f"HR@{k}" for k in k_list] + ["MRR"]

    print(f"\n{'System':<14}", end="")
    for mk in metric_keys:
        print(f"{mk:>{col_w}}", end="")
    print()
    print("-" * (14 + col_w * len(metric_keys)))

    for sys_name in systems:
        print(f"{sys_name:<14}", end="")
        for mk in metric_keys:
            vals = metrics[sys_name].get(mk, [])
            avg = np.mean(vals) if vals else 0.0
            print(f"{avg:>{col_w}.4f}", end="")
        print()

    print(f"{'='*65}\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_users", type=int, default=200)
    args = parser.parse_args()

    print(f"=== Offline Evaluation (n_users={args.n_users}) ===\n")

    ratings_df = load_ratings(sample_frac=0.1)
    pop_scores = load_popularity_scores(ratings_df)
    all_movie_ids = list(ratings_df["movieId"].unique())

    print("Loading recommender ...")
    from recommender import HybridRecommender
    rec = HybridRecommender(ratings_df=ratings_df, verbose=False)

    print(f"\nEvaluating {args.n_users} users ...")
    metrics, n_eval = evaluate_system(
        rec, ratings_df, pop_scores, all_movie_ids,
        n_users=args.n_users, k_list=EVAL_K_LIST
    )

    print_results(metrics, EVAL_K_LIST, n_eval)


if __name__ == "__main__":
    main()
