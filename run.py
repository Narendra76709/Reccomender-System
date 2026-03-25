"""
Unified CLI for the Hybrid Movie Recommender.
Commands: serve, train_models, train_reranker, evaluate, demo, test
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
import argparse
import subprocess


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable


def cmd_serve(args):
    print(f"Starting web server on http://localhost:{args.port} ...")
    cmd = [PYTHON, "-m", "uvicorn", "api:app",
           "--host", "0.0.0.0", "--port", str(args.port)]
    if args.reload:
        cmd.append("--reload")
    subprocess.run(cmd, cwd=BASE_DIR)


def cmd_train_models(args):
    print("Training NMF + SVD models ...")
    subprocess.run(
        [PYTHON, "train_models.py",
         "--sample_frac", str(args.sample_frac),
         "--nmf_components", str(args.nmf_components),
         "--svd_components", str(args.svd_components)],
        cwd=BASE_DIR
    )


def cmd_train_reranker(args):
    print("Training LightGBM reranker ...")
    subprocess.run(
        [PYTHON, "train_reranker.py",
         "--n_users", str(args.n_users)],
        cwd=BASE_DIR
    )


def cmd_evaluate(args):
    print("Running evaluation ...")
    subprocess.run(
        [PYTHON, "evaluation.py",
         "--n_users", str(args.n_users)],
        cwd=BASE_DIR
    )


def cmd_demo(args):
    print(f"Demo for user_id={args.user_id} ...")
    from recommender import HybridRecommender
    rec = HybridRecommender(verbose=True)
    results = rec.recommend(args.user_id, top_k=10, return_details=True)
    print(f"\n--- Top-10 Recommendations for User {args.user_id} ---")
    for i, r in enumerate(results, 1):
        print(f"  {i:2}. [{r['score']:.4f}] {r['title']}  ({r['genres']})")
    if not results:
        print("  No recommendations (user may not exist in training data)")


def cmd_test(args):
    """Smoke test: verify imports and basic pipeline."""
    print("Running smoke tests ...")
    errors = []

    # Test imports
    try:
        import config
        print("  [OK] config")
    except Exception as e:
        errors.append(f"config: {e}")
        print(f"  [FAIL] config: {e}")

    try:
        import data_loader
        print("  [OK] data_loader")
    except Exception as e:
        errors.append(f"data_loader: {e}")
        print(f"  [FAIL] data_loader: {e}")

    try:
        from generators.nmf_ann_generator import NMFANNGenerator
        from generators.svd_ann_generator import SVDANNGenerator
        from generators.popularity_generator import PopularityGenerator
        from generators.rank_fusion import weighted_rrf
        print("  [OK] generators")
    except Exception as e:
        errors.append(f"generators: {e}")
        print(f"  [FAIL] generators: {e}")

    try:
        from stage2_reranker import Stage2Reranker
        print("  [OK] stage2_reranker")
    except Exception as e:
        errors.append(f"stage2_reranker: {e}")
        print(f"  [FAIL] stage2_reranker: {e}")

    try:
        import recommender
        print("  [OK] recommender (import)")
    except Exception as e:
        errors.append(f"recommender: {e}")
        print(f"  [FAIL] recommender: {e}")

    try:
        import api
        print("  [OK] api (import)")
    except Exception as e:
        errors.append(f"api: {e}")
        print(f"  [FAIL] api: {e}")

    # Test rank fusion
    try:
        from generators.rank_fusion import weighted_rrf
        lists = [[(1, 0.9), (2, 0.8), (3, 0.7)],
                 [(2, 0.9), (3, 0.8), (4, 0.7)]]
        result = weighted_rrf(lists, [0.6, 0.4], top_n=4)
        assert len(result) > 0
        print("  [OK] rank_fusion logic")
    except Exception as e:
        errors.append(f"rank_fusion: {e}")
        print(f"  [FAIL] rank_fusion: {e}")

    # Test data loader (just movies)
    try:
        from data_loader import load_movies
        movies = load_movies()
        assert len(movies) > 0
        print(f"  [OK] data_loader.load_movies ({len(movies):,} movies)")
    except Exception as e:
        errors.append(f"load_movies: {e}")
        print(f"  [FAIL] load_movies: {e}")

    # Test model files exist
    from config import NMF_MODEL_PATH, SVD_MODEL_PATH, NMF_ANN_PATH, SVD_ANN_PATH, LGBM_MODEL_PATH
    for name, path in [
        ("NMF model", NMF_MODEL_PATH),
        ("NMF Annoy", NMF_ANN_PATH),
        ("SVD model", SVD_MODEL_PATH),
        ("SVD Annoy", SVD_ANN_PATH),
        ("LGBM model", LGBM_MODEL_PATH),
    ]:
        if os.path.exists(path):
            size_mb = os.path.getsize(path) / 1024 / 1024
            print(f"  [OK] {name} exists ({size_mb:.1f} MB)")
        else:
            print(f"  [WARN] {name} not found (run train_models.py first)")

    # Summary
    print()
    if errors:
        print(f"RESULT: {len(errors)} error(s) found:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("RESULT: All smoke tests passed!")


def main():
    parser = argparse.ArgumentParser(description="Hybrid Movie Recommender CLI")
    sub = parser.add_subparsers(dest="command")

    # serve
    p_serve = sub.add_parser("serve", help="Start FastAPI web server")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--reload", action="store_true")

    # train_models
    p_tm = sub.add_parser("train_models", help="Train NMF + SVD models")
    p_tm.add_argument("--sample_frac", type=float, default=0.1)
    p_tm.add_argument("--nmf_components", type=int, default=20)
    p_tm.add_argument("--svd_components", type=int, default=20)

    # train_reranker
    p_tr = sub.add_parser("train_reranker", help="Train LightGBM reranker")
    p_tr.add_argument("--n_users", type=int, default=2000)

    # evaluate
    p_ev = sub.add_parser("evaluate", help="Run offline evaluation")
    p_ev.add_argument("--n_users", type=int, default=200)

    # demo
    p_demo = sub.add_parser("demo", help="CLI demo for a user")
    p_demo.add_argument("--user_id", type=int, default=1)

    # test
    p_test = sub.add_parser("test", help="Smoke test all components")

    args = parser.parse_args()

    if args.command == "serve":
        cmd_serve(args)
    elif args.command == "train_models":
        cmd_train_models(args)
    elif args.command == "train_reranker":
        cmd_train_reranker(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "demo":
        cmd_demo(args)
    elif args.command == "test":
        cmd_test(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
