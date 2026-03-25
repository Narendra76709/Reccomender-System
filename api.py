"""
FastAPI backend for the Hybrid Movie Recommender.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import os
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

app = FastAPI(title="Hybrid Movie Recommender")

# Global recommender instance
recommender = None


def get_recommender():
    global recommender
    if recommender is None:
        raise HTTPException(status_code=503, detail="Recommender not loaded yet")
    return recommender


@app.on_event("startup")
def startup_event():
    global recommender
    print("Starting up recommender system ...")
    try:
        from recommender import HybridRecommender
        recommender = HybridRecommender(verbose=True)
        print("Recommender loaded successfully.")
    except Exception as e:
        print(f"WARNING: Could not load recommender: {e}")
        recommender = None


@app.get("/", response_class=HTMLResponse)
def serve_frontend():
    html_path = os.path.join(os.path.dirname(__file__), "frontend", "index.html")
    if os.path.exists(html_path):
        with open(html_path, "r", encoding="utf-8") as f:
            return f.read()
    return HTMLResponse("<h1>Movie Recommender API</h1><p>Frontend not found.</p>")


@app.get("/api/status")
def status():
    rec = recommender
    if rec is None:
        return {"status": "not_loaded", "ready": False}
    return {
        "status": "ok",
        "ready": True,
        "n_movies": len(rec.movies_df),
        "n_ratings": len(rec.ratings_df),
        "reranker_loaded": rec.reranker is not None,
        "nmf_loaded": rec.nmf_gen is not None,
        "svd_loaded": rec.svd_gen is not None,
    }


@app.get("/api/search")
def search(q: str = "", limit: int = 10):
    rec = get_recommender()
    if not q.strip():
        return []
    mask = rec.movies_df["title"].str.contains(q, case=False, na=False, regex=False)
    results = rec.movies_df[mask].head(limit)
    out = []
    for _, row in results.iterrows():
        mid = int(row["movieId"])
        stats = rec.movie_stats_df.loc[mid] if mid in rec.movie_stats_df.index else None
        out.append({
            "movieId": mid,
            "title": row["title"],
            "genres": row["genres"],
            "n_ratings": int(stats["n_ratings"]) if stats is not None else 0,
            "mean_rating": round(float(stats["mean_rating"]), 2) if stats is not None else 0.0,
        })
    return out


@app.get("/api/recommend")
def recommend(user_id: int = 1, top_k: int = 10):
    rec = get_recommender()
    try:
        results = rec.recommend(user_id, top_k=top_k, return_details=True)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/movie/{movie_id}")
def movie_detail(movie_id: int):
    rec = get_recommender()
    info = rec.get_movie_info(movie_id)
    if info is None:
        raise HTTPException(status_code=404, detail="Movie not found")
    # Add similar movies
    title = info.get("title", "")
    similar = rec.recommend_by_movie(title, top_k=6)
    info["similar"] = similar
    return info


@app.get("/api/popular")
def popular(limit: int = 20, genre: str = ""):
    rec = get_recommender()
    genre_filter = genre.strip() if genre else None
    return rec.get_popular_movies(limit=limit, genre=genre_filter)


@app.get("/api/genres")
def genres():
    rec = get_recommender()
    return rec.get_all_genres()


@app.post("/api/train")
def train(n_users: int = 500, background_tasks: BackgroundTasks = None):
    """Trigger reranker training in background."""
    def run_training():
        import subprocess
        subprocess.run(
            [sys.executable, "train_reranker.py", "--n_users", str(n_users)],
            cwd=os.path.dirname(__file__)
        )
    if background_tasks:
        background_tasks.add_task(run_training)
        return {"status": "Training started in background"}
    return {"status": "Use background tasks"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=False)
