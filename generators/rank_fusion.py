def weighted_rrf(ranked_lists, weights, k=60, top_n=200):
    """
    Weighted Reciprocal Rank Fusion.

    ranked_lists: list of lists, each list is [(movieId, score), ...]
    weights: list of floats, one per ranked list (same order)
    k: RRF constant (default 60)
    top_n: number of candidates to return

    Returns: [(movieId, fused_score), ...] sorted desc, top_n items.
    """
    scores = {}

    for ranked, weight in zip(ranked_lists, weights):
        for rank, (movie_id, _) in enumerate(ranked, start=1):
            rrf_score = weight / (k + rank)
            scores[movie_id] = scores.get(movie_id, 0.0) + rrf_score

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_n]
