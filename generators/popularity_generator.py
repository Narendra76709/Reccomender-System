import pandas as pd


class PopularityGenerator:
    """
    Recommends movies based on Bayesian average popularity score.
    Optionally boosts movies matching the user's preferred genres.
    """

    def __init__(self, popularity_scores, movies_df, ratings_df):
        self.pop_scores = popularity_scores      # dict: movieId -> float
        self.movies_df = movies_df
        self.ratings_df = ratings_df
        # Build genre lookup: movieId -> set of genres
        self.movie_genres = {}
        for _, row in movies_df.iterrows():
            genres = set(row["genres"].split("|")) if pd.notna(row.get("genres")) else set()
            self.movie_genres[row["movieId"]] = genres

    def is_available(self):
        return bool(self.pop_scores)

    def _get_user_genres(self, user_id):
        """Find genres the user has rated highly."""
        user_rows = self.ratings_df[
            (self.ratings_df["userId"] == user_id) &
            (self.ratings_df["rating"] >= 4.0)
        ]
        liked_movies = set(user_rows["movieId"].tolist())
        genres = set()
        for mid in liked_movies:
            genres.update(self.movie_genres.get(mid, set()))
        return genres

    def recommend(self, user_id, exclude_ids, top_k=60):
        """Returns list of (movieId, score) sorted by score desc."""
        if not self.is_available():
            return []

        # Get user's preferred genres for boosting
        user_genres = self._get_user_genres(user_id)

        scored = []
        for mid, score in self.pop_scores.items():
            if mid in exclude_ids:
                continue
            # Genre boost: +5% per matching genre, up to +25%
            if user_genres:
                movie_g = self.movie_genres.get(mid, set())
                overlap = len(user_genres & movie_g)
                boost = 1.0 + min(overlap * 0.05, 0.25)
                score = score * boost
            scored.append((mid, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
