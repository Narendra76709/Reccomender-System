import numpy as np
import scipy.sparse as sp


class SVDANNGenerator:
    """
    Generates recommendations using TruncatedSVD item factors + Annoy ANN index.
    User vector = svd.transform(user_rating_row)[0].
    """

    def __init__(self, svd_model, ann_index, ann_dim, encodings, ratings_df):
        self.svd = svd_model
        self.ann = ann_index
        self.ann_dim = ann_dim
        self.user_enc, self.movie_enc, self.user_ids, self.movie_ids = encodings
        self.ratings_df = ratings_df
        self.movie_id_arr = np.array(self.movie_ids)
        self.n_items = len(self.movie_ids)

    def is_available(self):
        return self.svd is not None and self.ann is not None

    def _build_user_vector(self, user_id):
        """Build a sparse user-rating row and project via SVD."""
        user_rows = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user_rows.empty:
            return None

        row_indices = []
        col_indices = []
        data = []
        for _, r in user_rows.iterrows():
            mid = r["movieId"]
            if mid in self.movie_enc:
                col_indices.append(self.movie_enc[mid])
                row_indices.append(0)
                data.append(r["rating"])

        if not data:
            return None

        user_mat = sp.csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(1, self.n_items)
        )
        vec = self.svd.transform(user_mat)[0]  # shape: (n_components,)

        # Handle dimension mismatch
        if len(vec) < self.ann_dim:
            vec = np.pad(vec, (0, self.ann_dim - len(vec)))
        elif len(vec) > self.ann_dim:
            vec = vec[:self.ann_dim]

        return vec

    def recommend(self, user_id, exclude_ids, top_k=100):
        """Returns list of (movieId, score) sorted by score desc."""
        if not self.is_available():
            return []

        vec = self._build_user_vector(user_id)
        if vec is None:
            return []

        n_fetch = min(top_k + len(exclude_ids) + 50, self.n_items)
        indices, distances = self.ann.get_nns_by_vector(
            vec.tolist(), n_fetch, include_distances=True
        )

        results = []
        for idx, dist in zip(indices, distances):
            if idx >= len(self.movie_id_arr):
                continue
            mid = int(self.movie_id_arr[idx])
            if mid in exclude_ids:
                continue
            score = 1.0 / (1.0 + dist)
            results.append((mid, score))
            if len(results) >= top_k:
                break

        return results
