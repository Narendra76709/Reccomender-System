import numpy as np
import scipy.sparse as sp


class NMFANNGenerator:
    """
    Generates recommendations using NMF item factors + Annoy ANN index.

    NMF was trained on item-user matrix:  X_items (n_items x n_users) = W @ H
    - W (item factors): shape (n_items, n_components)  -- stored in Annoy
    - H (user/col factors): shape (n_components, n_users) = model.components_

    For a new user: project ratings into latent space via the pseudo-inverse:
        user_latent = user_ratings_vec @ W @ pinv(W.T @ W)
    Then do ANN lookup in item factor space.
    """

    def __init__(self, nmf_model, ann_index, ann_dim, encodings, ratings_df, item_factors):
        self.nmf = nmf_model
        self.ann = ann_index
        self.ann_dim = ann_dim
        self.user_enc, self.movie_enc, self.user_ids, self.movie_ids = encodings
        self.ratings_df = ratings_df
        self.movie_id_arr = np.array(self.movie_ids)
        self.n_items = len(self.movie_ids)
        # item_factors: (n_items, n_components)
        self.item_factors = item_factors
        # Precompute (W.T @ W)^-1 @ W.T for projecting user ratings
        WtW = item_factors.T @ item_factors  # (n_comp, n_comp)
        try:
            self._WtW_inv_Wt = np.linalg.solve(WtW, item_factors.T)  # (n_comp, n_items)
        except np.linalg.LinAlgError:
            self._WtW_inv_Wt = np.linalg.pinv(item_factors)  # fallback

    def is_available(self):
        return self.nmf is not None and self.ann is not None

    def _build_user_vector(self, user_id):
        """Project user ratings into NMF latent space."""
        user_rows = self.ratings_df[self.ratings_df["userId"] == user_id]
        if user_rows.empty:
            return None

        # Build dense rating vector over all items
        rating_vec = np.zeros(self.n_items)
        for _, r in user_rows.iterrows():
            mid = r["movieId"]
            if mid in self.movie_enc:
                rating_vec[self.movie_enc[mid]] = r["rating"]

        if rating_vec.sum() == 0:
            return None

        # user_latent = (W.T W)^{-1} W.T r  => shape (n_comp,)
        vec = self._WtW_inv_Wt @ rating_vec  # (n_comp,)
        vec = np.maximum(vec, 0)  # NMF non-negativity

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
