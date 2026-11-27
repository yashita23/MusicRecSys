"""Baseline Popularity: bài có nhiều lượt nhất trong các phiên train (ticket #9, spec §6)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES, count_train_occurrences
from src.models.base import top_k_excluding


class Popularity:
    """Điểm gu = số lượt xuất hiện trong train; hòa thì xếp theo track_id cho ổn định."""

    name = "popularity"

    def fit(self, sessions: pd.DataFrame) -> Popularity:
        return self._from_counts(count_train_occurrences(sessions))

    @classmethod
    def from_tracks(cls, tracks: pd.DataFrame) -> Popularity:
        """Từ cột n_occurrences (số lượt trong train) của tracks: giống fit(sessions) mà không cần sessions (app, #20)."""
        counts = tracks.set_index("track_id")["n_occurrences"]
        return cls()._from_counts(counts[counts > 0])

    def _from_counts(self, counts: pd.Series) -> Popularity:
        counts = counts.rename_axis("track_id").reset_index(name="n").sort_values(["n", "track_id"], ascending=[False, True])
        self.ids = counts["track_id"].to_numpy()
        self.scores = counts["n"].to_numpy(dtype=float)
        return self

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        return top_k_excluding(self.ids, self.scores, session_track_ids, k)
