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
        counts = count_train_occurrences(sessions)
        counts = counts.rename_axis("track_id").reset_index(name="n").sort_values(["n", "track_id"], ascending=[False, True])
        self.ids = counts["track_id"].to_numpy()
        self.scores = counts["n"].to_numpy(dtype=float)
        return self

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        return top_k_excluding(self.ids, self.scores, session_track_ids, k)
