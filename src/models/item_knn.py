"""Baseline item-kNN liền kề (ticket #9, spec §6, GLOSSARY).

Ứng viên lấy theo ba tầng, tầng trước hết chỗ mới sang tầng sau:
1. Liền kề: những bài hay đứng ngay sau bài cuối của phiên trong các playlist train (đếm cặp có hướng a→b).
2. Đồng xuất hiện: cộng, qua mọi bài của phiên, số playlist train có cả bài đó lẫn ứng viên.
3. Popularity.

Điểm gu theo tầng để không tăng theo hạng mà vẫn giữ chênh lệch trong từng tầng:
tầng 1 trong (2, 3], tầng 2 trong (1, 2], tầng 3 trong (0, 1]; trong mỗi tầng = nền + đếm / đếm lớn nhất của tầng.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.schema import N_CANDIDATES
from src.models.popularity import Popularity

ADJACENT_BASE, COOCCURRENCE_BASE, POPULARITY_BASE = 2.0, 1.0, 0.0


def _nonzero_by_count(indices: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Chỉ giữ đếm > 0, xếp theo đếm giảm dần; hòa thì theo chỉ số bài (tức theo track_id)."""
    keep = counts > 0
    indices, counts = indices[keep], counts[keep]
    order = np.lexsort((indices, -counts))
    return indices[order], counts[order]


def _tier_scores(counts: np.ndarray, base: float) -> np.ndarray:
    return base + counts / counts.max() if len(counts) else counts.astype(float)


class ItemKNN:
    name = "itemknn"

    def fit(self, sessions: pd.DataFrame) -> ItemKNN:
        train = sessions[sessions["split"] == "train"].sort_values(["session_id", "pos"])
        self.vocab = pd.Index(np.sort(train["track_id"].unique()))
        n = len(self.vocab)
        track = self.vocab.get_indexer(train["track_id"])
        session = pd.factorize(train["session_id"])[0]

        same = session[1:] == session[:-1]
        a, b = track[:-1][same], track[1:][same]
        self.adjacent = sp.csr_matrix((np.ones(len(a)), (a, b)), shape=(n, n))  # trùng (a, b) được cộng dồn

        incidence = sp.csr_matrix((np.ones(len(track)), (session, track)), shape=(session.max() + 1, n))
        incidence.data[:] = 1.0  # có mặt hay không, không đếm số lần lặp trong một playlist
        self.playlists_of_track = incidence.T.tocsr()  # bài → playlist chứa nó
        self.tracks_of_playlist = incidence

        self.popularity = Popularity().fit(sessions)
        return self

    def _followers(self, last: int) -> tuple[np.ndarray, np.ndarray]:
        """Tầng 1: những bài đứng ngay sau `last` trong train, kèm số lần."""
        row = self.adjacent.getrow(last)
        return _nonzero_by_count(row.indices, row.data)

    def _cooccurring(self, session: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Tầng 2: cộng, qua từng bài (không lặp) của phiên, số playlist train có cả bài đó lẫn ứng viên."""
        weight = np.asarray(self.playlists_of_track[session].sum(axis=0)).ravel()  # playlist → số bài của phiên nó chứa
        playlists = np.flatnonzero(weight)
        counts = self.tracks_of_playlist[playlists].T @ weight[playlists]
        return _nonzero_by_count(np.arange(len(counts)), counts)

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        session_ids = list(session_track_ids)
        codes = self.vocab.get_indexer(session_ids)
        known = np.unique(codes[codes >= 0])  # bài lặp trong phiên chỉ tính một lần
        taken = np.zeros(len(self.vocab), dtype=bool)
        taken[known] = True  # bài đã có trong phiên không bao giờ là ứng viên

        tiers: list[tuple[Callable[[], tuple[np.ndarray, np.ndarray]], float]] = []
        if len(codes) and codes[-1] >= 0:
            tiers.append((lambda: self._followers(codes[-1]), ADJACENT_BASE))
        if len(known):
            tiers.append((lambda: self._cooccurring(known), COOCCURRENCE_BASE))

        picked: list[np.ndarray] = []
        scores: list[np.ndarray] = []
        for tier, base in tiers:
            room = k - sum(len(p) for p in picked)
            if room <= 0:
                break  # tầng trước đã đủ k: không tính tầng sau (tầng đồng xuất hiện tốn nhất)
            indices, counts = tier()
            fresh = ~taken[indices]
            indices, counts = indices[fresh][:room], counts[fresh][:room]
            taken[indices] = True
            picked.append(indices)
            scores.append(_tier_scores(counts, base))

        ids = [t for p in picked for t in self.vocab[p]]
        room = k - len(ids)
        if room > 0:
            pop_ids, pop_counts = self.popularity.candidates(session_ids + ids, k=room)
            ids += pop_ids
            scores.append(_tier_scores(pop_counts, POPULARITY_BASE))
        return ids, np.concatenate(scores)
