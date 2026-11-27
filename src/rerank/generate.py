"""Sinh chuỗi tham lam từ seed (ticket #18, spec §7 Độ mượt).

Mỗi bước: mô hình gu chọn top-n ứng viên cho phiên hiện tại → chấm điểm cuối (#17) với bài đang phát
là bài vừa thêm → lấy top-1, nối vào phiên. Lặp `length` lần. Mô hình gu đã loại bài có trong phiên nên chuỗi không lặp bài.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from src.models.base import BATCH_SESSIONS, TasteModel
from src.rerank.combine import MixScorer, final_scores

CHAIN_LENGTH = 10
N_CANDIDATES = 100


def _candidates_for(model: TasteModel, sessions: list[list[str]], k: int) -> list[tuple[list[str], np.ndarray]]:
    """Ứng viên cho mọi phiên; mô hình có candidates_batch (như SASRec) thì chấm theo lô, giống export_candidates."""
    batch = getattr(model, "candidates_batch", None)
    if batch is None:
        return [model.candidates(s, k=k) for s in sessions]
    return [r for start in range(0, len(sessions), BATCH_SESSIONS) for r in batch(sessions[start : start + BATCH_SESSIONS], k)]


def generate_chains(
    model: TasteModel,
    seeds: Sequence[Sequence[str]],
    mix: MixScorer,
    alpha: float,
    length: int = CHAIN_LENGTH,
    n_candidates: int = N_CANDIDATES,
) -> list[list[str]]:
    """Mỗi seed → [bài cuối seed, bài sinh 1, …, bài sinh `length`]; đúng đầu vào của src.eval.smoothness."""
    sessions = [list(seed) for seed in seeds]
    for _ in range(length):
        for session, (ids, taste) in zip(sessions, _candidates_for(model, sessions, n_candidates)):
            _, _, final = final_scores(ids, taste, session[-1], mix, alpha)
            session.append(ids[int(np.argmax(final))])  # argmax lấy ứng viên đầu tiên khi hòa, như rerank
    return [[session[len(seed) - 1], *session[len(seed):]] for session, seed in zip(sessions, seeds)]
