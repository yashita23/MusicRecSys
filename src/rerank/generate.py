"""Sinh chuỗi tham lam từ seed (ticket #18, spec §7 Độ mượt).

Mỗi bước: mô hình gu chọn top-n ứng viên cho phiên hiện tại → chấm điểm cuối (#17) với bài đang phát
là bài vừa thêm → lấy top-1, nối vào phiên. Lặp `length` lần. Mô hình gu đã loại bài có trong phiên nên chuỗi không lặp bài.
"""
from __future__ import annotations

from collections.abc import Sequence

from src.data.schema import N_CANDIDATES
from src.models.base import TasteModel, candidates_for_sessions
from src.rerank.combine import MixScorer, score_candidates

CHAIN_LENGTH = 10


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
        for session, (ids, taste) in zip(sessions, candidates_for_sessions(model, sessions, n_candidates)):
            session.append(score_candidates(ids, taste, session[-1], mix, alpha).best())
    return [[session[len(seed) - 1], *session[len(seed):]] for session, seed in zip(sessions, seeds)]
