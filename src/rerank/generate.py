"""Sinh chuỗi tham lam từ seed (ticket #18, spec §7 Độ mượt).

Mỗi bước: mô hình gu chọn top-n ứng viên cho phiên hiện tại → chấm điểm cuối (#17) với bài đang phát
là bài vừa thêm → lấy top-1, nối vào phiên. Lặp `length` lần. Mô hình gu đã loại bài có trong phiên nên chuỗi không lặp bài.

generate_chains_many làm việc đó cho nhiều cấu hình mix cùng α một lúc (grid search #25): mỗi bước, mỗi phiên khác nhau chỉ
hỏi mô hình gu và hàm điểm mix một lần, nên các cấu hình đi cùng một đường (thường gặp khi α gần 1) dùng chung ứng viên,
và hàm điểm mix tính được điểm của mọi cấu hình từ cùng một lần tra đặc trưng (vd. ba điểm thành phần của luật).
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np

from src.data.schema import N_CANDIDATES
from src.models.base import TasteModel, candidates_for_sessions
from src.rerank.combine import MixScorer, gu_norm, score_candidates

CHAIN_LENGTH = 10

# (bài đang phát, ứng viên) → điểm mix của từng cấu hình, mỗi cấu hình một mảng theo thứ tự ứng viên
MultiMixScorer = Callable[[str, Sequence[str]], Sequence[np.ndarray]]


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


def all_of(mixes: Sequence[MixScorer]) -> MultiMixScorer:
    """Gộp nhiều hàm điểm mix thành một MultiMixScorer (mỗi hàm một cấu hình)."""
    return lambda current, candidate_ids: [np.asarray(mix(current, candidate_ids), dtype=float) for mix in mixes]


def generate_chains_many(
    model: TasteModel,
    seeds: Sequence[Sequence[str]],
    mixes: MultiMixScorer,
    n_mixes: int,
    alpha: float,
    length: int = CHAIN_LENGTH,
    n_candidates: int = N_CANDIDATES,
) -> list[list[list[str]]]:
    """Như generate_chains cho từng cấu hình trong n_mixes cấu hình của mixes (cùng α), theo đúng thứ tự cấu hình.

    Chọn bài như score_candidates(...).best(): điểm cuối = α · gu_norm + (1 − α) · điểm mix, hòa thì ứng viên đứng trước.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"α phải trong khoảng 0–1, đang là {alpha}")
    sessions = [[list(seed) for seed in seeds] for _ in range(n_mixes)]  # sessions[m][s]: phiên của seed s, cấu hình m
    for _ in range(length):
        unique: dict[tuple[str, ...], int] = {}
        for per_mix in sessions:
            for session in per_mix:
                unique.setdefault(tuple(session), len(unique))
        keys = list(unique)
        candidates = candidates_for_sessions(model, [list(key) for key in keys], n_candidates)
        best: list[list[str] | None] = [None] * len(keys)  # best[u][m]: bài chọn ở phiên u với cấu hình m
        for u, (key, (ids, taste)) in enumerate(zip(keys, candidates)):
            scores = mixes(key[-1], ids)
            if len(scores) != n_mixes:
                raise ValueError(f"hàm điểm mix phải trả về {n_mixes} mảng điểm, đang trả về {len(scores)}")
            norm = gu_norm(taste)
            best[u] = [ids[int(np.argmax(alpha * norm + (1 - alpha) * np.asarray(s, dtype=float)))] for s in scores]
        for m, per_mix in enumerate(sessions):
            for session in per_mix:
                session.append(best[unique[tuple(session)]][m])
    return [
        [[session[len(seed) - 1], *session[len(seed):]] for session, seed in zip(per_mix, seeds)]
        for per_mix in sessions
    ]
