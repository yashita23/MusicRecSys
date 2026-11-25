"""Xếp lại ứng viên theo điểm cuối = α · gu_norm + (1 − α) · điểm mix (ticket #17, spec §6).

Điểm mix là một hàm bất kỳ cùng giao diện với RuleMix (#12): mix(bài đang phát, ứng viên) → mảng điểm theo đúng
thứ tự ứng viên. Nhờ vậy điểm mix học (#23) cắm vào được mà không phải sửa module này.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

MixScorer = Callable[[str, Sequence[str]], np.ndarray]
TOP_K = 10


def minmax(taste_scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """gu_norm: chuẩn hóa min-max về 0–1 trong tập ứng viên; mọi điểm bằng nhau thì gu_norm = 1."""
    x = np.asarray(taste_scores, dtype=float)
    span = x.max() - x.min()
    return np.ones_like(x) if span == 0 else (x - x.min()) / span


def rerank(
    candidate_ids: Sequence[str],
    taste_scores: Sequence[float] | np.ndarray,
    current_track_id: str,
    mix: MixScorer,
    alpha: float,
    k: int | None = TOP_K,
) -> pd.DataFrame:
    """Xếp lại ứng viên của một phiên; trả về top-k (k=None: cả danh sách).

    Cột: rank (1…), track_id, taste_score, gu_norm, mix, final. Hòa điểm cuối thì giữ thứ tự ứng viên đầu vào.
    """
    gu_norm = minmax(taste_scores)
    mix_scores = np.asarray(mix(current_track_id, list(candidate_ids)), dtype=float)
    final = alpha * gu_norm + (1 - alpha) * mix_scores
    out = pd.DataFrame(
        {"track_id": list(candidate_ids), "taste_score": np.asarray(taste_scores, dtype=float), "gu_norm": gu_norm, "mix": mix_scores, "final": final}
    )
    out = out.sort_values("final", ascending=False, kind="stable").head(k if k is not None else len(out))
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out.reset_index(drop=True)


def rerank_candidates(
    candidates: pd.DataFrame,
    eval_targets: pd.DataFrame,
    mix: MixScorer,
    alpha: float,
    k: int | None = TOP_K,
) -> pd.DataFrame:
    """Xếp lại cả một file candidates; bài đang phát của mỗi phiên là bài cuối của seed.

    Kết quả có thêm cột session_id, chấm thẳng được bằng src.eval.run.evaluate. Với α = 1 và k = None,
    thứ tự giữ nguyên như file gốc.
    """
    last_seed = eval_targets.set_index("session_id")["seed_track_ids"].map(lambda s: s[-1])
    parts = []
    for session_id, group in candidates.sort_values(["session_id", "rank"]).groupby("session_id", sort=True):
        out = rerank(group["track_id"].tolist(), group["taste_score"].to_numpy(), last_seed[session_id], mix, alpha, k)
        out.insert(0, "session_id", session_id)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)
