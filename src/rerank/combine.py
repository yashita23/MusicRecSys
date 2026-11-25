"""Xếp lại ứng viên theo điểm cuối = α · gu_norm + (1 − α) · điểm mix (ticket #17, spec §6).

Điểm mix là một hàm bất kỳ cùng giao diện với RuleMix (#12): mix(bài đang phát, ứng viên) → mảng điểm theo đúng
thứ tự ứng viên. Nhờ vậy điểm mix học (#23) cắm vào được mà không phải sửa module này.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from src.data.schema import SchemaError
from src.eval.run import K as TOP_K  # top-10 của hàng đợi trùng với k của HitRate@10, NDCG@10

MixScorer = Callable[[str, Sequence[str]], np.ndarray]


def gu_norm(taste_scores: Sequence[float] | np.ndarray) -> np.ndarray:
    """Điểm gu chuẩn hóa min-max về 0–1 trong tập ứng viên; mọi điểm bằng nhau thì gu_norm = 1."""
    x = np.asarray(taste_scores, dtype=float)
    span = x.max() - x.min()
    return np.ones_like(x) if span == 0 else (x - x.min()) / span


def _check_inputs(ids: list[str], taste: np.ndarray, alpha: float, k: int | None) -> None:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"α phải trong khoảng 0–1, đang là {alpha}")
    if k is not None and k < 1:
        raise ValueError(f"k phải ≥ 1 (hoặc None để lấy cả danh sách), đang là {k}")
    if taste.shape != (len(ids),):
        raise ValueError(f"điểm gu phải có đúng {len(ids)} giá trị, đang có shape {taste.shape}")
    if not np.isfinite(taste).all():
        raise ValueError(f"điểm gu phải hữu hạn, có {int((~np.isfinite(taste)).sum())} giá trị NaN/inf")
    if len(set(ids)) != len(ids):
        raise ValueError(f"ứng viên bị trùng: {len(ids) - len(set(ids))} bài xuất hiện nhiều lần")


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
    ids = list(candidate_ids)
    taste = np.asarray(taste_scores, dtype=float)
    _check_inputs(ids, taste, alpha, k)
    mix_scores = np.asarray(mix(current_track_id, ids), dtype=float)
    if mix_scores.shape != (len(ids),):
        raise ValueError(f"hàm điểm mix phải trả về mảng đúng độ dài {len(ids)} ứng viên, đang có shape {mix_scores.shape}")
    if not np.isfinite(mix_scores).all():
        raise ValueError(f"điểm mix phải hữu hạn, có {int((~np.isfinite(mix_scores)).sum())} giá trị NaN/inf")
    norm = gu_norm(taste)
    out = pd.DataFrame(
        {
            "track_id": ids,
            "taste_score": taste,
            "gu_norm": norm,
            "mix": mix_scores,
            "final": alpha * norm + (1 - alpha) * mix_scores,
        }
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
    if candidates.empty:
        raise SchemaError("candidates", ["danh sách rỗng, không có phiên nào để xếp lại"])
    last_seed = eval_targets.set_index("session_id")["seed_track_ids"].map(lambda s: s[-1])
    unknown = pd.Index(candidates["session_id"].unique()).difference(last_seed.index)
    if len(unknown):
        raise SchemaError("candidates", [f"{len(unknown)} phiên không có trong eval_targets (ví dụ {list(unknown[:3])})"])
    parts = []
    for session_id, group in candidates.sort_values(["session_id", "rank"]).groupby("session_id", sort=True):
        out = rerank(group["track_id"].tolist(), group["taste_score"].to_numpy(), last_seed[session_id], mix, alpha, k)
        out.insert(0, "session_id", session_id)
        parts.append(out)
    return pd.concat(parts, ignore_index=True)
