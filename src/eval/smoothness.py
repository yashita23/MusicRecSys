"""Chỉ số độ mượt của chuỗi bài (ticket #13, spec §7).

Dùng chung phép đo ΔBPM và định nghĩa tông tương thích với điểm mix theo luật (#12), nhưng ngưỡng
"cặp mix được" là của riêng phần đánh giá (GLOSSARY), nên chỉnh luật không làm chỉ số âm thầm đổi theo.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from src.features.track_features import TrackFeatures

MIXABLE_MAX_BPM_DELTA = 0.06   # cặp mix được: ΔBPM ≤ 6% (GLOSSARY)
BPM_TOLERANCE = 1e-9          # để lệch đúng 6% (sai số dấu phẩy động) vẫn tính là ≤ 6%


@dataclass(frozen=True)
class Smoothness:
    """Gộp trên mọi cặp liền kề của mọi chuỗi. Tỉ lệ tính bằng %; energy giữ thang 0–1."""

    n_pairs: int
    bpm_delta_pct: float       # ΔBPM trung bình (%), xét ×2/×0.5
    key_compatible_pct: float  # % cặp chuyển tông hợp
    energy_gap: float          # |Δenergy| trung bình, thang 0–1
    mixable_pct: float         # % cặp mix được: ΔBPM ≤ 6% và tông tương thích


def smoothness(chains: Sequence[Sequence[str]], features: TrackFeatures) -> Smoothness:
    """chains: mỗi chuỗi là [bài cuối seed, bài sinh 1, …]; đo từng cặp liền kề (chuỗi n bài có n − 1 cặp)."""
    pairs = [(c[i], c[i + 1]) for c in chains for i in range(len(c) - 1)]
    if not pairs:
        nan = float("nan")
        return Smoothness(0, nan, nan, nan, nan)
    rows_a = features.rows([a for a, _ in pairs])
    rows_b = features.rows([b for _, b in pairs])
    delta = features.bpm_delta(rows_a, rows_b)
    key_ok = features.compatible_keys(rows_a, rows_b)
    mixable = (delta <= MIXABLE_MAX_BPM_DELTA + BPM_TOLERANCE) & key_ok
    return Smoothness(
        n_pairs=len(pairs),
        bpm_delta_pct=float(delta.mean() * 100),
        key_compatible_pct=float(key_ok.mean() * 100),
        energy_gap=float(features.energy_gap(rows_a, rows_b).mean()),
        mixable_pct=float(mixable.mean() * 100),
    )
