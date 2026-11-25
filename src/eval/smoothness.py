"""Chỉ số độ mượt của chuỗi bài (ticket #13, spec §7). Dùng chung hàm BPM/Camelot với điểm mix theo luật (#12)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from src.rerank.mix_rule import BPM_FULL, TrackFeatures, bpm_delta

_EPS = 1e-9  # để lệch đúng 6% (sai số dấu phẩy động) vẫn tính là ≤ 6%


@dataclass(frozen=True)
class Smoothness:
    """Gộp trên mọi cặp liền kề của mọi chuỗi."""

    n_pairs: int
    mean_bpm_delta_pct: float    # ΔBPM trung bình (%), xét ×2/×0.5
    key_compatible_share: float  # tỉ lệ chuyển tông hợp (cùng mã, ±1 số cùng chữ, cùng số khác chữ)
    mean_energy_gap: float       # |Δenergy| trung bình
    mixable_share: float         # tỉ lệ cặp mix được: ΔBPM ≤ 6% và tông tương thích


def smoothness(chains: Sequence[Sequence[str]], features: TrackFeatures) -> Smoothness:
    """chains: mỗi chuỗi là [bài cuối seed, bài sinh 1, …]; đo từng cặp liền kề (chuỗi n bài có n − 1 cặp)."""
    pairs = [(c[i], c[i + 1]) for c in chains for i in range(len(c) - 1)]
    if not pairs:
        return Smoothness(0, float("nan"), float("nan"), float("nan"), float("nan"))
    rows_a = features.rows([a for a, _ in pairs])
    rows_b = features.rows([b for _, b in pairs])
    delta = bpm_delta(features.bpm[rows_a], features.bpm[rows_b])
    key_ok = features.components(rows_a, rows_b).key > 0
    energy_gap = np.abs(features.energy[rows_a] - features.energy[rows_b])
    mixable = (delta <= BPM_FULL + _EPS) & key_ok
    return Smoothness(
        n_pairs=len(pairs),
        mean_bpm_delta_pct=float(delta.mean() * 100),
        key_compatible_share=float(key_ok.mean()),
        mean_energy_gap=float(energy_gap.mean()),
        mixable_share=float(mixable.mean()),
    )
