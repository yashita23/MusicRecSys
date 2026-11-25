"""Điểm mix theo luật DJ cho cặp (bài đang phát a, ứng viên b), dạng vector (ticket #12, spec §6)."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.features.camelot import compatible_keys, parse_camelot, same_key
from src.features.track_features import TrackFeatures, bpm_delta

BPM_FULL = 0.06   # lệch ≤ 6% → s_bpm = 1
BPM_ZERO = 0.12   # lệch ≥ 12% → s_bpm = 0; ở giữa giảm tuyến tính
KEY_SAME = 1.0
KEY_NEIGHBOUR = 0.8  # tông tương thích nhưng khác mã


def _bpm_score(delta: np.ndarray) -> np.ndarray:
    return np.clip((BPM_ZERO - delta) / (BPM_ZERO - BPM_FULL), 0.0, 1.0)


def _key_score(same: np.ndarray, compatible: np.ndarray) -> np.ndarray:
    return np.where(same, KEY_SAME, np.where(compatible, KEY_NEIGHBOUR, 0.0))


def s_bpm(bpm_a: np.ndarray, bpm_b: np.ndarray) -> np.ndarray:
    """1 khi lệch ≤ 6%, giảm tuyến tính về 0 ở 12% (lệch tính bằng bpm_delta, xét ×2/×0.5)."""
    return _bpm_score(bpm_delta(bpm_a, bpm_b))


def s_key(camelot_a: Sequence[str] | np.ndarray, camelot_b: Sequence[str] | np.ndarray) -> np.ndarray:
    """Cùng mã = 1; ±1 số cùng chữ hoặc cùng số khác chữ = 0.8; còn lại (kể cả bước chéo) = 0."""
    a, b = parse_camelot(camelot_a), parse_camelot(camelot_b)
    return _key_score(same_key(*a, *b), compatible_keys(*a, *b))


def s_energy(energy_a: np.ndarray, energy_b: np.ndarray) -> np.ndarray:
    """1 − |Δenergy|."""
    return 1.0 - np.abs(np.asarray(energy_a, dtype=float) - np.asarray(energy_b, dtype=float))


@dataclass(frozen=True)
class MixWeights:
    """Trọng số của ba thành phần; không tự chuẩn hóa (ablation ở #27 tự chuẩn hóa lại nếu cần)."""

    bpm: float = 0.4
    key: float = 0.4
    energy: float = 0.2


@dataclass(frozen=True)
class MixComponents:
    """Ba điểm thành phần cho từng ứng viên, theo đúng thứ tự ứng viên."""

    bpm: np.ndarray
    key: np.ndarray
    energy: np.ndarray

    def weighted(self, w: MixWeights) -> np.ndarray:
        return w.bpm * self.bpm + w.key * self.key + w.energy * self.energy


class RuleMix:
    """Điểm mix theo luật: w_b·s_bpm + w_k·s_key + w_e·s_energy.

    Dựng một lần từ bảng tracks (cột track_id, bpm, camelot, energy), rồi gọi trực tiếp:
    `RuleMix(tracks)(current_track_id, candidate_track_ids)` → mảng điểm mix theo đúng thứ tự ứng viên.
    """

    def __init__(self, tracks: pd.DataFrame, weights: MixWeights = MixWeights()):
        self.features = TrackFeatures.from_tracks(tracks)
        self.weights = weights

    def with_weights(self, weights: MixWeights) -> RuleMix:
        """Cùng danh mục, trọng số khác; không đọc lại bảng tracks (cho grid search #25)."""
        clone = copy.copy(self)  # bản sao nông: dùng chung mảng đặc trưng, chỉ đổi trọng số
        clone.weights = weights
        return clone

    def components(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> MixComponents:
        f = self.features
        current, candidates = f.rows([current_track_id]), f.rows(list(candidate_track_ids))
        return MixComponents(
            bpm=_bpm_score(f.bpm_delta(current, candidates)),
            key=_key_score(f.same_key(current, candidates), f.compatible_keys(current, candidates)),
            energy=1.0 - f.energy_gap(current, candidates),
        )

    def __call__(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> np.ndarray:
        return self.components(current_track_id, candidate_track_ids).weighted(self.weights)
