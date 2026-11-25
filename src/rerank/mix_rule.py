"""Điểm mix theo luật DJ cho cặp (bài đang phát a, ứng viên b), dạng vector (ticket #12, spec §6)."""
from __future__ import annotations

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preview import preview_ids
from src.features.camelot import parse_camelot

BPM_FULL = 0.06   # lệch ≤ 6% → s_bpm = 1
BPM_ZERO = 0.12   # lệch ≥ 12% → s_bpm = 0; ở giữa giảm tuyến tính
TEMPO_RATIOS = (1.0, 2.0, 0.5)  # xét cả nhịp gấp đôi / một nửa
KEY_SAME = 1.0
KEY_NEIGHBOUR = 0.8  # ±1 số cùng chữ, hoặc cùng số khác chữ


def bpm_delta(bpm_a: np.ndarray, bpm_b: np.ndarray) -> np.ndarray:
    """Độ lệch BPM tương đối so với bài đang phát a: min_r |b·r − a| / a, r ∈ {1, 2, 0.5}."""
    a = np.asarray(bpm_a, dtype=float)[..., None]
    b = np.asarray(bpm_b, dtype=float)[..., None] * np.array(TEMPO_RATIOS)
    return np.min(np.abs(b - a), axis=-1) / a[..., 0]


def s_bpm(bpm_a: np.ndarray, bpm_b: np.ndarray) -> np.ndarray:
    """1 khi lệch ≤ 6%, giảm tuyến tính về 0 ở 12%."""
    return np.clip((BPM_ZERO - bpm_delta(bpm_a, bpm_b)) / (BPM_ZERO - BPM_FULL), 0.0, 1.0)


def _key_score(num_a: np.ndarray, letter_a: np.ndarray, num_b: np.ndarray, letter_b: np.ndarray) -> np.ndarray:
    step = np.abs(num_a - num_b) % 12
    step = np.minimum(step, 12 - step)  # vòng: 12 ↔ 1 cách nhau 1
    same_letter = letter_a == letter_b
    score = np.where(same_letter & (step == 0), KEY_SAME, 0.0)
    neighbour = (same_letter & (step == 1)) | (~same_letter & (step == 0))
    return np.where(neighbour, KEY_NEIGHBOUR, score)


def s_key(camelot_a: Sequence[str] | np.ndarray, camelot_b: Sequence[str] | np.ndarray) -> np.ndarray:
    """Cùng mã = 1; ±1 số cùng chữ hoặc cùng số khác chữ = 0.8; còn lại (kể cả bước chéo) = 0."""
    return _key_score(*parse_camelot(camelot_a), *parse_camelot(camelot_b))


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


@dataclass(frozen=True)
class _Features:
    row_of_track: pd.Index
    bpm: np.ndarray
    key_num: np.ndarray
    key_letter: np.ndarray
    energy: np.ndarray


class RuleMix:
    """Điểm mix theo luật: w_b·s_bpm + w_k·s_key + w_e·s_energy.

    Dựng một lần từ bảng tracks (cột track_id, bpm, camelot, energy), rồi gọi trực tiếp:
    `RuleMix(tracks)(current_track_id, candidate_track_ids)` → mảng điểm mix theo đúng thứ tự ứng viên.
    """

    def __init__(self, tracks: pd.DataFrame, weights: MixWeights = MixWeights()):
        key_num, key_letter = parse_camelot(tracks["camelot"].to_numpy())
        self._features = _Features(
            row_of_track=pd.Index(tracks["track_id"]),
            bpm=tracks["bpm"].to_numpy(dtype=float),
            key_num=key_num,
            key_letter=key_letter,
            energy=tracks["energy"].to_numpy(dtype=float),
        )
        self.weights = weights

    def with_weights(self, weights: MixWeights) -> RuleMix:
        """Cùng danh mục, trọng số khác; không đọc lại bảng tracks (cho grid search #25)."""
        clone = copy.copy(self)  # bản sao nông: dùng chung mảng đặc trưng, chỉ đổi trọng số
        clone.weights = weights
        return clone

    def _rows(self, track_ids: Sequence[str]) -> np.ndarray:
        rows = self._features.row_of_track.get_indexer(track_ids)
        if (rows < 0).any():
            raise KeyError(f"không có trong tracks: {preview_ids(t for t, r in zip(track_ids, rows) if r < 0)}")
        return rows

    def components(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> MixComponents:
        f = self._features
        a, b = self._rows([current_track_id]), self._rows(list(candidate_track_ids))
        return MixComponents(
            bpm=s_bpm(f.bpm[a], f.bpm[b]),
            key=_key_score(f.key_num[a], f.key_letter[a], f.key_num[b], f.key_letter[b]),
            energy=s_energy(f.energy[a], f.energy[b]),
        )

    def __call__(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> np.ndarray:
        return self.components(current_track_id, candidate_track_ids).weighted(self.weights)
