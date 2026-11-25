"""Điểm mix theo luật DJ cho cặp (bài đang phát a, ứng viên b), dạng vector (ticket #12, spec §6)."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

BPM_FULL = 0.06   # lệch ≤ 6% → s_bpm = 1
BPM_ZERO = 0.12   # lệch ≥ 12% → s_bpm = 0; ở giữa giảm tuyến tính
TEMPO_RATIOS = (1.0, 2.0, 0.5)  # xét cả nhịp gấp đôi / một nửa


def bpm_delta(bpm_a: np.ndarray, bpm_b: np.ndarray) -> np.ndarray:
    """Độ lệch BPM tương đối so với bài đang phát a: min_r |b·r − a| / a, r ∈ {1, 2, 0.5}."""
    a = np.asarray(bpm_a, dtype=float)[..., None]
    b = np.asarray(bpm_b, dtype=float)[..., None] * np.array(TEMPO_RATIOS)
    return np.min(np.abs(b - a), axis=-1) / a[..., 0]


def s_bpm(bpm_a: np.ndarray, bpm_b: np.ndarray) -> np.ndarray:
    return np.clip((BPM_ZERO - bpm_delta(bpm_a, bpm_b)) / (BPM_ZERO - BPM_FULL), 0.0, 1.0)


KEY_SAME = 1.0
KEY_NEIGHBOUR = 0.8  # ±1 số cùng chữ, hoặc cùng số khác chữ


def parse_camelot(codes) -> tuple[np.ndarray, np.ndarray]:
    """'8A' → (8, 'A'): mảng số 1–12 và mảng chữ A/B."""
    codes = np.asarray(codes, dtype=str)
    return np.char.rstrip(codes, "AB").astype(int), np.char.lstrip(codes, "0123456789")


def key_score(num_a: np.ndarray, letter_a: np.ndarray, num_b: np.ndarray, letter_b: np.ndarray) -> np.ndarray:
    """s_key từ mã Camelot đã tách; khoảng cách số tính vòng (12 ↔ 1 cách nhau 1)."""
    step = np.abs(num_a - num_b) % 12
    step = np.minimum(step, 12 - step)
    same_letter = letter_a == letter_b
    score = np.where(same_letter & (step == 0), KEY_SAME, 0.0)
    neighbour = (same_letter & (step == 1)) | (~same_letter & (step == 0))
    return np.where(neighbour, KEY_NEIGHBOUR, score)


def s_key(camelot_a, camelot_b) -> np.ndarray:
    return key_score(*parse_camelot(camelot_a), *parse_camelot(camelot_b))


def s_energy(energy_a: np.ndarray, energy_b: np.ndarray) -> np.ndarray:
    return 1.0 - np.abs(np.asarray(energy_a, dtype=float) - np.asarray(energy_b, dtype=float))


@dataclass(frozen=True)
class MixWeights:
    """Trọng số của ba thành phần; không tự chuẩn hóa (ablation ở #27 tự chuẩn hóa lại nếu cần)."""

    bpm: float = 0.4
    key: float = 0.4
    energy: float = 0.2


class RuleMix:
    """Điểm mix theo luật: mix(bài đang phát, ứng viên) = w_b·s_bpm + w_k·s_key + w_e·s_energy.

    Dựng một lần từ bảng tracks (cột track_id, bpm, camelot, energy), rồi gọi
    mix(current_track_id, candidate_track_ids) → mảng điểm mix theo đúng thứ tự ứng viên.
    """

    def __init__(self, tracks: pd.DataFrame, weights: MixWeights = MixWeights()):
        self.weights = weights
        self._row = pd.Index(tracks["track_id"])
        self._bpm = tracks["bpm"].to_numpy(dtype=float)
        self._key_num, self._key_letter = parse_camelot(tracks["camelot"].to_numpy())
        self._energy = tracks["energy"].to_numpy(dtype=float)

    def _rows(self, track_ids: Sequence[str]) -> np.ndarray:
        rows = self._row.get_indexer(track_ids)
        if (rows < 0).any():
            unknown = [t for t, r in zip(track_ids, rows) if r < 0]
            raise KeyError(f"không có trong tracks: {', '.join(unknown[:5])}")
        return rows

    def __call__(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> np.ndarray:
        a = self._rows([current_track_id])
        b = self._rows(list(candidate_track_ids))
        w = self.weights
        return (
            w.bpm * s_bpm(self._bpm[a], self._bpm[b])
            + w.key * key_score(self._key_num[a], self._key_letter[a], self._key_num[b], self._key_letter[b])
            + w.energy * s_energy(self._energy[a], self._energy[b])
        )
