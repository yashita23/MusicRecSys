"""Tra đặc trưng âm thanh theo track_id và đo các tính chất của cặp bài (a → b).

Dùng chung cho điểm mix theo luật (#12) và chỉ số độ mượt (#13). Ở đây chỉ có tính chất của cặp;
cách chấm điểm (ngưỡng, mức 0.8, …) là chuyện riêng của từng nơi dùng.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.preview import preview_ids
from src.features.camelot import camelot_distance, compatible_keys, parse_camelot, same_key

TEMPO_RATIOS = (1.0, 2.0, 0.5)  # xét cả nhịp gấp đôi / một nửa


def bpm_delta(bpm_a: np.ndarray, bpm_b: np.ndarray) -> np.ndarray:
    """Độ lệch BPM tương đối so với bài đang phát a: min_r |b·r − a| / a, r ∈ {1, 2, 0.5}."""
    a = np.asarray(bpm_a, dtype=float)[..., None]
    b = np.asarray(bpm_b, dtype=float)[..., None] * np.array(TEMPO_RATIOS)
    return np.min(np.abs(b - a), axis=-1) / a[..., 0]


@dataclass(frozen=True)
class TrackFeatures:
    """Đặc trưng của cả danh mục. Các phương thức nhận dòng a và dòng b; a có thể là một dòng để so với nhiều dòng b."""

    row_of_track: pd.Index
    bpm: np.ndarray
    key_num: np.ndarray
    key_letter: np.ndarray
    energy: np.ndarray

    @classmethod
    def from_tracks(cls, tracks: pd.DataFrame) -> TrackFeatures:
        """Từ bảng có cột track_id, bpm, camelot, energy (tracks.parquet)."""
        key_num, key_letter = parse_camelot(tracks["camelot"].to_numpy())
        return cls(
            row_of_track=pd.Index(tracks["track_id"]),
            bpm=tracks["bpm"].to_numpy(dtype=float),
            key_num=key_num,
            key_letter=key_letter,
            energy=tracks["energy"].to_numpy(dtype=float),
        )

    def rows(self, track_ids: Sequence[str]) -> np.ndarray:
        rows = self.row_of_track.get_indexer(track_ids)
        if (rows < 0).any():
            raise KeyError(f"không có trong tracks: {preview_ids(t for t, r in zip(track_ids, rows) if r < 0)}")
        return rows

    def bpm_delta(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        return bpm_delta(self.bpm[rows_a], self.bpm[rows_b])

    def same_key(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        return same_key(self.key_num[rows_a], self.key_letter[rows_a], self.key_num[rows_b], self.key_letter[rows_b])

    def compatible_keys(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        return compatible_keys(self.key_num[rows_a], self.key_letter[rows_a], self.key_num[rows_b], self.key_letter[rows_b])

    def camelot_distance(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        return camelot_distance(self.key_num[rows_a], self.key_letter[rows_a], self.key_num[rows_b], self.key_letter[rows_b])

    def energy_gap(self, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
        return np.abs(self.energy[rows_a] - self.energy[rows_b])
