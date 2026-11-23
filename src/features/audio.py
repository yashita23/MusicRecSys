"""Chuẩn hóa đặc trưng âm thanh thành tracks.parquet theo docs/data-contract.md (ticket #6)."""
from __future__ import annotations

import pandas as pd

from src.data.coverage import usable_mask
from src.data.schema import TRACKS_COLUMNS, count_train_occurrences
from src.features.camelot import camelot_code


def make_tracks(tracks_raw: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """tracks_raw (#5) + sessions đã chia (#7) → bảng tracks đúng data-contract.

    bpm = tempo (tempo của Spotify vốn là BPM), camelot từ (key, mode),
    n_occurrences = số lượt trong các phiên train. Bài không dùng được là lỗi của #5, nên báo lỗi chứ không bỏ.
    """
    bad = ~usable_mask(tracks_raw)
    if bad.any():
        raise ValueError(f"tracks_raw có {int(bad.sum())} bài không dùng được (tempo ≤ 0, key ngoài 0–11, …); #5 lẽ ra đã loại")
    missing = set(sessions["track_id"]) - set(tracks_raw["track_id"])
    if missing:
        raise ValueError(f"{len(missing)} bài trong sessions không có trong tracks_raw")

    tracks = tracks_raw.copy()
    tracks["bpm"] = tracks["tempo"].astype(float)
    tracks["camelot"] = [camelot_code(int(k), int(m)) for k, m in zip(tracks["key"], tracks["mode"])]
    tracks["n_occurrences"] = tracks["track_id"].map(count_train_occurrences(sessions)).fillna(0).astype("int64")
    return tracks[list(TRACKS_COLUMNS)].reset_index(drop=True)
