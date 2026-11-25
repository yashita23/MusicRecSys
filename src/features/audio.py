"""Chuẩn hóa đặc trưng âm thanh thành tracks.parquet theo docs/data-contract.md (ticket #6)."""
from __future__ import annotations

import pandas as pd

from src.data.coverage import usable_mask
from src.data.preview import preview_ids
from src.data.schema import TRACKS_COLUMNS, SchemaError, count_train_occurrences, validate_tracks
from src.features.camelot import camelot_code


def make_tracks(tracks_raw: pd.DataFrame, sessions: pd.DataFrame) -> pd.DataFrame:
    """tracks_raw (#5) + sessions đã chia (#7) → bảng tracks đúng data-contract (đã qua validate_tracks).

    bpm = tempo (tempo của Spotify vốn là BPM); energy giữ nguyên (vốn trong 0–1); camelot từ (key, mode);
    n_occurrences = số lượt trong các phiên train. Bài không dùng được là lỗi của #5, nên báo lỗi chứ không bỏ.
    """
    unusable = tracks_raw.loc[~usable_mask(tracks_raw), "track_id"].tolist()
    if unusable:
        raise SchemaError("tracks_raw", [f"{len(unusable)} bài không dùng được (tempo ≤ 0, key ngoài 0–11, …; #5 lẽ ra đã loại): {preview_ids(unusable)}"])
    missing = sorted(set(sessions["track_id"]) - set(tracks_raw["track_id"]))
    if missing:
        raise SchemaError("tracks_raw", [f"{len(missing)} bài trong sessions không có trong tracks_raw: {preview_ids(missing)}"])

    tracks = tracks_raw.astype({"key": "int64", "mode": "int64"})
    tracks["bpm"] = tracks["tempo"].astype(float)
    tracks["camelot"] = [camelot_code(k, m) for k, m in zip(tracks["key"], tracks["mode"])]
    tracks["n_occurrences"] = tracks["track_id"].map(count_train_occurrences(sessions)).fillna(0).astype("int64")
    tracks = tracks[list(TRACKS_COLUMNS)].reset_index(drop=True)
    validate_tracks(tracks)
    return tracks
