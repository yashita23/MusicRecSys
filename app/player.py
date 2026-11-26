"""Dữ liệu và logic của app, tách khỏi giao diện Streamlit để test được (#11).

Prototype lấy ứng viên từ một file candidates. Demo v1 (#21) sẽ tính ứng viên bằng vector phiên và embedding,
còn phần xếp lại theo α (rerank, RuleMix) giữ nguyên.
"""
from __future__ import annotations

import pandas as pd

from app.config import AppConfig
from src.data.schema import validate_candidates, validate_eval_targets, validate_tracks
from src.rerank.combine import TOP_K, MixScorer, rerank
from src.rerank.mix_rule import RuleMix

TRACK_INFO = ["track_id", "track_name", "artist_name", "bpm", "camelot", "energy"]


class Player:
    """Các playlist chọn được, bài đang phát và hàng đợi gợi ý theo α."""

    def __init__(self, tracks: pd.DataFrame, eval_targets: pd.DataFrame, candidates: pd.DataFrame, mix: MixScorer | None = None):
        self._info = tracks[TRACK_INFO].set_index("track_id", drop=False)
        self._seed = eval_targets.set_index("session_id")["seed_track_ids"].map(list)
        self._candidates = {sid: g.sort_values("rank") for sid, g in candidates.groupby("session_id", sort=True)}
        self._mix = mix if mix is not None else RuleMix(tracks)

    @classmethod
    def from_config(cls, cfg: AppConfig) -> Player:
        """Đọc ba file trong config và kiểm tra schema theo docs/data-contract.md."""
        tracks, eval_targets, candidates = (pd.read_parquet(p) for p in (cfg.tracks, cfg.eval_targets, cfg.candidates))
        validate_tracks(tracks)
        validate_eval_targets(eval_targets)
        validate_candidates(candidates)
        return cls(tracks, eval_targets, candidates)

    @property
    def playlists(self) -> list[int]:
        """Phiên có ứng viên, theo session_id tăng dần."""
        return [int(sid) for sid in self._candidates]

    def seed(self, session_id: int) -> pd.DataFrame:
        return self._info.loc[self._seed[session_id]].reset_index(drop=True)

    def now_playing(self, session_id: int) -> pd.Series:
        """Bài đang phát là bài cuối của seed."""
        return self._info.loc[self._seed[session_id][-1]]

    def queue(self, session_id: int, alpha: float, k: int = TOP_K) -> pd.DataFrame:
        """Top-k sau khi xếp lại theo điểm cuối, kèm thông tin bài để hiển thị."""
        group = self._candidates[session_id]
        ranked = rerank(group["track_id"].tolist(), group["taste_score"].to_numpy(), self.now_playing(session_id)["track_id"], self._mix, alpha, k)
        info = self._info.loc[ranked["track_id"]].reset_index(drop=True).drop(columns="track_id")
        return pd.concat([ranked, info], axis=1)
