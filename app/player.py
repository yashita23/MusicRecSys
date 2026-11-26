"""Dữ liệu và logic của app, tách khỏi giao diện Streamlit để test được (#11).

Prototype lấy ứng viên từ một file candidates. Demo v1 (#21) sẽ tính ứng viên bằng vector phiên và embedding,
còn phần xếp lại theo α (rerank, RuleMix) giữ nguyên.
"""
from __future__ import annotations

import pandas as pd

from app.config import AppConfig
from src.data.preview import preview_ids
from src.data.schema import SchemaError, validate_candidates, validate_eval_targets, validate_tracks
from src.rerank.combine import TOP_K, MixScorer, rerank
from src.rerank.mix_rule import RuleMix

TRACK_INFO = ["track_id", "track_name", "artist_name", "bpm", "camelot", "energy"]


def _cross_table_problems(tracks: pd.DataFrame, eval_targets: pd.DataFrame, candidates: pd.DataFrame) -> list[str]:
    """Các bảng khớp nhau như docs/data-contract.md: phiên của candidates có trong eval_targets, cùng một split,
    mọi bài trong candidates và seed có trong tracks. (validate_dataset cần cả sessions, app không đọc bảng đó.)"""
    problems = []
    sessions = set(candidates["session_id"])
    targets = eval_targets[eval_targets["session_id"].isin(sessions)]
    unknown = sessions - set(targets["session_id"])
    if unknown:
        problems.append(f"{len(unknown)} phiên trong candidates không có trong eval_targets (ví dụ {sorted(unknown)[:3]})")
    splits = sorted(targets["split"].unique())
    if len(splits) > 1:
        problems.append(f"candidates trộn phiên của nhiều split: {', '.join(splits)}")
    known = set(tracks["track_id"])
    used = set(candidates["track_id"]).union(*map(set, targets["seed_track_ids"]))
    if used - known:
        problems.append(f"candidates hoặc seed có bài không có trong tracks: {preview_ids(used - known)}")
    return problems


class Player:
    """Các playlist chọn được, bài đang phát và hàng đợi gợi ý theo α."""

    def __init__(self, tracks: pd.DataFrame, eval_targets: pd.DataFrame, candidates: pd.DataFrame, mix: MixScorer | None = None):
        problems = _cross_table_problems(tracks, eval_targets, candidates)
        if problems:
            raise SchemaError("app", problems)
        targets = eval_targets[eval_targets["session_id"].isin(candidates["session_id"])].set_index("session_id")
        self.split: str = targets["split"].iloc[0]
        self._seed = targets["seed_track_ids"].map(list)
        self._info = tracks[TRACK_INFO].set_index("track_id", drop=False)
        self._candidates = {sid: g.sort_values("rank") for sid, g in candidates.groupby("session_id", sort=True)}
        self._mix = mix if mix is not None else RuleMix(tracks)  # điểm mix học (#23) cắm vào đây

    @classmethod
    def from_config(cls, cfg: AppConfig) -> Player:
        """Đọc ba file trong config và kiểm tra schema theo docs/data-contract.md."""
        tracks, eval_targets, candidates = (pd.read_parquet(p) for p in cfg.files)
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

    def queue(self, session_id: int, alpha: float) -> pd.DataFrame:
        """Top-10 sau khi xếp lại theo điểm cuối, kèm thông tin bài để hiển thị."""
        group = self._candidates[session_id]
        ranked = rerank(group["track_id"].tolist(), group["taste_score"].to_numpy(), self.now_playing(session_id)["track_id"], self._mix, alpha, TOP_K)
        info = self._info.loc[ranked["track_id"]].reset_index(drop=True).drop(columns="track_id")
        return pd.concat([ranked, info], axis=1)
