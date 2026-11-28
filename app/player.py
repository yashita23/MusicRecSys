"""Logic của app, tách khỏi giao diện Streamlit để test được (#11, #21).

MixNext chỉ gợi ý: người dùng chọn bài tiếp theo trong hàng đợi, app không tự phát hay tự chuyển bài. Mỗi lần chọn,
bài đó nối vào cuối phiên và thành bài đang phát; vector phiên và top-100 ứng viên tính lại bằng numpy từ embedding
trong artifacts/ (SessionVectorTaste), rồi xếp lại theo α bằng rerank và điểm mix theo luật.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES
from src.export_artifacts import Artifacts, load_artifacts
from src.rerank.combine import TOP_K, MixScorer, rerank
from src.rerank.explain import explain

TRACK_INFO = ["track_id", "track_name", "artist_name", "bpm", "camelot", "energy"]
CACHE_SESSIONS = 64  # số phiên giữ sẵn ứng viên: kéo α hay quay lại không phải tính lại


@dataclass(frozen=True)
class Session:
    """Một phiên nghe trong app: seed của playlist demo cộng các bài người dùng đã chọn, theo thứ tự."""

    playlist_id: int
    tracks: tuple[str, ...]
    n_seed: int

    @property
    def chosen(self) -> tuple[str, ...]:
        return self.tracks[self.n_seed :]

    @property
    def current(self) -> str:
        return self.tracks[-1]


class Player:
    """Playlist demo, bài đang phát, hàng đợi theo α, và các thao tác chọn / quay lại / bắt đầu lại."""

    def __init__(self, artifacts: Artifacts, mix: MixScorer | None = None):
        self._artifacts = artifacts
        self._taste = artifacts.taste_model()
        self._mix = mix if mix is not None else artifacts.rule_mix()  # điểm mix học (#23, #28) cắm vào đây
        self._info = artifacts.tracks[TRACK_INFO].set_index("track_id", drop=False)
        self._seeds = artifacts.demo.set_index("session_id")["seed_track_ids"].map(tuple)
        self._candidates: OrderedDict[tuple[str, ...], tuple[list[str], np.ndarray]] = OrderedDict()

    @classmethod
    def from_dir(cls, path: str | Path) -> Player:
        """Đọc artifacts/ bằng load_artifacts (#20): chỉ numpy và pandas."""
        return cls(load_artifacts(path))

    @property
    def playlists(self) -> list[int]:
        return sorted(int(sid) for sid in self._seeds.index)

    @property
    def default_alpha(self) -> float:
        """α của mix theo luật trong mix_params.json (#19, #25 ghi)."""
        return self._artifacts.mix_params.rule_alpha

    @property
    def model(self) -> str:
        return self._artifacts.model

    @property
    def tracks_with_vectors(self) -> list[str]:
        return self._artifacts.track_index["track_id"].tolist()

    def start(self, playlist_id: int) -> Session:
        seed = self._seeds[playlist_id]
        return Session(playlist_id, tuple(seed), len(seed))

    def now_playing(self, session: Session) -> pd.Series:
        return self._info.loc[session.current]

    def _session_candidates(self, session: Session) -> tuple[list[str], np.ndarray]:
        """Top-100 ứng viên của phiên (không có bài nào đã trong phiên); nhớ theo phiên để kéo α chỉ phải xếp lại."""
        key = session.tracks
        if key in self._candidates:
            self._candidates.move_to_end(key)
        else:
            self._candidates[key] = self._taste.candidates(list(key), k=N_CANDIDATES)
            if len(self._candidates) > CACHE_SESSIONS:
                self._candidates.popitem(last=False)
        return self._candidates[key]

    def queue(self, session: Session, alpha: float, k: int | None = TOP_K) -> pd.DataFrame:
        """Hàng đợi top-k (k=None: cả 100 ứng viên) xếp theo điểm cuối, kèm thông tin bài và một dòng giải thích."""
        ids, taste = self._session_candidates(session)
        ranked = rerank(ids, taste, session.current, self._mix, alpha, k)
        info = self._info.loc[ranked["track_id"]].reset_index(drop=True).drop(columns="track_id")
        out = pd.concat([ranked, info], axis=1)
        playing = self.now_playing(session)
        out["explanation"] = [explain(playing, row) for row in info.to_dict("records")]
        return out

    def choose(self, session: Session, track_id: str, alpha: float) -> Session:
        """Người dùng chọn track_id trong hàng đợi hiện tại (ở α đang dùng): nối vào phiên và thành bài đang phát."""
        if track_id not in set(self.queue(session, alpha)["track_id"]):
            raise ValueError(f"{track_id} không có trong hàng đợi hiện tại: chỉ chọn được bài trong hàng đợi")
        return Session(session.playlist_id, session.tracks + (track_id,), session.n_seed)

    def back(self, session: Session) -> Session:
        """Bỏ bài vừa chọn; đang ở seed thì giữ nguyên."""
        if not session.chosen:
            return session
        return Session(session.playlist_id, session.tracks[:-1], session.n_seed)

    def restart(self, session: Session) -> Session:
        return self.start(session.playlist_id)

    def history(self, session: Session) -> pd.DataFrame:
        """Mọi bài của phiên theo thứ tự (seed rồi các bài đã chọn), kèm giải thích bước chuyển từ bài trước."""
        info = self._info.loc[list(session.tracks)].reset_index(drop=True)
        records = info.to_dict("records")
        explanations = [""] + [explain(a, b) for a, b in zip(records, records[1:])]
        source = ["seed"] * session.n_seed + ["chọn"] * len(session.chosen)
        return info.assign(step=np.arange(len(info)), source=source, explanation=explanations)
