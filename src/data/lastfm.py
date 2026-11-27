"""Cắt phiên nghe last.fm (Music4All-Onion) từ lịch sử nghe thành bảng phiên giống sessions.parquet của MPD (ticket #55).

Các bước, theo thứ tự:
1. Bỏ lượt có timestamp trước MIN_TIMESTAMP (last.fm ra đời năm 2002; dữ liệu có mốc 1970-01-01, tức epoch 0).
2. Sắp theo (user_id, timestamp, track_id) và cắt phiên mới khi đổi user hoặc khoảng nghỉ > gap_minutes.
   Khoảng nghỉ tính trên mọi lượt nghe, trước khi lọc bài, để phản ánh thời gian nghe thật.
3. Bỏ bài không có đặc trưng dùng được (không có trong essentia.parquet hoặc rhythm.bpm ≤ 0).
4. Gộp lượt lặp liên tiếp A A A → A (lặp không phải chuyển bài). Làm sau bước 3, nên A X A với X bị bỏ cũng thành A.
5. Giữ phiên ≥ min_length bài; đánh lại session_id 0…n−1 theo (user, thời điểm) và pos 0…n−1.
Rồi chia train/val/test theo user (assign_user_splits): mọi phiên của một user cùng một split.
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.data.schema import SchemaError
from src.data.split import RANDOM_SEED, assign_splits

GAP_MINUTES = 30
MIN_TIMESTAMP = pd.Timestamp("2002-01-01")
MIN_LENGTH = 5  # như MPD: đủ 5 bài seed (spec §7)


def usable_tracks(essentia: pd.DataFrame) -> set[str]:
    """Bài có đặc trưng dùng được: có trong essentia.parquet và rhythm.bpm > 0 (NaN không tính)."""
    return set(essentia.loc[essentia["rhythm.bpm"] > 0, "id"].astype(str))


def _track_codes(track: pd.Series) -> np.ndarray:
    """Mã bài theo thứ tự track_id (Categorical thì dùng categories đã sắp xếp, không đổi mảng)."""
    if isinstance(track.dtype, pd.CategoricalDtype):
        categories = track.cat.categories
        if not categories.is_monotonic_increasing:
            track = track.cat.reorder_categories(categories.sort_values())
        return track.cat.codes.to_numpy().astype(np.int64)
    return pd.factorize(track, sort=True)[0].astype(np.int64)


def _listening_order(frame: pd.DataFrame, origin: pd.Timestamp) -> np.ndarray:
    """Chỉ số sắp theo (user_id, timestamp, track_id). Gộp ba khóa thành một số int64 (user 17 bit, giây từ origin
    30 bit, mã bài 16 bit) nên argsort một mảng thay vì sắp nhiều khóa; vượt giới hạn thì dùng lexsort.
    Đo trên 10 triệu lượt: sort_values nhiều khóa 20 giây, lexsort 19 giây.
    """
    user = frame["user_id"].to_numpy().astype(np.int64)
    seconds = (frame["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64) - int(origin.timestamp()))
    track = _track_codes(frame["track_id"])
    fits = len(frame) == 0 or (
        user.min() >= 0 and user.max() < 2**17 and seconds.min() >= 0 and seconds.max() < 2**30 and track.max() < 2**16
    )
    if not fits:
        return np.lexsort((track, seconds, user))
    return np.argsort((user << 46) | (seconds << 16) | track, kind="stable")


def _snapshot(step: str, frame: pd.DataFrame, session: str | None) -> dict:
    return {
        "step": step,
        "events": int(len(frame)),
        "sessions": int(frame[session].nunique()) if session else None,
        "tracks": int(frame["track_id"].nunique()),
        "users": int(frame["user_id"].nunique()),
    }


def cut_sessions(
    events: pd.DataFrame,
    usable: Collection[str],
    gap_minutes: float = GAP_MINUTES,
    min_timestamp: pd.Timestamp = MIN_TIMESTAMP,
    min_length: int = MIN_LENGTH,
) -> tuple[pd.DataFrame, list[dict]]:
    """events (user_id, track_id, timestamp) → (phiên: session_id, user_id, pos, track_id; phễu từng bước).

    track_id nên là Categorical với categories đã sắp xếp khi chạy trên dữ liệu thật (tiết kiệm bộ nhớ, sắp nhanh);
    thứ tự giữa hai lượt cùng thời điểm khi đó vẫn theo track_id.
    """
    funnel = []
    frame = events[["user_id", "track_id", "timestamp"]]
    funnel.append(_snapshot("đầu vào", frame, None))

    frame = frame[frame["timestamp"] >= min_timestamp]
    funnel.append(_snapshot("bỏ timestamp hỏng", frame, None))

    frame = frame.iloc[_listening_order(frame, min_timestamp)]
    user = frame["user_id"].to_numpy()
    seconds = frame["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64)
    new = np.ones(len(frame), dtype=bool)
    new[1:] = (user[1:] != user[:-1]) | (np.diff(seconds) > gap_minutes * 60)
    frame = frame.assign(session=np.cumsum(new) - 1)
    funnel.append(_snapshot("cắt theo khoảng nghỉ", frame, "session"))

    frame = frame[frame["track_id"].isin(set(usable))]
    funnel.append(_snapshot("bỏ bài không có đặc trưng", frame, "session"))

    session, track = frame["session"].to_numpy(), frame["track_id"].to_numpy()
    keep = np.ones(len(frame), dtype=bool)
    keep[1:] = ~((session[1:] == session[:-1]) & (track[1:] == track[:-1]))
    frame = frame[keep]
    funnel.append(_snapshot("gộp lặp liên tiếp", frame, "session"))

    length = frame.groupby("session")["session"].transform("size")
    frame = frame[length >= min_length]
    funnel.append(_snapshot(f"giữ phiên ≥ {min_length} bài", frame, "session"))

    out = pd.DataFrame(
        {
            "session_id": pd.factorize(frame["session"])[0],
            "user_id": frame["user_id"].to_numpy(),
            "pos": frame.groupby("session").cumcount().to_numpy(),
            "track_id": frame["track_id"].astype(str).to_numpy(),
        }
    )
    return out, funnel


def assign_user_splits(sessions: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Thêm cột split theo user (80/10/10, cùng cách hoán vị với MPD): mọi phiên của một user cùng một split.

    Chia theo user chứ không theo phiên, để val/test không chứa thói quen chuyển bài của user đã học ở train.
    """
    split_of = assign_splits(sessions["user_id"], random_seed)
    out = sessions.assign(split=sessions["user_id"].map(split_of))
    return out[["session_id", "user_id", "split", "pos", "track_id"]]


ONION_TRACK_ID = r"^[A-Za-z0-9]{16}$"  # track_id của Music4All-Onion (không phải Spotify ID 22 ký tự)
SPLITS = ("train", "val", "test")
SENSITIVITY_GAPS = (15, 30, 60)


def validate_lastfm_sessions(sessions: pd.DataFrame) -> None:
    """Kiểm tra riêng cho phiên last.fm (validate_sessions của MPD đòi track_id 22 ký tự nên không dùng được)."""
    problems = []
    expected = ["session_id", "user_id", "split", "pos", "track_id"]
    if sessions.columns.tolist() != expected:
        raise SchemaError("lastfm_sessions", [f"cột phải là {expected}, đang là {sessions.columns.tolist()}"])
    if not sessions["track_id"].astype(str).str.fullmatch(ONION_TRACK_ID).all():
        problems.append("track_id phải là ID 16 ký tự của Music4All-Onion")
    if not sessions["split"].isin(SPLITS).all():
        problems.append(f"split phải thuộc {SPLITS}")
    if (sessions.groupby("user_id")["split"].nunique() > 1).any():
        problems.append("có user nằm ở hơn một split")
    ordered = sessions.sort_values(["session_id", "pos"])
    if not (ordered.groupby("session_id").cumcount() == ordered["pos"]).all():
        problems.append("pos của mỗi phiên phải là 0…n−1 liền mạch")
    if (sessions.groupby("session_id").size() < MIN_LENGTH).any():
        problems.append(f"có phiên ngắn hơn {MIN_LENGTH} bài")
    same = (ordered["session_id"].to_numpy()[1:] == ordered["session_id"].to_numpy()[:-1]) & (
        ordered["track_id"].to_numpy()[1:] == ordered["track_id"].to_numpy()[:-1]
    )
    if same.any():
        problems.append("có lượt lặp liên tiếp chưa gộp")
    if problems:
        raise SchemaError("lastfm_sessions", problems)


def load_events(path: Path, sample_users: float = 1.0, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Đọc listening_events.parquet với track_id dạng Categorical (categories sắp xếp) cho vừa RAM; lấy mẫu user nếu cần."""
    table = pq.read_table(path, columns=["user_id", "track_id", "timestamp"], read_dictionary=["track_id"])
    events = table.to_pandas()
    events["track_id"] = events["track_id"].cat.reorder_categories(sorted(events["track_id"].cat.categories))
    if sample_users < 1.0:
        users = np.sort(events["user_id"].unique())
        chosen = np.random.default_rng(random_seed).choice(users, size=round(sample_users * len(users)), replace=False)
        events = events[events["user_id"].isin(chosen)]
    return events


def run(onion_dir: str | Path, out_dir: str | Path, gap_minutes: float = GAP_MINUTES, sample_users: float = 1.0, random_seed: int = RANDOM_SEED) -> dict:
    """Đọc output của notebook 09, cắt phiên, chia theo user, ghi sessions.parquet và lastfm_log.json vào out_dir."""
    onion, out = Path(onion_dir), Path(out_dir)
    events = load_events(onion / "listening_events.parquet", sample_users, random_seed)
    usable = usable_tracks(pd.read_parquet(onion / "essentia.parquet", columns=["id", "rhythm.bpm"]))

    sensitivity = {}
    for gap in sorted({*SENSITIVITY_GAPS, gap_minutes}):
        cut, funnel = cut_sessions(events, usable, gap_minutes=gap)
        n_sessions = int(cut["session_id"].nunique())
        sensitivity[str(gap)] = {"sessions": n_sessions, "pairs": int(len(cut) - n_sessions)}
        if gap == gap_minutes:
            sessions, chosen_funnel = assign_user_splits(cut, random_seed), funnel
    validate_lastfm_sessions(sessions)

    out.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(out / "sessions.parquet", index=False)
    log = {
        "gap_minutes": gap_minutes,
        "min_timestamp": str(MIN_TIMESTAMP),
        "min_length": MIN_LENGTH,
        "sample_users": sample_users,
        "random_seed": random_seed,
        "funnel": chosen_funnel,
        "sensitivity": sensitivity,
        "splits": {
            split: {"users": int(g["user_id"].nunique()), "sessions": int(g["session_id"].nunique()), "events": int(len(g))}
            for split, g in sessions.groupby("split")
        },
    }
    (out / "lastfm_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Cắt phiên nghe last.fm từ Music4All-Onion (#55)")
    ap.add_argument("--onion", required=True, help="Thư mục có listening_events.parquet, essentia.parquet (Dataset mixnext-onion)")
    ap.add_argument("--out", required=True, help="Thư mục ra, ví dụ /kaggle/working/out/lastfm")
    ap.add_argument("--gap-minutes", type=float, default=GAP_MINUTES)
    ap.add_argument("--sample-users", type=float, default=1.0, help="Tỉ lệ user lấy mẫu nếu thiếu RAM (ghi vào log)")
    args = ap.parse_args(argv)
    log = run(args.onion, args.out, args.gap_minutes, args.sample_users)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return log


if __name__ == "__main__":
    main()
