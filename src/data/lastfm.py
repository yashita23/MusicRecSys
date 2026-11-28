"""Cắt phiên nghe last.fm (Music4All-Onion) từ lịch sử nghe thành bảng phiên giống sessions.parquet của MPD (ticket #55).

Các bước, theo thứ tự:
1. Bỏ lượt có timestamp trước MIN_TIMESTAMP (last.fm ra đời năm 2002; dữ liệu có mốc 1970-01-01, tức epoch 0).
2. Sắp theo (user_id, timestamp, track_id) và cắt phiên mới khi đổi user hoặc khoảng nghỉ > gap_minutes.
   Khoảng nghỉ tính trên mọi lượt nghe, trước khi lọc bài, để phản ánh thời gian nghe thật.
3. Bỏ bài không có đặc trưng dùng được (không có trong essentia.parquet hoặc rhythm.bpm ≤ 0).
4. Gộp lượt lặp liên tiếp A A A → A (lặp không phải chuyển bài). Làm sau bước 3, nên A X A với X bị bỏ cũng thành A.
5. Giữ phiên ≥ min_length bài; đánh lại session_id 0…n−1 theo (user, thời điểm) và pos 0…n−1.
Rồi chia train/val/test theo user (assign_user_splits): mọi phiên của một user cùng một split.

Giới hạn: hai lượt cùng một giây được xếp theo track_id (không biết thứ tự thật), nên có thể sinh cặp chuyển bài
không có thật; hiếm vì một bài dài vài phút.

Chạy trên 253 triệu lượt: lọc và sắp một lần (prepare), rồi với mỗi khoảng nghỉ chỉ tính ranh giới phiên trên mảng
mã số (_cut); chỉ khoảng nghỉ được chọn mới dựng bảng phiên đầy đủ (_sessions_frame).
"""
from __future__ import annotations

import argparse
import json
from collections.abc import Collection
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.compute as pc
import pyarrow.parquet as pq

from src.data.schema import SPLITS, SchemaError
from src.data.split import RANDOM_SEED, assign_splits

GAP_MINUTES = 30
SENSITIVITY_GAPS = (15, 30, 60)
MIN_TIMESTAMP = pd.Timestamp("2002-01-01")
MIN_LENGTH = 5  # theo #55 (đủ seed 5 bài); MPD giữ phiên ≥ 10 bài (schema.MIN_SESSION_LENGTH)
ONION_TRACK_ID = r"^[A-Za-z0-9]{16}$"  # track_id của Music4All-Onion (không phải Spotify ID 22 ký tự)
SESSION_COLUMNS = ["session_id", "user_id", "split", "pos", "track_id"]


def usable_tracks(essentia: pd.DataFrame) -> set[str]:
    """Bài có đặc trưng dùng được: có trong essentia.parquet và rhythm.bpm > 0 (NaN không tính)."""
    return set(essentia.loc[essentia["rhythm.bpm"] > 0, "id"].astype(str))


@dataclass
class Prepared:
    """Lượt nghe sau bước 1, đã sắp theo (user, timestamp, track_id), dạng mảng mã số dùng chung cho mọi khoảng nghỉ."""

    user: np.ndarray       # int32
    seconds: np.ndarray    # int64, giây kể từ min_timestamp
    track: np.ndarray      # int32, mã bài theo thứ tự track_id
    track_ids: np.ndarray  # mã → track_id
    usable: np.ndarray     # mã → có đặc trưng dùng được
    funnel: list[dict]     # phễu đến hết bước 1


def _track_codes(track: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """(mã bài, danh mục track_id đã sắp xếp); Categorical thì dùng sẵn mã, không tạo mảng chuỗi."""
    if isinstance(track.dtype, pd.CategoricalDtype):
        if not track.cat.categories.is_monotonic_increasing:
            track = track.cat.reorder_categories(track.cat.categories.sort_values())
        codes, categories = track.cat.codes.to_numpy(), track.cat.categories.to_numpy()
    else:
        codes, categories = pd.factorize(track, sort=True)
    if len(codes) and codes.min() < 0:
        raise ValueError(f"{int((codes < 0).sum())} lượt nghe có track_id rỗng")
    return codes.astype(np.int32), np.asarray(categories, dtype=object)


def _runs(values: np.ndarray) -> int:
    """Số đoạn giá trị liên tiếp; với mảng đã sắp chính là số giá trị khác nhau, khỏi np.unique trên cả mảng."""
    return int(len(values) > 0) + int(np.count_nonzero(values[1:] != values[:-1]))


def _count(step: str, user: np.ndarray, track: np.ndarray, n_tracks: int, session: np.ndarray | None) -> dict:
    """Một dòng phễu; user (và session) phải đã sắp."""
    return {
        "step": step,
        "events": int(len(user)),
        "sessions": _runs(session) if session is not None else None,
        "tracks": int(np.count_nonzero(np.bincount(track, minlength=n_tracks))),
        "users": _runs(user),
    }


def prepare(events: pd.DataFrame, usable: Collection[str], min_timestamp: pd.Timestamp = MIN_TIMESTAMP) -> Prepared:
    """Bước 1 và sắp xếp. Gộp (user 17 bit, giây 31 bit, mã bài 16 bit) thành một khóa int64 để argsort một mảng
    (đo trên 10 triệu lượt: cả cut_sessions 6 giây, so với 32 giây khi sort_values nhiều khóa); vượt giới hạn thì lexsort.
    """
    track, track_ids = _track_codes(events["track_id"])
    user = events["user_id"].to_numpy().astype(np.int64)
    seconds = events["timestamp"].to_numpy().astype("datetime64[s]").astype(np.int64) - int(min_timestamp.timestamp())
    funnel = [_count("đầu vào", np.sort(user), track, len(track_ids), None)]

    valid = seconds >= 0
    user, seconds, track = user[valid], seconds[valid], track[valid]
    funnel.append(_count("bỏ timestamp hỏng", np.sort(user), track, len(track_ids), None))

    fits = len(user) == 0 or (user.min() >= 0 and user.max() < 2**17 and seconds.max() < 2**31 and len(track_ids) <= 2**16)
    if fits:
        order = np.argsort((user << 47) | (seconds << 16) | track.astype(np.int64), kind="stable")
    else:
        order = np.lexsort((track, seconds, user))
    usable_mask = pd.Index(track_ids).isin(list(usable))  # tra bảng băm; np.isin trên mảng chuỗi mất ~90 giây
    return Prepared(user[order].astype(np.int32), seconds[order], track[order], track_ids, usable_mask, funnel)


def _session_index(prep: Prepared, gap_minutes: float) -> np.ndarray:
    """Chỉ số phiên của từng lượt (trên mọi lượt sau bước 1): phiên mới khi đổi user hoặc nghỉ > gap_minutes."""
    new = np.ones(len(prep.user), dtype=bool)
    new[1:] = (prep.user[1:] != prep.user[:-1]) | (np.diff(prep.seconds) > gap_minutes * 60)
    return np.cumsum(new) - 1


def _cut(prep: Prepared, gap_minutes: float, min_length: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Bước 2–5: (chỉ số các lượt được giữ trong prep, chỉ số phiên của chúng, phễu từ bước 2)."""
    n_tracks = len(prep.track_ids)
    session = _session_index(prep, gap_minutes)
    funnel = [_count("cắt theo khoảng nghỉ", prep.user, prep.track, n_tracks, session)]

    kept = np.flatnonzero(prep.usable[prep.track])
    funnel.append(_count("bỏ bài không có đặc trưng", prep.user[kept], prep.track[kept], n_tracks, session[kept]))

    s, t = session[kept], prep.track[kept]
    repeat = np.zeros(len(kept), dtype=bool)
    repeat[1:] = (s[1:] == s[:-1]) & (t[1:] == t[:-1])
    kept = kept[~repeat]
    funnel.append(_count("gộp lặp liên tiếp", prep.user[kept], prep.track[kept], n_tracks, session[kept]))

    s = session[kept]
    lengths = np.bincount(s, minlength=int(session[-1]) + 1 if len(session) else 0)
    kept = kept[lengths[s] >= min_length]
    funnel.append(_count(f"giữ phiên ≥ {min_length} bài", prep.user[kept], prep.track[kept], n_tracks, session[kept]))
    return kept, session[kept], funnel


def _sessions_frame(prep: Prepared, kept: np.ndarray, session: np.ndarray) -> pd.DataFrame:
    """Bảng phiên (session_id 0…n−1, user_id, pos, track_id) từ các lượt được giữ, đã theo thứ tự phiên."""
    new = np.ones(len(kept), dtype=bool)
    new[1:] = session[1:] != session[:-1]
    starts = np.flatnonzero(new)
    pos = np.arange(len(kept)) - np.repeat(starts, np.diff(np.append(starts, len(kept))))
    return pd.DataFrame(
        {"session_id": np.cumsum(new) - 1, "user_id": prep.user[kept], "pos": pos, "track_id": prep.track_ids[prep.track[kept]]}
    )


def cut_sessions(
    events: pd.DataFrame,
    usable: Collection[str],
    gap_minutes: float = GAP_MINUTES,
    min_timestamp: pd.Timestamp = MIN_TIMESTAMP,
    min_length: int = MIN_LENGTH,
) -> tuple[pd.DataFrame, list[dict]]:
    """events (user_id, track_id, timestamp) → (phiên: session_id, user_id, pos, track_id; phễu từng bước).

    track_id nên là Categorical khi chạy trên dữ liệu thật (tiết kiệm bộ nhớ); thứ tự hai lượt cùng thời điểm theo track_id.
    """
    prep = prepare(events, usable, min_timestamp)
    kept, session, funnel = _cut(prep, gap_minutes, min_length)
    return _sessions_frame(prep, kept, session), prep.funnel + funnel


def assign_user_splits(sessions: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Thêm cột split theo user (80/10/10, cùng cách hoán vị với MPD): mọi phiên của một user cùng một split.

    Chia theo user chứ không theo phiên, để val/test không chứa thói quen chuyển bài của user đã học ở train.
    assign_splits nhận id bất kỳ (tên tham số session_ids là của MPD); ở đây truyền user_id.
    """
    split_of = assign_splits(sessions["user_id"], random_seed)
    out = sessions.assign(split=sessions["user_id"].map(split_of))
    return out[SESSION_COLUMNS]


def validate_lastfm_sessions(sessions: pd.DataFrame) -> None:
    """Kiểm tra riêng cho phiên last.fm (validate_sessions của MPD đòi track_id 22 ký tự nên không dùng được).

    Làm trên mảng numpy theo thứ tự (session_id, pos); chỉ sắp lại khi bảng chưa theo thứ tự đó.
    Regex chỉ chạy trên các track_id khác nhau, không trên cả bảng.
    """
    if sessions.columns.tolist() != SESSION_COLUMNS:
        raise SchemaError("lastfm_sessions", [f"cột phải là {SESSION_COLUMNS}, đang là {sessions.columns.tolist()}"])
    if np.any(np.diff(sessions["session_id"].to_numpy()) < 0):
        sessions = sessions.sort_values(["session_id", "pos"])
    sid, pos = sessions["session_id"].to_numpy(), sessions["pos"].to_numpy()
    track_codes, track_ids = pd.factorize(sessions["track_id"])

    problems = []
    if not pd.Series(track_ids).astype(str).str.fullmatch(ONION_TRACK_ID).all():
        problems.append("track_id phải là ID 16 ký tự của Music4All-Onion")
    if not sessions["split"].isin(SPLITS).all():
        problems.append(f"split phải thuộc {SPLITS}")
    if (sessions.groupby("user_id")["split"].nunique() > 1).any():
        problems.append("có user nằm ở hơn một split")
    if len(sid) and (sid[0] != 0 or np.any(np.diff(sid) > 1)):
        problems.append("session_id phải liền mạch 0…n−1")

    new = np.ones(len(sid), dtype=bool)
    new[1:] = sid[1:] != sid[:-1]
    starts = np.flatnonzero(new)
    lengths = np.diff(np.append(starts, len(sid)))
    if not np.array_equal(pos, np.arange(len(sid)) - np.repeat(starts, lengths)):
        problems.append("pos của mỗi phiên phải là 0…n−1 liền mạch")
    if np.any(lengths < MIN_LENGTH):
        problems.append(f"có phiên ngắn hơn {MIN_LENGTH} bài")
    if np.any(~new[1:] & (track_codes[1:] == track_codes[:-1])):
        problems.append("có lượt lặp liên tiếp chưa gộp")
    if problems:
        raise SchemaError("lastfm_sessions", problems)


def load_events(path: Path, sample_users: float = 1.0, random_seed: int = RANDOM_SEED) -> tuple[pd.DataFrame, int]:
    """Đọc listening_events.parquet (track_id dạng Categorical cho vừa RAM); lấy mẫu user ngay khi đọc nếu < 1.

    Trả về (lượt nghe, tổng số lượt trong file trước khi lấy mẫu).
    """
    total = pq.ParquetFile(path).metadata.num_rows
    filters = None
    if sample_users < 1.0:
        users = np.unique(pq.read_table(path, columns=["user_id"])["user_id"].to_numpy())
        chosen = np.random.default_rng(random_seed).choice(users, size=round(sample_users * len(users)), replace=False)
        filters = pc.field("user_id").isin(chosen.tolist())
    table = pq.read_table(path, columns=["user_id", "track_id", "timestamp"], read_dictionary=["track_id"], filters=filters)
    return table.to_pandas(), total


def _gap_key(gap: float) -> str:
    return str(int(gap)) if float(gap).is_integer() else str(gap)


def run(onion_dir: str | Path, out_dir: str | Path, gap_minutes: float = GAP_MINUTES, sample_users: float = 1.0, random_seed: int = RANDOM_SEED) -> dict:
    """Đọc output của notebook 09, cắt phiên, chia theo user, ghi sessions.parquet và lastfm_log.json vào out_dir."""
    onion, out = Path(onion_dir), Path(out_dir)
    events, total = load_events(onion / "listening_events.parquet", sample_users, random_seed)
    usable = usable_tracks(pd.read_parquet(onion / "essentia.parquet", columns=["id", "rhythm.bpm"]))
    prep = prepare(events, usable)
    del events  # từ đây chỉ dùng mảng mã số trong prep

    sensitivity, chosen = {}, None
    for gap in sorted({*SENSITIVITY_GAPS, gap_minutes}):
        kept, session, funnel = _cut(prep, gap, MIN_LENGTH)
        n_sessions = funnel[-1]["sessions"]
        sensitivity[_gap_key(gap)] = {"sessions": n_sessions, "pairs": int(len(kept) - n_sessions)}
        if gap == gap_minutes:
            chosen = (kept, session, funnel)
        del kept, session
    kept, session, funnel = chosen
    sessions = assign_user_splits(_sessions_frame(prep, kept, session), random_seed)
    validate_lastfm_sessions(sessions)

    out.mkdir(parents=True, exist_ok=True)
    sessions.to_parquet(out / "sessions.parquet", index=False)
    log = {
        "gap_minutes": gap_minutes,
        "min_timestamp": str(MIN_TIMESTAMP),
        "min_length": MIN_LENGTH,
        "sample_users": sample_users,
        "events_total": total,
        "random_seed": random_seed,
        "funnel": prep.funnel + funnel,
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
    ap.add_argument("--sample-users", type=float, default=1.0, help="Tỉ lệ user lấy mẫu ngay khi đọc nếu thiếu RAM (ghi vào log)")
    args = ap.parse_args(argv)
    log = run(args.onion, args.out, args.gap_minutes, args.sample_users)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return log


if __name__ == "__main__":
    main()
