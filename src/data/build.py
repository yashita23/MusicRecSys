"""Lấy mẫu, ghép features và lọc playlist MPD thành tập phiên sạch (ticket #5)."""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import pandas as pd

from src.data.coverage import list_slices, load_features, load_mpd, sample_slices, usable_mask
from src.data.schema import MIN_SESSION_LENGTH

MIN_TRACK_OCCURRENCES = 5
N_SLICES = 100        # mỗi slice 1.000 playlist → ≈ 100k playlist
RANDOM_SEED = 42
RAW_TRACK_COLUMNS = ["track_id", "track_name", "artist_name", "tempo", "key", "mode", "energy"]


@dataclass(frozen=True)
class BuildConfig:
    n_slices: int | None = N_SLICES          # None = mọi slice
    random_seed: int = RANDOM_SEED
    min_track_occurrences: int = MIN_TRACK_OCCURRENCES
    min_session_length: int = MIN_SESSION_LENGTH


@dataclass(frozen=True)
class FilterStep:
    """Số phiên, số bài và số lượt xuất hiện còn lại sau một bước."""

    name: str
    n_sessions: int
    n_tracks: int
    n_occurrences: int


def _count_after(name: str, occ: pd.DataFrame) -> FilterStep:
    return FilterStep(name, int(occ["session_id"].nunique()), int(occ["track_id"].nunique()), len(occ))


def _keep_frequent(occ: pd.DataFrame, column: str, minimum: int) -> pd.DataFrame:
    """Giữ các dòng mà giá trị ở `column` xuất hiện ≥ minimum lần."""
    return occ[occ[column].map(occ[column].value_counts()) >= minimum]


def filter_sessions(
    occ: pd.DataFrame,
    min_track_occurrences: int = MIN_TRACK_OCCURRENCES,
    min_session_length: int = MIN_SESSION_LENGTH,
) -> tuple[pd.DataFrame, list[FilterStep]]:
    """Bỏ bài có ít hơn min_track_occurrences lượt, bỏ phiên ít hơn min_session_length bài, lặp đến khi ổn định."""
    log = [_count_after("đầu vào", occ)]
    round_no = 0
    while True:
        round_no += 1
        n_before = len(occ)
        occ = _keep_frequent(occ, "track_id", min_track_occurrences)
        log.append(_count_after(f"vòng {round_no}: bỏ bài < {min_track_occurrences} lượt", occ))
        occ = _keep_frequent(occ, "session_id", min_session_length)
        log.append(_count_after(f"vòng {round_no}: bỏ phiên < {min_session_length} bài", occ))
        if len(occ) == n_before:
            break
    occ = occ.sort_values(["session_id", "pos"]).reset_index(drop=True)
    occ["pos"] = occ.groupby("session_id").cumcount()
    return occ[["session_id", "pos", "track_id"]], log


@dataclass
class BuildResult:
    sessions: pd.DataFrame  # sessions_raw: session_id, pos, track_id (chưa chia)
    tracks: pd.DataFrame    # tracks_raw: RAW_TRACK_COLUMNS, đúng các bài còn trong sessions
    log: list[FilterStep]
    config: BuildConfig = field(default_factory=BuildConfig)
    features_path: str = ""


def build(mpd_dir: str | Path, features_path: str | Path, config: BuildConfig = BuildConfig()) -> BuildResult:
    """Lấy mẫu slice MPD → ghép features → bỏ bài không dùng được → lọc lặp đến khi ổn định."""
    slices = sample_slices(list_slices(mpd_dir), config.n_slices, config.random_seed)
    occ, names = load_mpd(slices)
    occ = occ.rename(columns={"pid": "session_id"})
    if occ.duplicated(["session_id", "pos"]).any():
        raise ValueError("session_id (pid) bị trùng giữa các slice")
    log = [_count_after(f"lấy mẫu {len(slices)} slice", occ)]

    feats = load_features(features_path)
    feats = feats[usable_mask(feats)]
    occ = occ[occ["track_id"].isin(set(feats["track_id"]))]  # bỏ cả file local (track_id = None)
    log.append(_count_after("bỏ bài không có features dùng được", occ))

    sessions, filter_log = filter_sessions(occ, config.min_track_occurrences, config.min_session_length)
    log += filter_log[1:]  # bước "đầu vào" trùng với bước ngay trên

    kept = sessions["track_id"].unique()
    tracks = (
        feats[feats["track_id"].isin(kept)]
        .merge(names, on="track_id", how="left")
        .fillna({"track_name": "", "artist_name": ""})
        .sort_values("track_id")
        .reset_index(drop=True)[RAW_TRACK_COLUMNS]
        .astype({"key": "int64", "mode": "int64"})  # SQLite có thể trả float nếu cột từng có NULL
    )
    return BuildResult(sessions, tracks, log, config, str(features_path))


def write_build(result: BuildResult, out_dir: str | Path) -> list[Path]:
    """Ghi sessions_raw.parquet, tracks_raw.parquet và build_log.json (file trung gian của A)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sessions_path, tracks_path, log_path = out / "sessions_raw.parquet", out / "tracks_raw.parquet", out / "build_log.json"
    result.sessions.to_parquet(sessions_path, index=False)
    result.tracks.to_parquet(tracks_path, index=False)
    log = {
        "config": {**asdict(result.config), "features": result.features_path},
        "steps": [asdict(s) for s in result.log],
    }
    log_path.write_text(json.dumps(log, indent=2, ensure_ascii=False), encoding="utf-8")
    return [sessions_path, tracks_path, log_path]


def format_log(log: list[FilterStep]) -> str:
    """Bảng log; cột cuối là tỉ lệ lượt xuất hiện còn lại so với bước đầu (lấy mẫu)."""
    base = log[0].n_occurrences or 1
    rows = [f"{'bước':<40} {'phiên':>10} {'bài':>10} {'lượt':>12} {'% lượt':>8}"]
    rows += [
        f"{s.name:<40} {s.n_sessions:>10,} {s.n_tracks:>10,} {s.n_occurrences:>12,} {s.n_occurrences / base:>8.1%}"
        for s in log
    ]
    return "\n".join(rows)


def main(argv: list[str] | None = None) -> BuildResult:
    ap = argparse.ArgumentParser(description="Lấy mẫu, ghép và lọc playlist MPD (ticket #5)")
    ap.add_argument("--mpd-dir", required=True)
    ap.add_argument("--features", required=True, help="File .sqlite (hoặc .csv) của bộ features")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-slices", type=int, default=N_SLICES)
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--min-track-occurrences", type=int, default=MIN_TRACK_OCCURRENCES)
    ap.add_argument("--min-session-length", type=int, default=MIN_SESSION_LENGTH)
    args = ap.parse_args(argv)
    config = BuildConfig(args.n_slices, args.random_seed, args.min_track_occurrences, args.min_session_length)
    result = build(args.mpd_dir, args.features, config)
    write_build(result, args.out)
    print(format_log(result.log))
    return result


if __name__ == "__main__":
    main()
