"""Đo độ phủ khi ghép playlist MPD với bộ audio features (ticket #1).

Chạy trên Kaggle:
    python -m src.data.coverage --mpd-dir /kaggle/input/spotify-million \
        --features /kaggle/input/2-million-songs-from-mpd-with-audio-features/extracted.sqlite \
        --n-slices 100 --out /kaggle/working/coverage.json
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sqlite3
from pathlib import Path

import pandas as pd

FEATURE_COLS = ["tempo", "key", "mode", "energy"]
_ID_RE = re.compile(r"([A-Za-z0-9]{22})$")


def normalize_track_id(uri: str | None) -> str | None:
    """'spotify:track:XXXX', 'https://open.spotify.com/track/XXXX?si=…' hoặc 'XXXX' → 'XXXX' (22 ký tự)."""
    if uri is None:
        return None
    s = str(uri).strip().split("?")[0].rstrip("/")
    m = _ID_RE.search(s)
    return m.group(1) if m else None


def list_slices(mpd_dir: str | Path) -> list[Path]:
    paths = sorted(Path(mpd_dir).rglob("mpd.slice.*.json"))
    if not paths:
        raise FileNotFoundError(f"Không thấy mpd.slice.*.json trong {mpd_dir}")
    return paths


def sample_slices(paths: list[Path], n_slices: int | None, random_seed: int = 42) -> list[Path]:
    if n_slices is None or n_slices >= len(paths):
        return paths
    return sorted(random.Random(random_seed).sample(paths, n_slices))


def load_mpd_occurrences(slice_paths: list[Path]) -> pd.DataFrame:
    """Một dòng cho mỗi lượt xuất hiện: pid, pos, track_id."""
    pids, poss, tids = [], [], []
    for p in slice_paths:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        for pl in data["playlists"]:
            pid = pl["pid"]
            for t in pl["tracks"]:
                pids.append(pid)
                poss.append(t["pos"])
                tids.append(normalize_track_id(t["track_uri"]))
    return pd.DataFrame({"pid": pids, "pos": poss, "track_id": tids})


def _find_table(con: sqlite3.Connection, table: str | None) -> str:
    names = [r[0] for r in con.execute("SELECT name FROM sqlite_master WHERE type='table'")]
    if table:
        if table not in names:
            raise ValueError(f"Không có bảng {table}; các bảng: {names}")
        return table
    for name in names:
        cols = {r[1] for r in con.execute(f'PRAGMA table_info("{name}")')}
        if {"track_uri", *FEATURE_COLS} <= cols:
            return name
    raise ValueError(f"Không bảng nào có track_uri + {FEATURE_COLS}; các bảng: {names}")


ID_COL_CANDIDATES = ("track_uri", "track_id", "id", "uri")


def load_features(path: str | Path, table: str | None = None) -> pd.DataFrame:
    """track_id + tempo, key, mode, energy, mỗi bài một dòng.

    Nhận file .sqlite/.db (bộ krishsharma0413) hoặc .csv (bộ dự phòng, ví dụ rodolfofigueroa).
    """
    path = Path(path)
    if path.suffix.lower() == ".csv":
        header = pd.read_csv(path, nrows=0).columns
        id_col = next((c for c in ID_COL_CANDIDATES if c in header), None)
        if id_col is None:
            raise ValueError(f"CSV không có cột ID nào trong {ID_COL_CANDIDATES}; các cột: {list(header)}")
        df = pd.read_csv(path, usecols=[id_col, *FEATURE_COLS]).rename(columns={id_col: "track_uri"})
    else:
        con = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        try:
            tbl = _find_table(con, table)
            cols = ", ".join(["track_uri", *FEATURE_COLS])
            df = pd.read_sql_query(f'SELECT {cols} FROM "{tbl}"', con)
        finally:
            con.close()
    df["track_id"] = df.pop("track_uri").map(normalize_track_id)
    return df.dropna(subset=["track_id"]).drop_duplicates("track_id")


def usable_mask(feats: pd.DataFrame) -> pd.Series:
    """Bài dùng được cho điểm mix: đủ 4 đặc trưng, tempo > 0, key được xác định (≠ -1)."""
    return feats[FEATURE_COLS].notna().all(axis=1) & (feats["tempo"] > 0) & (feats["key"] >= 0)


def coverage_report(occ: pd.DataFrame, feats: pd.DataFrame, min_tracks: int = 10, min_occurrences: int = 5) -> dict:
    occ = occ.dropna(subset=["track_id"])
    matched_ids = set(feats["track_id"])
    usable_ids = set(feats.loc[usable_mask(feats), "track_id"])

    track_counts = occ["track_id"].value_counts()
    tracks = track_counts.index
    in_feats = tracks.isin(list(matched_ids))
    in_usable = tracks.isin(list(usable_ids))
    frequent = track_counts.values >= min_occurrences

    occ_matched = occ["track_id"].isin(matched_ids)
    occ_usable = occ["track_id"].isin(usable_ids)

    per_pl_total = occ.groupby("pid").size()
    per_pl_usable = occ[occ_usable].groupby("pid").size().reindex(per_pl_total.index, fill_value=0)

    m = feats[feats["track_id"].isin(set(tracks))]
    missing = {
        "null_any": float(m[FEATURE_COLS].isna().any(axis=1).mean()) if len(m) else None,
        "tempo_le_0": float((m["tempo"] <= 0).mean()) if len(m) else None,
        "key_minus_1": float((m["key"] < 0).mean()) if len(m) else None,
        "energy_out_of_0_1": float(((m["energy"] < 0) | (m["energy"] > 1)).mean()) if len(m) else None,
    }

    def share(x, total):
        return float(x) / float(total) if total else None

    return {
        "n_playlists": int(per_pl_total.size),
        "n_occurrences": int(len(occ)),
        "n_unique_tracks": int(len(tracks)),
        "n_feature_rows": int(len(feats)),
        "track_coverage": share(in_feats.sum(), len(tracks)),
        "occurrence_coverage": share(occ_matched.sum(), len(occ)),
        "track_coverage_usable": share(in_usable.sum(), len(tracks)),
        "occurrence_coverage_usable": share(occ_usable.sum(), len(occ)),
        f"track_coverage_usable_among_ge{min_occurrences}_occ": share((in_usable & frequent).sum(), frequent.sum()),
        f"playlists_ge{min_tracks}_usable_tracks": share((per_pl_usable >= min_tracks).sum(), per_pl_total.size),
        "missing_among_matched": missing,
    }


def _fmt(v) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.2%}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def format_report(r: dict) -> str:
    lines = []
    for k, v in r.items():
        if isinstance(v, dict):
            lines += [f"{k}:"] + [f"  {kk}: {_fmt(vv)}" for kk, vv in v.items()]
        else:
            lines.append(f"{k}: {_fmt(v)}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--mpd-dir", required=True)
    ap.add_argument("--features", required=True, help="Đường dẫn file .sqlite")
    ap.add_argument("--table", default=None)
    ap.add_argument("--n-slices", type=int, default=100, help="Mỗi slice 1.000 playlist")
    ap.add_argument("--random-seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)

    slices = sample_slices(list_slices(a.mpd_dir), a.n_slices, a.random_seed)
    occ = load_mpd_occurrences(slices)
    feats = load_features(a.features, a.table)
    report = coverage_report(occ, feats)
    report["config"] = {"n_slices": len(slices), "random_seed": a.random_seed, "features": str(a.features)}
    print(format_report(report))
    if a.out:
        Path(a.out).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    main()
