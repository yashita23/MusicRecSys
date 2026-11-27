"""Nhập Music4All-Onion (last.fm) từ Zenodo thành parquet trên Kaggle, không qua máy cá nhân (ticket #54).

Nguồn: Moscati và cộng sự, "Music4All-Onion: A Large-Scale Multi-faceted Content-Centric Music Recommendation
Dataset", CIKM 2022. Zenodo record 15394646, DOI 10.5281/zenodo.15394646, giấy phép CC BY 4.0.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import urllib.request
from collections.abc import Callable, Sequence
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.data.schema import SchemaError

RECORD = 15394646
EVENTS_FILE = "userid_trackid_timestamp.tsv.bz2"
ESSENTIA_FILE = "id_essentia.tsv.bz2"
README_FILE = "README.md"
# md5 theo trang Zenodo của record (kiểm ngày 27/11/2026)
ZENODO_MD5 = {
    EVENTS_FILE: "dfe82201036765f7463e6f3ce3d0f991",   # 2.211.449.511 byte
    ESSENTIA_FILE: "6222143003c2cfa0b250dd80a92ac929",  # 407.458.247 byte
    README_FILE: "df379ed27dcefe5652b8db4e6783e304",
}
EXPECTED = {"events_rows": 252_984_396, "events_users": 119_140, "essentia_tracks": 109_269}  # theo README của record


def zenodo_url(name: str) -> str:
    return f"https://zenodo.org/api/records/{RECORD}/files/{name}/content"


EVENTS_SCHEMA = pa.schema([("user_id", pa.int32()), ("track_id", pa.string()), ("timestamp", pa.timestamp("s"))])
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"  # ví dụ 2013-01-27 21:42:38; README không ghi múi giờ, giữ nguyên
CHUNK_ROWS = 5_000_000  # đọc 253 triệu dòng theo khối để vừa RAM của Kaggle
HPCP_MEAN = tuple(f"tonal.hpcp.mean_{i}" for i in range(36))  # HPCP trung bình, 36 bin (3 bin mỗi nốt)
ESSENTIA_FIELDS = (
    "id",
    "rhythm.bpm",
    "rhythm.bpm_histogram_first_peak_bpm",
    "rhythm.bpm_histogram_second_peak_bpm",
    "rhythm.onset_rate",
    "rhythm.danceability",
    "tonal.key_edma.strength",
    "tonal.key_krumhansl.strength",
    "tonal.key_temperley.strength",
    "lowlevel.loudness_ebu128.integrated",
    "lowlevel.average_loudness",
)


def essentia_columns(header: Sequence[str]) -> list[str]:
    """Cột giữ lại từ id_essentia.tsv (1.035 cột): ESSENTIA_FIELDS rồi 36 bin HPCP trung bình theo thứ tự bin."""
    wanted = [*ESSENTIA_FIELDS, *HPCP_MEAN]
    missing = [c for c in wanted if c not in set(header)]
    if missing:
        raise SchemaError("id_essentia", [f"thiếu cột: {', '.join(missing)}"])
    return wanted


def convert_events(src: str | Path, out: str | Path, chunksize: int = CHUNK_ROWS) -> dict:
    """userid_trackid_timestamp.tsv(.bz2) → parquet, đọc và ghi theo khối, giữ mọi dòng theo thứ tự gốc (chưa lọc).

    Trả về số dòng, số user, số bài và khoảng thời gian, để đối chiếu với README của Zenodo.
    """
    users: set[int] = set()
    tracks: set[str] = set()
    rows, t_min, t_max = 0, None, None
    reader = pd.read_csv(src, sep="\t", dtype={"user_id": "int32", "track_id": "string"}, chunksize=chunksize)
    with pq.ParquetWriter(out, EVENTS_SCHEMA) as writer:
        for chunk in reader:
            chunk["timestamp"] = pd.to_datetime(chunk["timestamp"].str.strip(), format=TIMESTAMP_FORMAT)
            writer.write_table(pa.Table.from_pandas(chunk, schema=EVENTS_SCHEMA, preserve_index=False))
            rows += len(chunk)
            users.update(chunk["user_id"].unique().tolist())
            tracks.update(chunk["track_id"].unique().tolist())
            lo, hi = chunk["timestamp"].min(), chunk["timestamp"].max()
            t_min = lo if t_min is None else min(t_min, lo)
            t_max = hi if t_max is None else max(t_max, hi)
    return {
        "rows": rows,
        "users": len(users),
        "tracks": len(tracks),
        "timestamp_min": None if t_min is None else str(t_min),
        "timestamp_max": None if t_max is None else str(t_max),
    }


def convert_essentia(src: str | Path, out: str | Path) -> dict:
    """id_essentia.tsv(.bz2) → parquet chỉ gồm essentia_columns (đặc trưng float32, id string)."""
    header = pd.read_csv(src, sep="\t", nrows=0).columns.tolist()
    columns = essentia_columns(header)
    table = pd.read_csv(src, sep="\t", usecols=columns, dtype={c: ("string" if c == "id" else "float32") for c in columns})
    table[columns].to_parquet(out, index=False)
    return {"rows": len(table), "tracks": int(table["id"].nunique()), "columns": len(columns)}


def md5sum(path: str | Path, block: int = 1 << 24) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as f:
        while chunk := f.read(block):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, path: Path) -> None:
    """Tải theo luồng vào path (qua file tạm, nên lần tải dở không để lại file hỏng)."""
    partial = path.with_name(path.name + ".part")
    with urllib.request.urlopen(url) as response, open(partial, "wb") as f:
        shutil.copyfileobj(response, f, length=1 << 24)
    partial.replace(path)


def run_import(
    work_dir: str | Path,
    out_dir: str | Path,
    md5: dict[str, str] = ZENODO_MD5,
    url_for: Callable[[str], str] = zenodo_url,
) -> dict:
    """Tải ba file vào work_dir (trên Kaggle: /tmp, không phải /kaggle/working), kiểm md5 cả ba,
    rồi mới ghi vào out_dir: listening_events.parquet, essentia.parquet, README.md, import_log.json.

    File đã có trong work_dir và đúng md5 thì không tải lại. md5 sai → ValueError, không ghi gì vào out_dir.
    """
    work, out = Path(work_dir), Path(out_dir)
    work.mkdir(parents=True, exist_ok=True)
    for name, expected in md5.items():
        path = work / name
        if not path.exists() or md5sum(path) != expected:
            download(url_for(name), path)
        actual = md5sum(path)
        if actual != expected:
            raise ValueError(f"md5 của {name} là {actual}, Zenodo ghi {expected}: tải lại hoặc kiểm tra record")

    out.mkdir(parents=True, exist_ok=True)
    log = {
        "source": f"Zenodo record {RECORD}, DOI 10.5281/zenodo.{RECORD}, CC BY 4.0",
        "md5": dict(md5),
        "events": convert_events(work / EVENTS_FILE, out / "listening_events.parquet"),
        "essentia": convert_essentia(work / ESSENTIA_FILE, out / "essentia.parquet"),
    }
    actual = {
        "events_rows": log["events"]["rows"],
        "events_users": log["events"]["users"],
        "essentia_tracks": log["essentia"]["tracks"],
    }
    log["check"] = {key: {"expected": EXPECTED[key], "actual": actual[key], "ok": actual[key] == EXPECTED[key]} for key in EXPECTED}
    shutil.copyfile(work / README_FILE, out / "README.md")
    (out / "import_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    return log


def main(argv: list[str] | None = None) -> dict:
    import argparse

    ap = argparse.ArgumentParser(description="Nhập Music4All-Onion từ Zenodo thành parquet (#54)")
    ap.add_argument("--work", default="/tmp/onion", help="Thư mục tải file gốc (không nằm trong output)")
    ap.add_argument("--out", required=True, help="Thư mục ra, ví dụ /kaggle/working/out/onion")
    args = ap.parse_args(argv)
    log = run_import(args.work, args.out)
    print(json.dumps(log, ensure_ascii=False, indent=2))
    return log


if __name__ == "__main__":
    main()
