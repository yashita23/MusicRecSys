"""Chia train/val/test theo phiên và tách seed → bài đích (ticket #7, ADR 0003).

    python -m src.data.split --raw-dir <thư mục của #5> --out <thư mục ra>
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import EVAL_SPLITS, SEED_LENGTH, SPLITS, session_track_lists, validate_dataset
from src.features.audio import make_tracks

RANDOM_SEED = 42
VAL_SHARE = TEST_SHARE = 0.1
RAW_FILES = ("sessions_raw.parquet", "tracks_raw.parquet", "build_log.json")  # file của #5, giữ lại trong v2


def assign_splits(session_ids, random_seed: int = RANDOM_SEED) -> dict[int, str]:
    """session_id → train/val/test theo hoán vị ngẫu nhiên; 10% val, 10% test, còn lại train."""
    ids = np.sort(np.unique(np.asarray(session_ids)))
    order = np.random.default_rng(random_seed).permutation(ids)
    n_val, n_test = round(VAL_SHARE * len(ids)), round(TEST_SHARE * len(ids))
    split_of = {int(s): "train" for s in ids}
    split_of.update({int(s): "val" for s in order[:n_val]})
    split_of.update({int(s): "test" for s in order[n_val : n_val + n_test]})
    return split_of


def split_sessions(sessions_raw: pd.DataFrame, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Thêm cột split; mọi lượt của một phiên cùng thuộc một split."""
    split_of = assign_splits(sessions_raw["session_id"], random_seed)
    sessions = sessions_raw.assign(split=sessions_raw["session_id"].map(split_of))
    return sessions[["session_id", "split", "pos", "track_id"]]


def make_eval_targets(sessions: pd.DataFrame) -> pd.DataFrame:
    """Phiên val/test → seed = pos 0–4, bài đích = pos 5. Bỏ phiên có bài đích nằm trong seed."""
    first_six = sessions[sessions["split"].isin(EVAL_SPLITS) & (sessions["pos"] <= SEED_LENGTH)]
    tracks_of = session_track_lists(first_six)
    split_of = first_six.groupby("session_id")["split"].first()
    kept = [sid for sid, t in tracks_of.items() if t[SEED_LENGTH] not in t[:SEED_LENGTH]]
    return pd.DataFrame(
        {
            "session_id": pd.Series(kept, dtype="int64"),
            "split": [split_of[s] for s in kept],
            "seed_track_ids": [tracks_of[s][:SEED_LENGTH] for s in kept],
            "target_track_id": [tracks_of[s][SEED_LENGTH] for s in kept],
        }
    )


@dataclass(frozen=True)
class SplitReport:
    random_seed: int
    n_sessions: dict[str, int]                        # theo split
    n_eval_targets: dict[str, int]                    # val, test
    n_dropped_target_in_seed: dict[str, int]          # phiên val/test bị bỏ vì bài đích nằm trong seed
    share_target_unseen_in_train: dict[str, float]    # tỉ lệ bài đích không có trong train (luôn trượt)
    fingerprint: str                                  # sha256 của tracks, sessions, eval_targets theo đúng thứ tự dòng


def fingerprint(*tables: pd.DataFrame) -> str:
    """Dấu vân tay có tính thứ tự: cùng dữ liệu và cùng thứ tự dòng thì ra cùng chuỗi."""
    h = hashlib.sha256()
    for df in tables:
        hashable = df.assign(**{c: df[c].map(lambda v: "|".join(v)) for c in df.columns if c == "seed_track_ids"})
        h.update(pd.util.hash_pandas_object(hashable, index=False).to_numpy().tobytes())
    return h.hexdigest()


def split_report(tracks: pd.DataFrame, sessions: pd.DataFrame, eval_targets: pd.DataFrame, random_seed: int) -> SplitReport:
    n_sessions = sessions.groupby("split")["session_id"].nunique().reindex(SPLITS, fill_value=0)
    n_eval = eval_targets["split"].value_counts().reindex(EVAL_SPLITS, fill_value=0)
    train_tracks = set(sessions.loc[sessions["split"] == "train", "track_id"])
    unseen = (~eval_targets["target_track_id"].isin(train_tracks)).groupby(eval_targets["split"]).mean()
    return SplitReport(
        random_seed=random_seed,
        n_sessions={s: int(n_sessions[s]) for s in SPLITS},
        n_eval_targets={s: int(n_eval[s]) for s in EVAL_SPLITS},
        # mọi phiên có ≥ 10 bài nên phiên val/test vắng mặt trong eval_targets chỉ có thể là phiên bị bỏ vì bài đích ∈ seed
        n_dropped_target_in_seed={s: int(n_sessions[s] - n_eval[s]) for s in EVAL_SPLITS},
        share_target_unseen_in_train={s: float(unseen.get(s, 0.0)) for s in EVAL_SPLITS},
        fingerprint=fingerprint(tracks, sessions, eval_targets),
    )


@dataclass
class PrepareResult:
    tracks: pd.DataFrame
    sessions: pd.DataFrame
    eval_targets: pd.DataFrame
    report: SplitReport


def prepare(sessions_raw: pd.DataFrame, tracks_raw: pd.DataFrame, random_seed: int = RANDOM_SEED) -> PrepareResult:
    """sessions_raw + tracks_raw (#5) → ba bảng chính của data-contract: tracks (#6), sessions và eval_targets (#7)."""
    sessions = split_sessions(sessions_raw, random_seed)
    eval_targets = make_eval_targets(sessions)
    tracks = make_tracks(tracks_raw, sessions)
    validate_dataset(tracks, sessions, eval_targets)
    return PrepareResult(tracks, sessions, eval_targets, split_report(tracks, sessions, eval_targets, random_seed))


def write_processed(
    out_dir: str | Path, tracks: pd.DataFrame, sessions: pd.DataFrame, eval_targets: pd.DataFrame, report: SplitReport
) -> list[Path]:
    """Ghi ba bảng chính và split_report.json theo bố cục mixnext-processed (dùng chung cho dữ liệu thật và giả)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in [("tracks", tracks), ("sessions", sessions), ("eval_targets", eval_targets)]:
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths.append(path)
    report_path = out / "split_report.json"
    report_path.write_text(json.dumps(asdict(report), indent=2, ensure_ascii=False), encoding="utf-8")
    return paths + [report_path]


def write_prepared(result: PrepareResult, out_dir: str | Path, raw_dir: str | Path | None = None) -> list[Path]:
    """Ghi kết quả của prepare; nếu có raw_dir thì chép thêm các file của #5, để thư mục ra đủ làm mixnext-processed v2."""
    paths = write_processed(out_dir, result.tracks, result.sessions, result.eval_targets, result.report)
    if raw_dir is not None:
        for name in RAW_FILES:
            if (Path(raw_dir) / name).exists():
                paths.append(Path(shutil.copy(Path(raw_dir) / name, Path(out_dir) / name)))
    return paths


def main(argv: list[str] | None = None) -> PrepareResult:
    ap = argparse.ArgumentParser(description="Chia train/val/test, tách seed → bài đích, ghi file theo data-contract (#6, #7)")
    ap.add_argument("--raw-dir", required=True, help="Thư mục có sessions_raw.parquet và tracks_raw.parquet (#5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args(argv)
    raw = Path(args.raw_dir)
    result = prepare(pd.read_parquet(raw / "sessions_raw.parquet"), pd.read_parquet(raw / "tracks_raw.parquet"), args.random_seed)
    write_prepared(result, args.out, raw_dir=raw)
    print(json.dumps(asdict(result.report), indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
