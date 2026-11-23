"""Chia train/val/test theo phiên và tách seed → bài đích (ticket #7, ADR 0003)."""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import EVAL_SPLITS, SEED_LENGTH, SPLITS, session_track_lists, validate_dataset
from src.features.audio import make_tracks

RANDOM_SEED = 42
VAL_SHARE = 0.1
TEST_SHARE = 0.1


def split_sessions(
    sessions_raw: pd.DataFrame,
    random_seed: int = RANDOM_SEED,
    val_share: float = VAL_SHARE,
    test_share: float = TEST_SHARE,
) -> pd.DataFrame:
    """Gán mỗi phiên vào đúng một split (train/val/test) theo hoán vị ngẫu nhiên của session_id."""
    ids = np.sort(sessions_raw["session_id"].unique())
    order = np.random.default_rng(random_seed).permutation(ids)
    n_val, n_test = round(val_share * len(ids)), round(test_share * len(ids))
    split_of = {int(s): "val" for s in order[:n_val]}
    split_of.update({int(s): "test" for s in order[n_val : n_val + n_test]})
    out = sessions_raw.copy()
    out["split"] = out["session_id"].map(split_of).fillna("train")
    return out[["session_id", "split", "pos", "track_id"]]


def make_eval_targets(sessions: pd.DataFrame) -> pd.DataFrame:
    """Phiên val/test → seed = pos 0–4, bài đích = pos 5. Bỏ phiên có bài đích nằm trong seed."""
    head = sessions[sessions["split"].isin(EVAL_SPLITS) & (sessions["pos"] <= SEED_LENGTH)]
    lists = session_track_lists(head)
    split_of = head.groupby("session_id")["split"].first()
    keep = [sid for sid, t in lists.items() if t[SEED_LENGTH] not in t[:SEED_LENGTH]]
    return pd.DataFrame(
        {
            "session_id": pd.Series(keep, dtype="int64"),
            "split": [split_of[s] for s in keep],
            "seed_track_ids": [lists[s][:SEED_LENGTH] for s in keep],
            "target_track_id": [lists[s][SEED_LENGTH] for s in keep],
        }
    )


@dataclass
class PrepareResult:
    tracks: pd.DataFrame
    sessions: pd.DataFrame
    eval_targets: pd.DataFrame
    report: dict


def _split_report(sessions: pd.DataFrame, eval_targets: pd.DataFrame, random_seed: int) -> dict:
    n_sessions = sessions.groupby("split")["session_id"].nunique().reindex(SPLITS, fill_value=0)
    n_eval = eval_targets["split"].value_counts().reindex(EVAL_SPLITS, fill_value=0)
    train_tracks = set(sessions.loc[sessions["split"] == "train", "track_id"])
    unseen = (~eval_targets["target_track_id"].isin(train_tracks)).groupby(eval_targets["split"]).mean()
    return {
        "random_seed": random_seed,
        "n_sessions": {s: int(n_sessions[s]) for s in SPLITS},
        "n_eval_targets": {s: int(n_eval[s]) for s in EVAL_SPLITS},
        "n_dropped_target_in_seed": {s: int(n_sessions[s] - n_eval[s]) for s in EVAL_SPLITS},
        "share_target_unseen_in_train": {s: float(unseen.get(s, 0.0)) for s in EVAL_SPLITS},
    }


def prepare(sessions_raw: pd.DataFrame, tracks_raw: pd.DataFrame, random_seed: int = RANDOM_SEED) -> PrepareResult:
    """sessions_raw + tracks_raw (#5) → tracks, sessions, eval_targets đúng data-contract, kèm báo cáo chia tập."""
    sessions = split_sessions(sessions_raw, random_seed)
    eval_targets = make_eval_targets(sessions)
    tracks = make_tracks(tracks_raw, sessions)
    validate_dataset(tracks, sessions, eval_targets)
    return PrepareResult(tracks, sessions, eval_targets, _split_report(sessions, eval_targets, random_seed))


def write_prepared(result: PrepareResult, out_dir: str | Path) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, df in [("tracks", result.tracks), ("sessions", result.sessions), ("eval_targets", result.eval_targets)]:
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        paths.append(path)
    report_path = out / "split_report.json"
    report_path.write_text(json.dumps(result.report, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths + [report_path]


def main(argv: list[str] | None = None) -> PrepareResult:
    ap = argparse.ArgumentParser(description="Chia train/val/test, tách seed → bài đích, ghi file theo data-contract (#6, #7)")
    ap.add_argument("--raw-dir", required=True, help="Thư mục có sessions_raw.parquet và tracks_raw.parquet (#5)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args(argv)
    raw = Path(args.raw_dir)
    result = prepare(pd.read_parquet(raw / "sessions_raw.parquet"), pd.read_parquet(raw / "tracks_raw.parquet"), args.random_seed)
    write_prepared(result, args.out)
    print(json.dumps(result.report, indent=2, ensure_ascii=False))
    return result


if __name__ == "__main__":
    main()
