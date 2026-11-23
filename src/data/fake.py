"""Dữ liệu giả theo docs/data-contract.md, để B và C làm việc trước khi có dữ liệu thật.

    python -m src.data.fake --out data/fake

Dữ liệu giả cố ý có đủ các trường hợp khó của dữ liệu thật:
phiên lặp bài, bài đích không có trong train (luôn trượt), và phiên val/test
có bài đích nằm trong seed (bị bỏ khỏi eval_targets).
"""
from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import (
    EVAL_SPLITS,
    MIN_SESSION_LENGTH,
    N_CANDIDATES,
    SEED_LENGTH,
    count_train_occurrences,
    session_track_lists,
)
from src.features.camelot import camelot_code

FAKE_MODEL = "fake"
EMBEDDING_DIM = 16
_ALPHABET = np.array(list(string.ascii_letters + string.digits))
_MAX_SESSION_LENGTH = 40
_RARE_SHARE = 0.05        # phần danh mục chỉ dùng làm bài đích, không bao giờ có trong train
_P_UNSEEN_TARGET = 0.3    # xác suất bài đích của phiên val/test là bài không có trong train
_P_TARGET_IN_SEED = 0.1   # xác suất bài đích của phiên val/test lặp lại một bài trong seed
_P_REPEAT = 0.3           # xác suất một phiên có lặp bài sau bài đích


@dataclass
class FakeData:
    tracks: pd.DataFrame
    sessions: pd.DataFrame
    eval_targets: pd.DataFrame
    candidates: dict[str, pd.DataFrame]  # split → candidates của mô hình "fake", lần chạy 0
    embeddings: dict[str, tuple[np.ndarray, pd.DataFrame]]  # "fake" → (vectors, index)


def _track_ids(rng: np.random.Generator, n: int) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add("".join(rng.choice(_ALPHABET, 22)))
    return sorted(ids)


def _session(rng: np.random.Generator, n_tracks: int, popularity: np.ndarray, rare: np.ndarray, is_eval: bool) -> list[int]:
    """Một phiên dưới dạng chỉ số bài. Bài hiếm chỉ xuất hiện ở vị trí bài đích của phiên val/test."""
    length = int(rng.integers(MIN_SESSION_LENGTH, _MAX_SESSION_LENGTH + 1))
    chosen = list(rng.choice(n_tracks, size=length, replace=False, p=popularity))
    if is_eval:
        u = rng.random()
        if u < _P_UNSEEN_TARGET:
            chosen[SEED_LENGTH] = int(rng.choice(rare))
        elif u < _P_UNSEEN_TARGET + _P_TARGET_IN_SEED:
            chosen[SEED_LENGTH] = chosen[int(rng.integers(0, SEED_LENGTH))]
    if rng.random() < _P_REPEAT:
        chosen.append(chosen[int(rng.integers(0, len(chosen)))])
    return [int(t) for t in chosen]


def make_fake_dataset(n_sessions: int = 200, n_tracks: int = 500, random_seed: int = 0) -> FakeData:
    n_rare = max(1, int(_RARE_SHARE * n_tracks))
    if n_tracks - n_rare - SEED_LENGTH < max(N_CANDIDATES, _MAX_SESSION_LENGTH):
        raise ValueError(f"n_tracks quá nhỏ: cần đủ {N_CANDIDATES} ứng viên ngoài seed và bài hiếm")
    rng = np.random.default_rng(random_seed)
    ids = _track_ids(rng, n_tracks)
    rare = rng.permutation(n_tracks)[:n_rare]
    popularity = 1.0 / np.arange(1, n_tracks + 1) ** 0.8
    popularity = popularity[rng.permutation(n_tracks)]
    popularity[rare] = 0
    popularity /= popularity.sum()

    order = rng.permutation(n_sessions)
    n_eval = max(1, round(0.1 * n_sessions))
    split_of = {int(s): "val" for s in order[:n_eval]}
    split_of.update({int(s): "test" for s in order[n_eval : 2 * n_eval]})

    rows = []
    for sid in range(n_sessions):
        split = split_of.get(sid, "train")
        chosen = _session(rng, n_tracks, popularity, rare, is_eval=split != "train")
        rows += [{"session_id": sid, "split": split, "pos": p, "track_id": ids[t]} for p, t in enumerate(chosen)]
    sessions = pd.DataFrame(rows)

    key = rng.integers(0, 12, n_tracks)
    mode = rng.integers(0, 2, n_tracks)
    tempo = rng.uniform(70, 180, n_tracks).round(3)
    train_counts = count_train_occurrences(sessions)
    tracks = pd.DataFrame(
        {
            "track_id": ids,
            "track_name": [f"Fake song {i}" for i in range(n_tracks)],
            "artist_name": [f"Fake artist {i % 50}" for i in range(n_tracks)],
            "tempo": tempo,
            "bpm": tempo,
            "key": key,
            "mode": mode,
            "camelot": [camelot_code(int(k), int(m)) for k, m in zip(key, mode)],
            "energy": rng.uniform(0, 1, n_tracks).round(3),
        }
    )
    tracks["n_occurrences"] = tracks["track_id"].map(train_counts).fillna(0).astype("int64")

    track_lists = session_track_lists(sessions)
    eval_sids = [s for s in sorted(split_of) if track_lists[s][SEED_LENGTH] not in track_lists[s][:SEED_LENGTH]]
    eval_targets = pd.DataFrame(
        {
            "session_id": eval_sids,
            "split": [split_of[s] for s in eval_sids],
            "seed_track_ids": [track_lists[s][:SEED_LENGTH] for s in eval_sids],
            "target_track_id": [track_lists[s][SEED_LENGTH] for s in eval_sids],
        }
    )

    index_of = {t: i for i, t in enumerate(ids)}
    candidates: dict[str, pd.DataFrame] = {}
    for split in EVAL_SPLITS:
        parts = []
        for sid, seed in eval_targets.loc[eval_targets["split"] == split, ["session_id", "seed_track_ids"]].itertuples(index=False):
            p = popularity.copy()
            p[[index_of[t] for t in seed]] = 0
            picked = rng.choice(n_tracks, size=N_CANDIDATES, replace=False, p=p / p.sum())
            scores = np.sort(rng.uniform(0, 1, N_CANDIDATES))[::-1]
            parts.append(
                pd.DataFrame({"session_id": sid, "rank": np.arange(1, N_CANDIDATES + 1), "track_id": [ids[t] for t in picked], "taste_score": scores})
            )
        candidates[split] = pd.concat(parts, ignore_index=True)

    learned = sorted(train_counts.index)
    vectors = rng.normal(size=(len(learned), EMBEDDING_DIM)).astype(np.float32)
    index = pd.DataFrame({"row": np.arange(len(learned)), "track_id": learned})
    return FakeData(tracks, sessions, eval_targets, candidates, {FAKE_MODEL: (vectors, index)})


def write_fake(data: FakeData, out_dir: str | Path) -> list[Path]:
    """Ghi dữ liệu giả theo bố cục thư mục trong docs/data-contract.md (lần chạy 0)."""
    out = Path(out_dir)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    (out / "item_embeddings").mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in [("tracks", data.tracks), ("sessions", data.sessions), ("eval_targets", data.eval_targets)]:
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
    for split, df in data.candidates.items():
        path = out / "candidates" / f"{FAKE_MODEL}_{split}_run0.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
    for model, (vectors, index) in data.embeddings.items():
        npy, idx = out / "item_embeddings" / f"{model}.npy", out / "item_embeddings" / f"{model}_tracks.parquet"
        np.save(npy, vectors)
        index.to_parquet(idx, index=False)
        written += [npy, idx]
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sinh dữ liệu giả theo docs/data-contract.md")
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--n-tracks", type=int, default=500)
    ap.add_argument("--random-seed", type=int, default=0)
    args = ap.parse_args(argv)
    for path in write_fake(make_fake_dataset(args.n_sessions, args.n_tracks, args.random_seed), args.out):
        print(path)


if __name__ == "__main__":
    main()
