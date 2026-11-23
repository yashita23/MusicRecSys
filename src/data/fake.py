"""Dữ liệu giả theo docs/data-contract.md, để B và C làm việc trước khi có dữ liệu thật.

    python -m src.data.fake --out data/fake
"""
from __future__ import annotations

import argparse
import string
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES, SEED_LENGTH

FAKE_MODEL = "fake"
_ALPHABET = np.array(list(string.ascii_letters + string.digits))


@dataclass
class FakeData:
    tracks: pd.DataFrame
    sessions: pd.DataFrame
    eval_targets: pd.DataFrame
    candidates: dict[str, pd.DataFrame]  # split → candidates của mô hình "fake", lần chạy 0
    embeddings: dict[str, tuple[np.ndarray, pd.DataFrame]]  # "fake" → (vectors, index)


def _camelot(key: int, mode: int) -> str:
    """Pitch class + mode → mã Camelot (C trưởng = 8B, A thứ = 8A)."""
    if mode == 1:
        return f"{(7 * key + 7) % 12 + 1}B"
    return f"{(7 * key + 4) % 12 + 1}A"


def _track_ids(rng: np.random.Generator, n: int) -> list[str]:
    ids: set[str] = set()
    while len(ids) < n:
        ids.add("".join(rng.choice(_ALPHABET, 22)))
    return sorted(ids)


def make_fake_dataset(n_sessions: int = 200, n_tracks: int = 500, random_seed: int = 0, dim: int = 16) -> FakeData:
    if n_tracks < N_CANDIDATES + SEED_LENGTH:
        raise ValueError(f"n_tracks phải ≥ {N_CANDIDATES + SEED_LENGTH} để đủ {N_CANDIDATES} ứng viên")
    rng = np.random.default_rng(random_seed)
    ids = _track_ids(rng, n_tracks)
    popularity = 1.0 / np.arange(1, n_tracks + 1) ** 0.8
    popularity = popularity[rng.permutation(n_tracks)]
    popularity /= popularity.sum()

    order = rng.permutation(n_sessions)
    n_eval = max(1, round(0.1 * n_sessions))
    split_of = {int(s): "val" for s in order[:n_eval]}
    split_of.update({int(s): "test" for s in order[n_eval : 2 * n_eval]})

    rows = []
    for sid in range(n_sessions):
        length = int(rng.integers(10, 41))
        chosen = rng.choice(n_tracks, size=length, replace=False, p=popularity)  # không lặp bài → bài đích ∉ seed
        split = split_of.get(sid, "train")
        rows += [{"session_id": sid, "split": split, "pos": p, "track_id": ids[t]} for p, t in enumerate(chosen)]
    sessions = pd.DataFrame(rows)

    key = rng.integers(0, 12, n_tracks)
    mode = rng.integers(0, 2, n_tracks)
    tempo = rng.uniform(70, 180, n_tracks).round(3)
    train_counts = sessions.loc[sessions["split"] == "train", "track_id"].value_counts()
    tracks = pd.DataFrame(
        {
            "track_id": ids,
            "track_name": [f"Fake song {i}" for i in range(n_tracks)],
            "artist_name": [f"Fake artist {i % 50}" for i in range(n_tracks)],
            "tempo": tempo,
            "bpm": tempo,
            "key": key,
            "mode": mode,
            "camelot": [_camelot(int(k), int(m)) for k, m in zip(key, mode)],
            "energy": rng.uniform(0, 1, n_tracks).round(3),
        }
    )
    tracks["n_occurrences"] = tracks["track_id"].map(train_counts).fillna(0).astype("int64")

    first = sessions[sessions["pos"] <= SEED_LENGTH].sort_values(["session_id", "pos"]).groupby("session_id")["track_id"].agg(list)
    eval_sids = sorted(split_of)
    eval_targets = pd.DataFrame(
        {
            "session_id": eval_sids,
            "split": [split_of[s] for s in eval_sids],
            "seed_track_ids": [first[s][:SEED_LENGTH] for s in eval_sids],
            "target_track_id": [first[s][SEED_LENGTH] for s in eval_sids],
        }
    )

    index_of = {t: i for i, t in enumerate(ids)}
    candidates: dict[str, pd.DataFrame] = {}
    for split in ("val", "test"):
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
    vectors = rng.normal(size=(len(learned), dim)).astype(np.float32)
    index = pd.DataFrame({"row": np.arange(len(learned)), "track_id": learned})
    return FakeData(tracks, sessions, eval_targets, candidates, {FAKE_MODEL: (vectors, index)})


def write_fake(data: FakeData, out_dir: str | Path, run: int = 0) -> list[Path]:
    """Ghi dữ liệu giả theo bố cục thư mục trong docs/data-contract.md."""
    out = Path(out_dir)
    (out / "candidates").mkdir(parents=True, exist_ok=True)
    (out / "item_embeddings").mkdir(parents=True, exist_ok=True)
    written = []
    for name, df in [("tracks", data.tracks), ("sessions", data.sessions), ("eval_targets", data.eval_targets)]:
        path = out / f"{name}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
    for split, df in data.candidates.items():
        path = out / "candidates" / f"{FAKE_MODEL}_{split}_run{run}.parquet"
        df.to_parquet(path, index=False)
        written.append(path)
    for model, (vectors, index) in data.embeddings.items():
        np.save(out / "item_embeddings" / f"{model}.npy", vectors)
        index.to_parquet(out / "item_embeddings" / f"{model}_tracks.parquet", index=False)
        written += [out / "item_embeddings" / f"{model}.npy", out / "item_embeddings" / f"{model}_tracks.parquet"]
    return written


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Sinh dữ liệu giả theo docs/data-contract.md")
    ap.add_argument("--out", default="data/fake")
    ap.add_argument("--n-sessions", type=int, default=200)
    ap.add_argument("--n-tracks", type=int, default=500)
    ap.add_argument("--random-seed", type=int, default=0)
    a = ap.parse_args(argv)
    for path in write_fake(make_fake_dataset(a.n_sessions, a.n_tracks, a.random_seed), a.out):
        print(path)


if __name__ == "__main__":
    main()
