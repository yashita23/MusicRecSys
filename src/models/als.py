"""Mô hình gu ALS (thư viện implicit) trên ma trận playlist × bài (ticket #14, spec §6).

ALS chỉ dùng để học vector bài. Ứng viên tính bằng vector phiên của src/models/session.py, giống item2vec (#15),
nên app tính lại được bằng numpy từ item_embeddings/als.npy.

    python -m src.models.als --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra>
    python -m src.models.als --data ... --out ... --run 1 --params <out>/als_params.json

Lần chạy 0 chọn (factors, regularization, decay) theo NDCG@10 trên val trong lưới GRID, ghi als_params.json,
candidates/als_{val,test}_run0.parquet và item_embeddings/als.npy + als_tracks.parquet.
Lần chạy 1, 2 (#26) dùng lại tham số đã chọn với random_seed = số lần chạy, chỉ ghi candidates.
In ra JSON: tham số, độ chính xác trên val, và lưới (nếu có chọn tham số).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.schema import EVAL_SPLITS, N_CANDIDATES, validate_dataset
from src.eval.run import evaluate
from src.models.base import export_candidates, save_candidates, save_embeddings
from src.models.baselines import VAL
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste

CONFIDENCE = 40.0  # alpha của implicit: độ tin cậy của một ô có mặt = 1 + 40 (Hu, Koren, Volinsky 2008); ô trống = 1
GRID = {"factors": (64, 128), "regularization": (0.01, 0.1), "decay": (0.5, 0.7, 0.9, 1.0)}
PARAMS_FILE = "als_params.json"


class ALS:
    name = "als"

    def __init__(self, factors: int = 64, regularization: float = 0.01, decay: float = 0.9, iterations: int = 15, random_seed: int = 0):
        self.factors = factors
        self.regularization = regularization
        self.decay = decay
        self.iterations = iterations
        self.random_seed = random_seed

    def fit(self, sessions: pd.DataFrame) -> ALS:
        from implicit.cpu.als import AlternatingLeastSquares  # nhập muộn: app và các mô hình khác không cần implicit
        from threadpoolctl import threadpool_limits  # đi kèm implicit

        train = sessions[sessions["split"] == "train"]
        track_ids = np.sort(train["track_id"].unique())
        track = pd.Index(track_ids).get_indexer(train["track_id"])
        playlist = pd.factorize(train["session_id"])[0]
        playlist_tracks = sp.csr_matrix(
            (np.ones(len(track), dtype=np.float32), (playlist, track)), shape=(playlist.max() + 1, len(track_ids))
        )
        playlist_tracks.data[:] = 1.0  # có mặt hay không; bài lặp trong một playlist không tính thêm

        with threadpool_limits(1, "blas"):  # implicit tự chia luồng; để OpenBLAS chia luồng nữa thì chậm hẳn
            model = AlternatingLeastSquares(
                factors=self.factors,
                regularization=self.regularization,
                alpha=CONFIDENCE,
                iterations=self.iterations,
                random_state=self.random_seed,
            )
            model.fit(playlist_tracks, show_progress=False)
        self.taste = SessionVectorTaste(self.name, model.item_factors, track_ids, self.decay, Popularity().fit(sessions))
        return self

    @property
    def params(self) -> dict[str, float]:
        return {"factors": self.factors, "regularization": self.regularization, "decay": self.decay}

    def with_decay(self, decay: float) -> ALS:
        """Cùng vector bài đã học, decay khác (chọn decay trên val không cần học lại)."""
        clone = copy.copy(self)
        clone.decay = decay
        clone.taste = self.taste.with_decay(decay)
        return clone

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        return self.taste.candidates(session_track_ids, k=k)

    def embeddings(self) -> tuple[np.ndarray, pd.DataFrame]:
        """(vectors, index) theo docs/data-contract.md: item_embeddings/als.npy và als_tracks.parquet."""
        index = pd.DataFrame({"row": np.arange(len(self.taste.index)), "track_id": self.taste.index.to_numpy()})
        return self.taste.vectors, index


def tune(sessions: pd.DataFrame, eval_targets: pd.DataFrame, random_seed: int = 0) -> tuple[ALS, list[dict]]:
    """Chọn tham số có NDCG@10 cao nhất trên val; hòa thì lấy cấu hình đứng trước trong GRID.

    Mỗi cặp (factors, regularization) chỉ học ALS một lần; decay chỉ đổi cách tính vector phiên.
    """
    grid: list[dict] = []
    best, best_ndcg = None, -1.0
    for factors, regularization in product(GRID["factors"], GRID["regularization"]):
        fitted = ALS(factors, regularization, random_seed=random_seed).fit(sessions)
        for decay in GRID["decay"]:
            model = fitted.with_decay(decay)
            acc = evaluate(export_candidates(model, eval_targets, VAL), eval_targets)
            grid.append({**model.params, "hit_rate": acc.hit_rate, "ndcg": acc.ndcg, "mrr": acc.mrr})
            if acc.ndcg > best_ndcg:
                best, best_ndcg = model, acc.ndcg
    assert best is not None
    return best, grid


def run_als(data_dir: str | Path, out_dir: str | Path, run: int = 0, params_path: str | Path | None = None) -> dict:
    data = Path(data_dir)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")

    grid = None
    if params_path is None:
        model, grid = tune(sessions, eval_targets, random_seed=run)
    else:
        params = json.loads(Path(params_path).read_text(encoding="utf-8"))["params"]
        model = ALS(**params, random_seed=run).fit(sessions)

    cands = {split: export_candidates(model, eval_targets, split) for split in EVAL_SPLITS}
    embeddings = {model.name: model.embeddings()} if run == 0 else {}  # tên file embedding không có số lần chạy
    validate_dataset(tracks, sessions, eval_targets, candidates=cands, embeddings=embeddings)
    save_candidates(cands, out_dir, model.name, run)
    for vectors, index in embeddings.values():
        save_embeddings(vectors, index, out_dir, model.name)

    report: dict = {"params": model.params, "run": run, "val": asdict(evaluate(cands[VAL], eval_targets))}
    if grid is not None:
        report["grid"] = grid
        chosen = {"params": model.params, "grid": grid}
        Path(out_dir, PARAMS_FILE).write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="ALS: chọn tham số trên val, xuất candidates và item_embeddings (#14)")
    ap.add_argument("--data", required=True, help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--out", required=True, help="Thư mục ra")
    ap.add_argument("--run", type=int, default=0, help="Số lần chạy, cũng là random_seed (0, 1, 2)")
    ap.add_argument("--params", help=f"{PARAMS_FILE} của lần chạy 0; có thì không chọn lại tham số")
    args = ap.parse_args(argv)
    report = run_als(args.data, args.out, args.run, args.params)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
