"""Mô hình gu ALS (thư viện implicit) trên ma trận playlist × bài (ticket #14, spec §6).

ALS chỉ dùng để học vector bài. Ứng viên tính bằng vector phiên của src/models/session.py, giống item2vec (#15),
nên app tính lại được bằng numpy từ item_embeddings/als.npy.

    python -m src.models.als --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra>
    python -m src.models.als --data ... --out ... --run 1 --params <out>/als_params.json

Lần chạy 0 chọn (factors, regularization, decay) theo NDCG@10 trên val trong lưới GRID, ghi als_params.json,
candidates/als_{val,test}_run0.parquet và item_embeddings/als.npy + als_tracks.parquet.
Lần chạy 1, 2 (#26) bắt buộc dùng lại tham số đã chọn, với random_seed = số lần chạy, và chỉ ghi candidates.
In ra JSON: tham số, độ chính xác trên val, và lưới (nếu có chọn tham số).
"""
from __future__ import annotations

import argparse
import copy
import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.data.schema import EVAL_SPLITS, N_CANDIDATES, validate_dataset
from src.eval.run import VAL, evaluate
from src.models.base import export_candidates, save_candidates, save_embeddings
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste

CONFIDENCE = 40.0  # alpha của implicit: độ tin cậy của một ô có mặt = 1 + 40 (Hu, Koren, Volinsky 2008); ô trống = 1
ITERATIONS = 15
# Lưới lần 1 {64, 128} × {0,01; 0,1}: tốt nhất ở mép (128; 0,1), nên lần 2 mở rộng về phía đó (notebooks/05_als.ipynb)
GRID = {"factors": (128, 256), "regularization": (0.1, 1.0, 10.0), "decay": (0.5, 0.7, 0.9, 1.0)}
PARAMS_FILE = "als_params.json"


@dataclass(frozen=True)
class ALSParams:
    """Tham số chọn trên val: hai tham số của ALS và decay của vector phiên."""

    factors: int = 64
    regularization: float = 0.01
    decay: float = 0.9


class ALS:
    name = "als"

    def __init__(self, params: ALSParams = ALSParams(), iterations: int = ITERATIONS, random_seed: int = 0):
        self._init_params = params
        self.iterations = iterations
        self.random_seed = random_seed

    def fit(self, sessions: pd.DataFrame) -> ALS:
        from implicit.cpu.als import AlternatingLeastSquares  # nhập muộn: app và các mô hình khác không cần implicit
        from threadpoolctl import threadpool_limits  # đi kèm implicit

        train = sessions[sessions["split"] == "train"]
        if train.empty:
            raise ValueError("không có phiên train nào để học ALS")
        track_ids = np.sort(train["track_id"].unique())
        track_col = pd.Index(track_ids).get_indexer(train["track_id"])
        playlist_row = pd.factorize(train["session_id"])[0]
        playlist_tracks = sp.csr_matrix(
            (np.ones(len(track_col), dtype=np.float32), (playlist_row, track_col)), shape=(playlist_row.max() + 1, len(track_ids))
        )
        playlist_tracks.data[:] = 1.0  # có mặt hay không; bài lặp trong một playlist không tính thêm

        p = self._init_params
        with threadpool_limits(1, "blas"):  # implicit tự chia luồng; để OpenBLAS chia luồng nữa thì chậm hẳn
            model = AlternatingLeastSquares(
                factors=p.factors,
                regularization=p.regularization,
                alpha=CONFIDENCE,
                iterations=self.iterations,
                random_state=self.random_seed,
            )
            model.fit(playlist_tracks, show_progress=False)
        self.taste = SessionVectorTaste(self.name, model.item_factors, track_ids, p.decay, Popularity().fit(sessions))
        return self

    @property
    def params(self) -> ALSParams:
        """decay lấy từ vector phiên đang dùng (with_decay chỉ đổi ở đó)."""
        p = self._init_params
        return ALSParams(p.factors, p.regularization, self.taste.decay)

    def with_decay(self, decay: float) -> ALS:
        """Cùng vector bài đã học, decay khác (chọn decay trên val không cần học lại)."""
        clone = copy.copy(self)
        clone.taste = self.taste.with_decay(decay)
        return clone

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        return self.taste.candidates(session_track_ids, k=k)


def tune(sessions: pd.DataFrame, eval_targets: pd.DataFrame, random_seed: int = 0) -> tuple[ALS, pd.DataFrame, list[dict]]:
    """Chọn tham số có NDCG@10 cao nhất trên val; hòa thì lấy cấu hình đứng trước trong GRID.

    Mỗi cặp (factors, regularization) chỉ học ALS một lần; decay chỉ đổi cách tính vector phiên.
    Trả về mô hình đã chọn, candidates val của nó (để khỏi tính lại) và cả lưới.
    """
    grid: list[dict] = []
    best: tuple[ALS, pd.DataFrame] | None = None
    best_ndcg = -1.0
    for factors, regularization in product(GRID["factors"], GRID["regularization"]):
        fitted = ALS(ALSParams(factors, regularization), random_seed=random_seed).fit(sessions)
        for decay in GRID["decay"]:
            model = fitted.with_decay(decay)
            val_cands = export_candidates(model, eval_targets, VAL)
            acc = evaluate(val_cands, eval_targets)
            grid.append({**asdict(model.params), "hit_rate": acc.hit_rate, "ndcg": acc.ndcg, "mrr": acc.mrr})
            if acc.ndcg > best_ndcg:
                best, best_ndcg = (model, val_cands), acc.ndcg
    assert best is not None
    return best[0], best[1], grid


def run_als(data_dir: str | Path, out_dir: str | Path, run: int = 0, params_path: str | Path | None = None) -> dict:
    if params_path is None and run != 0:
        raise ValueError(f"lần chạy {run} phải dùng lại tham số của lần chạy 0: thêm --params <out>/{PARAMS_FILE}")
    data = Path(data_dir)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")

    grid = None
    if params_path is None:
        model, val_cands, grid = tune(sessions, eval_targets, random_seed=run)
        cands = {VAL: val_cands}
    else:
        params = ALSParams(**json.loads(Path(params_path).read_text(encoding="utf-8"))["params"])
        model, cands = ALS(params, random_seed=run).fit(sessions), {}
    for split in EVAL_SPLITS:
        if split not in cands:
            cands[split] = export_candidates(model, eval_targets, split)

    embeddings = {model.name: model.taste.embeddings()} if run == 0 else {}  # tên file embedding không có số lần chạy
    validate_dataset(tracks, sessions, eval_targets, candidates=cands, embeddings=embeddings)
    save_candidates(cands, out_dir, model.name, run)
    for vectors, index in embeddings.values():
        save_embeddings(vectors, index, out_dir, model.name)

    params_dict = asdict(model.params)
    report: dict = {"params": params_dict, "run": run, "val": asdict(evaluate(cands[VAL], eval_targets))}
    if grid is not None:
        report["grid"] = grid
        chosen = {"params": params_dict, "grid": grid}
        Path(out_dir, PARAMS_FILE).write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="ALS: chọn tham số trên val, xuất candidates và item_embeddings (#14)")
    ap.add_argument("--data", required=True, help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--out", required=True, help="Thư mục ra")
    ap.add_argument("--run", type=int, default=0, help="Số lần chạy, cũng là random_seed (0, 1, 2)")
    ap.add_argument("--params", help=f"{PARAMS_FILE} của lần chạy 0; bắt buộc khi --run khác 0")
    args = ap.parse_args(argv)
    report = run_als(args.data, args.out, args.run, args.params)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
