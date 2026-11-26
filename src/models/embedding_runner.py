"""Khung chung cho mô hình gu dựa trên embedding bài: ALS (#14), item2vec (#15).

Mỗi mô hình chỉ cần học vector bài (`_learn`); phần còn lại dùng chung:
- ứng viên tính bằng vector phiên (src/models/session.py), dự phòng Popularity;
- chọn tham số trên val theo NDCG@10: mỗi bộ tham số học (khác decay) chỉ học một lần, decay chỉ đổi cách tính vector phiên;
- CLI: lần chạy 0 chọn tham số, ghi candidates, item_embeddings và {model}_params.json;
  lần chạy 1, 2 (#26) bắt buộc dùng lại tham số qua --params, random_seed = số lần chạy, chỉ ghi candidates.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.data.schema import EVAL_SPLITS, N_CANDIDATES, validate_dataset
from src.eval.run import VAL, evaluate
from src.models.base import export_candidates, save_candidates, save_embeddings
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste

DECAY = "decay"  # tham số của vector phiên, có ở mọi lớp Params; không cần học lại khi đổi


class EmbeddingTasteModel:
    """Mô hình gu: học vector bài từ phiên train rồi gợi ý bằng vector phiên.

    Lớp con đặt `name` và cài `_learn(train) -> (vectors, track_ids)`. track_ids phải đã sắp xếp,
    vì khi hòa điểm SessionVectorTaste xếp theo thứ tự này (để hòa thì theo track_id).
    `params` là một dataclass đóng băng có trường `decay`.
    """

    name: str

    def __init__(self, params: Any, random_seed: int = 0):
        self._init_params = params
        self.random_seed = random_seed

    def _learn(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def fit(self, sessions: pd.DataFrame) -> EmbeddingTasteModel:
        train = sessions[sessions["split"] == "train"]
        if train.empty:
            raise ValueError(f"không có phiên train nào để học {self.name}")
        vectors, track_ids = self._learn(train)
        self.taste = SessionVectorTaste(self.name, vectors, track_ids, self._init_params.decay, Popularity().fit(sessions))
        return self

    @property
    def params(self) -> Any:
        """decay lấy từ vector phiên đang dùng (with_decay chỉ đổi ở đó)."""
        return dataclasses.replace(self._init_params, decay=self.taste.decay)

    def with_decay(self, decay: float) -> EmbeddingTasteModel:
        """Cùng vector bài đã học, decay khác (chọn decay trên val không cần học lại)."""
        clone = copy.copy(self)
        clone.taste = self.taste.with_decay(decay)
        return clone

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        return self.taste.candidates(session_track_ids, k=k)


@dataclass(frozen=True)
class EmbeddingSpec:
    """Những gì khác nhau giữa các mô hình embedding khi chọn tham số và chạy CLI."""

    name: str
    params_cls: type
    build: Callable[[Any, int], EmbeddingTasteModel]  # (params, random_seed) → mô hình chưa học
    grid: dict[str, tuple]  # tên tham số → các giá trị thử; phải có "decay"

    @property
    def params_file(self) -> str:
        return f"{self.name}_params.json"


def tune(spec: EmbeddingSpec, sessions: pd.DataFrame, eval_targets: pd.DataFrame, random_seed: int = 0) -> tuple[EmbeddingTasteModel, pd.DataFrame, list[dict]]:
    """Chọn tham số có NDCG@10 cao nhất trên val; hòa thì lấy cấu hình đứng trước trong lưới.

    Trả về mô hình đã chọn, candidates val của nó (để khỏi tính lại) và cả lưới.
    """
    learned = [key for key in spec.grid if key != DECAY]
    rows: list[dict] = []
    best: tuple[EmbeddingTasteModel, pd.DataFrame] | None = None
    best_ndcg = -1.0
    for values in product(*(spec.grid[key] for key in learned)):
        fitted = spec.build(spec.params_cls(**dict(zip(learned, values))), random_seed).fit(sessions)
        for decay in spec.grid[DECAY]:
            model = fitted.with_decay(decay)
            val_cands = export_candidates(model, eval_targets, VAL)
            acc = evaluate(val_cands, eval_targets)
            rows.append({**asdict(model.params), "hit_rate": acc.hit_rate, "ndcg": acc.ndcg, "mrr": acc.mrr})
            if acc.ndcg > best_ndcg:
                best, best_ndcg = (model, val_cands), acc.ndcg
    assert best is not None
    return best[0], best[1], rows


def run(spec: EmbeddingSpec, data_dir: str | Path, out_dir: str | Path, run: int = 0, params_path: str | Path | None = None) -> dict:
    if params_path is None and run != 0:
        raise ValueError(f"lần chạy {run} phải dùng lại tham số của lần chạy 0: thêm --params <out>/{spec.params_file}")
    data = Path(data_dir)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")

    grid = None
    if params_path is None:
        model, val_cands, grid = tune(spec, sessions, eval_targets, random_seed=run)
        cands = {VAL: val_cands}
    else:
        params = spec.params_cls(**json.loads(Path(params_path).read_text(encoding="utf-8"))["params"])
        model, cands = spec.build(params, run).fit(sessions), {}
    for split in EVAL_SPLITS:
        if split not in cands:
            cands[split] = export_candidates(model, eval_targets, split)

    embeddings = {spec.name: model.taste.embeddings()} if run == 0 else {}  # tên file embedding không có số lần chạy
    validate_dataset(tracks, sessions, eval_targets, candidates=cands, embeddings=embeddings)
    save_candidates(cands, out_dir, spec.name, run)
    for vectors, index in embeddings.values():
        save_embeddings(vectors, index, out_dir, spec.name)

    params_dict = asdict(model.params)
    report: dict = {"params": params_dict, "run": run, "val": asdict(evaluate(cands[VAL], eval_targets))}
    if grid is not None:
        report["grid"] = grid
        chosen = {"params": params_dict, "grid": grid}
        Path(out_dir, spec.params_file).write_text(json.dumps(chosen, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(spec: EmbeddingSpec, argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=f"{spec.name}: chọn tham số trên val, xuất candidates và item_embeddings")
    ap.add_argument("--data", required=True, help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--out", required=True, help="Thư mục ra")
    ap.add_argument("--run", type=int, default=0, help="Số lần chạy, cũng là random_seed (0, 1, 2)")
    ap.add_argument("--params", help=f"{spec.params_file} của lần chạy 0; bắt buộc khi --run khác 0")
    args = ap.parse_args(argv)
    report = run(spec, args.data, args.out, args.run, args.params)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report
