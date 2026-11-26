"""Mô hình gu ALS (thư viện implicit) trên ma trận playlist × bài (ticket #14, spec §6).

ALS chỉ dùng để học vector bài. Ứng viên tính bằng vector phiên của src/models/session.py, giống item2vec (#15),
nên app tính lại được bằng numpy từ item_embeddings/als.npy. Chọn tham số và CLI ở src/models/embedding_runner.py.

    python -m src.models.als --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra>
    python -m src.models.als --data ... --out ... --run 1 --params <out>/als_params.json
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import scipy.sparse as sp

from src.models import embedding_runner
from src.models.embedding_runner import EmbeddingSpec, EmbeddingTasteModel

CONFIDENCE = 40.0  # alpha của implicit: độ tin cậy của một ô có mặt = 1 + 40 (Hu, Koren, Volinsky 2008); ô trống = 1
ITERATIONS = 15
# Lưới lần 1 {64, 128} × {0,01; 0,1}: tốt nhất ở mép (128; 0,1), nên lần 2 mở rộng về phía đó (notebooks/05_als.ipynb)
GRID = {"factors": (128, 256), "regularization": (0.1, 1.0, 10.0), "decay": (0.5, 0.7, 0.9, 1.0)}


@dataclass(frozen=True)
class ALSParams:
    """Tham số chọn trên val: hai tham số của ALS và decay của vector phiên."""

    factors: int = 64
    regularization: float = 0.01
    decay: float = 0.9


class ALS(EmbeddingTasteModel):
    name = "als"

    def __init__(self, params: ALSParams = ALSParams(), random_seed: int = 0, iterations: int = ITERATIONS):
        super().__init__(params, random_seed)
        self.iterations = iterations

    def _learn(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from implicit.cpu.als import AlternatingLeastSquares  # nhập muộn: app và các mô hình khác không cần implicit
        from threadpoolctl import threadpool_limits  # đi kèm implicit

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
        return model.item_factors, track_ids


SPEC = EmbeddingSpec(name=ALS.name, params_cls=ALSParams, build=ALS, grid=GRID)


def main(argv: list[str] | None = None) -> dict:
    return embedding_runner.main(SPEC, argv)


if __name__ == "__main__":
    main()
