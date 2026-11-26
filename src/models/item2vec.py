"""Mô hình gu item2vec: Word2Vec (gensim) trên chuỗi bài của mỗi playlist train (ticket #15, spec §6).

Mỗi playlist là một "câu", mỗi bài là một "từ" (Barkan & Koenigstein 2016). Chỉ dùng để học vector bài;
ứng viên tính bằng vector phiên như ALS (#14), nên app tính lại được bằng numpy từ item_embeddings/item2vec.npy.
Chọn tham số và CLI ở src/models/embedding_runner.py.

    python -m src.models.item2vec --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra>
    python -m src.models.item2vec --data ... --out ... --run 1 --params <out>/item2vec_params.json

Học nhiều luồng (workers = số CPU) cho nhanh, nên cùng random_seed không ra cùng vector từng bit
(gensim chỉ tái lập được với workers=1 và PYTHONHASHSEED cố định).

Vòng 1 trên Kaggle (window {3, 10} × vector_size {128, 256}, tham số mặc định của gensim): NDCG@10 val chỉ 0,020–0,028,
window và vector_size gần như không ảnh hưởng. Vòng 2 giữ (3, 128) và thử epochs, ns_exponent
(Caselles-Dupré và cộng sự 2018: với dữ liệu nghe nhạc, hai tham số này quan trọng) và chuẩn hóa vector bài
(normalize, xem SessionVectorTaste). Bảng hai vòng ở notebooks/06_item2vec.ipynb.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.data.schema import session_track_lists
from src.models import embedding_runner
from src.models.embedding_runner import EmbeddingSpec, EmbeddingTasteModel

SKIP_GRAM = 1  # skip-gram như bài item2vec gốc
NEGATIVE = 5   # số mẫu âm mỗi cặp (mặc định của gensim)
GRID = {
    "window": (3,),
    "vector_size": (128,),
    "epochs": (5, 20),
    "ns_exponent": (0.75, -0.5),  # 0,75 là mặc định của gensim; âm thì lấy mẫu âm nghiêng về bài ít phổ biến
    "normalize": (False, True),
    "decay": (0.5, 0.7, 0.9, 1.0),
}


@dataclass(frozen=True)
class Item2VecParams:
    """Tham số chọn trên val: của Word2Vec (window, vector_size, epochs, ns_exponent) và cách chấm (normalize, decay)."""

    window: int = 3
    vector_size: int = 128
    epochs: int = 5
    ns_exponent: float = 0.75
    normalize: bool = False
    decay: float = 0.5


class Item2Vec(EmbeddingTasteModel):
    name = "item2vec"

    def __init__(self, params: Item2VecParams = Item2VecParams(), random_seed: int = 0, workers: int | None = None):
        super().__init__(params, random_seed)
        self.workers = workers if workers is not None else os.cpu_count() or 1  # None: mọi CPU

    def _learn(self, train: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        from gensim.models import Word2Vec  # nhập muộn: app và các mô hình khác không cần gensim

        playlists = session_track_lists(train).tolist()
        p = self._init_params
        model = Word2Vec(
            playlists,
            vector_size=p.vector_size,
            window=p.window,
            sg=SKIP_GRAM,
            negative=NEGATIVE,
            min_count=1,  # mọi bài train đều có vector
            epochs=p.epochs,
            ns_exponent=p.ns_exponent,
            seed=self.random_seed,
            workers=self.workers,
        )
        track_ids = np.sort(np.asarray(model.wv.index_to_key, dtype=object))  # gensim xếp theo tần suất; đổi về theo track_id
        return model.wv[track_ids.tolist()].astype(np.float32), track_ids


SPEC = EmbeddingSpec(name=Item2Vec.name, params_cls=Item2VecParams, build=Item2Vec, grid=GRID)


def main(argv: list[str] | None = None) -> dict:
    return embedding_runner.main(SPEC, argv)


if __name__ == "__main__":
    main()
