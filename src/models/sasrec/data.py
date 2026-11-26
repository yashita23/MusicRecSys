"""Chuỗi huấn luyện cho SASRec: mã hóa phiên train và cắt cửa sổ (đầu vào, bài kế tiếp)."""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from src.data.schema import session_track_lists

PAD = 0  # mã đệm; bài thật mã từ 1


def train_sequences(sessions: pd.DataFrame) -> tuple[np.ndarray, list[np.ndarray]]:
    """Bài train sắp theo track_id (mã = vị trí + 1) và chuỗi mã của từng phiên train theo thứ tự nghe (theo session_id)."""
    train = sessions[sessions["split"] == "train"]
    track_ids = np.sort(train["track_id"].unique()).astype(object)
    index = pd.Index(track_ids)
    sequences = [index.get_indexer(tracks) + 1 for tracks in session_track_lists(train)]
    return track_ids, sequences


def training_windows(sequences: Sequence[np.ndarray], maxlen: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Mỗi phiên ≥ 2 bài cho một cửa sổ liên tiếp ngẫu nhiên dài tối đa maxlen + 1 bài.

    Đầu vào là cửa sổ bỏ bài cuối, bài kế tiếp là cửa sổ bỏ bài đầu; cả hai đệm PAD bên trái đến maxlen.
    Gọi lại mỗi epoch để playlist dài hơn maxlen được dùng hết các đoạn.
    """
    usable = [s for s in sequences if len(s) >= 2]
    inputs = np.full((len(usable), maxlen), PAD, dtype=np.int64)
    targets = np.full((len(usable), maxlen), PAD, dtype=np.int64)
    for row, seq in enumerate(usable):
        width = min(len(seq), maxlen + 1)
        start = int(rng.integers(0, len(seq) - width + 1))
        window = seq[start : start + width]
        inputs[row, maxlen - width + 1 :] = window[:-1]
        targets[row, maxlen - width + 1 :] = window[1:]
    return inputs, targets
