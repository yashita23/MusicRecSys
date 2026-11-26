"""Dữ liệu đồ chơi dùng chung cho test mô hình gu theo embedding (ALS #14, item2vec #15)."""
import numpy as np
import pandas as pd


def tid(prefix: str, i: int) -> str:
    return f"{prefix}{i}".ljust(22, "x")


ROCK = [tid("rock", i) for i in range(12)]
JAZZ = [tid("jazz", i) for i in range(12)]
NEVER_TRAIN = tid("val", 0)


def two_clusters(n_per_cluster: int = 40, length: int = 6, random_seed: int = 0) -> pd.DataFrame:
    """Playlist train chỉ chứa bài rock hoặc chỉ bài jazz; thêm một phiên val để chắc mô hình chỉ học từ train."""
    rng = np.random.default_rng(random_seed)
    rows = []
    for sid in range(2 * n_per_cluster):
        cluster = ROCK if sid % 2 == 0 else JAZZ
        for pos, t in enumerate(rng.choice(cluster, size=length, replace=False)):
            rows.append({"session_id": sid, "split": "train", "pos": pos, "track_id": t})
    rows += [{"session_id": 999, "split": "val", "pos": p, "track_id": NEVER_TRAIN} for p in range(6)]
    return pd.DataFrame(rows)
