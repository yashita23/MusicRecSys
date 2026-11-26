"""Giao diện chung của mô hình gu và cách xuất file candidates (ticket #9, docs/data-contract.md).

Hai baseline (Popularity, item-kNN liền kề) và các mô hình gu (ALS, item2vec, SASRec) học từ các phiên train qua `fit`, rồi
`candidates(phiên, k)` trả về top-k ứng viên kèm điểm gu, **không chứa bài nào đã có trong phiên**.
Khi đánh giá thì phiên là seed; khi sinh chuỗi tham lam (#18) thì phiên dài dần nên không lặp bài.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES, validate_candidates


class TasteModel(Protocol):
    """Giao diện chung của baseline và mô hình gu: cái gì sinh được candidates."""

    name: str  # tên dùng trong file candidates/{name}_{split}_run{r}.parquet

    def fit(self, sessions: pd.DataFrame) -> TasteModel: ...

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]: ...


def top_k_excluding(ordered_ids: np.ndarray, ordered_scores: np.ndarray, exclude: Sequence[str], k: int) -> tuple[list[str], np.ndarray]:
    """k bài đầu của một danh sách đã xếp sẵn, bỏ các bài trong exclude."""
    head = ordered_ids[: k + len(exclude)]
    keep = ~np.isin(head, list(exclude))
    ids, scores = head[keep][:k].tolist(), ordered_scores[: len(head)][keep][:k]
    if len(ids) < k:
        raise ValueError(f"chỉ có {len(ids)} ứng viên ngoài phiên, cần {k}: train quá ít bài")
    return ids, scores


def top_k_by_score(track_ids: np.ndarray, scores: np.ndarray, exclude: Sequence[str], k: int) -> tuple[list[str], np.ndarray]:
    """k bài điểm cao nhất (track_ids[i] có điểm scores[i]), bỏ các bài trong exclude; hòa thì theo thứ tự track_ids.

    Chỉ xếp k + len(exclude) bài đầu (argpartition), không xếp cả danh mục.
    """
    n_top = min(k + len(exclude), len(scores))  # đủ chỗ cho k ứng viên sau khi bỏ bài trong phiên
    top = np.argpartition(-scores, n_top - 1)[:n_top]
    top = top[np.lexsort((top, -scores[top]))]  # điểm giảm dần; hòa thì theo vị trí trong track_ids
    return top_k_excluding(track_ids[top], scores[top], exclude, k)


def export_candidates(model: TasteModel, eval_targets: pd.DataFrame, split: str, k: int = N_CANDIDATES) -> pd.DataFrame:
    """Bảng candidates của một split theo docs/data-contract.md: mỗi phiên k dòng, rank 1…k, seed bị loại."""
    parts = []
    for session_id, seed in eval_targets.loc[eval_targets["split"] == split, ["session_id", "seed_track_ids"]].itertuples(index=False):
        ids, scores = model.candidates(list(seed), k=k)
        parts.append(pd.DataFrame({"session_id": session_id, "rank": np.arange(1, len(ids) + 1), "track_id": ids, "taste_score": scores}))
    out = pd.concat(parts, ignore_index=True)
    validate_candidates(out, k=k)
    return out


def candidates_path(root: str | Path, model_name: str, split: str, run: int = 0) -> Path:
    return Path(root) / "candidates" / f"{model_name}_{split}_run{run}.parquet"


def save_candidates(candidates: dict[str, pd.DataFrame], root: str | Path, model_name: str, run: int = 0) -> list[Path]:
    """Ghi {split: bảng candidates} vào <root>/candidates/{model}_{split}_run{r}.parquet."""
    paths = []
    for split, df in candidates.items():
        path = candidates_path(root, model_name, split, run)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
        paths.append(path)
    return paths


def save_embeddings(vectors: np.ndarray, index: pd.DataFrame, root: str | Path, model_name: str) -> list[Path]:
    """Ghi <root>/item_embeddings/{model}.npy và {model}_tracks.parquet theo docs/data-contract.md."""
    folder = Path(root) / "item_embeddings"
    folder.mkdir(parents=True, exist_ok=True)
    npy, idx = folder / f"{model_name}.npy", folder / f"{model_name}_tracks.parquet"
    np.save(npy, vectors)
    index.to_parquet(idx, index=False)
    return [npy, idx]
