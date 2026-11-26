"""Mô hình gu SASRec: học trên chuỗi bài của phiên train, gợi ý từ đặc trưng ở vị trí cuối của seed (#16, spec §6).

Huấn luyện bằng cross-entropy với mẫu âm dùng chung trong batch (Klenitskiy & Vasilev, RecSys 2023), không phải
BCE 1 mẫu âm như bản gốc. Mỗi epoch mỗi playlist cho một đoạn liên tiếp ngẫu nhiên (data.training_windows).
Có eval_targets thì sau mỗi epoch đo NDCG@10 trên một mẫu phiên val, giữ trạng thái tốt nhất và dừng sớm.
"""
from __future__ import annotations

import copy
import time
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.data.schema import N_CANDIDATES
from src.eval.run import VAL, evaluate
from src.models.base import export_candidates, top_k_by_score
from src.models.popularity import Popularity
from src.models.sasrec.data import PAD, train_sequences, training_windows
from src.models.sasrec.network import SASRecNetwork


@dataclass(frozen=True)
class SASRecParams:
    hidden: int = 128
    blocks: int = 2
    heads: int = 2
    dropout: float = 0.2
    lr: float = 0.001
    batch_size: int = 256
    maxlen: int = 50          # spec §6: độ dài tối đa 50
    negatives: int = 256      # mẫu âm dùng chung cho cả batch
    max_epochs: int = 200
    patience: int = 3         # dừng khi NDCG@10 val không tăng sau từng ấy epoch
    minutes: float = 40.0     # dừng khi hết thời gian (tiêu chí #16: ≤ 1 giờ GPU)
    val_sessions: int = 2000  # số phiên val đo sau mỗi epoch


class SASRec:
    name = "sasrec"

    def __init__(self, params: SASRecParams = SASRecParams(), random_seed: int = 0, device: str | None = None):
        self.params = params
        self.random_seed = random_seed
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.history: list[dict] = []

    def _build(self, track_ids: np.ndarray, sessions: pd.DataFrame) -> None:
        p = self.params
        self.index = pd.Index(track_ids)
        self.popularity = Popularity().fit(sessions)
        self.network = SASRecNetwork(len(track_ids), p.maxlen, p.hidden, p.blocks, p.heads, p.dropout).to(self.device)

    def fit(self, sessions: pd.DataFrame, eval_targets: pd.DataFrame | None = None) -> SASRec:
        torch.manual_seed(self.random_seed)
        rng = np.random.default_rng(self.random_seed)
        track_ids, sequences = train_sequences(sessions)
        if not any(len(s) >= 2 for s in sequences):
            raise ValueError("không có phiên train nào đủ 2 bài để học SASRec")
        self._build(track_ids, sessions)
        p = self.params
        optimizer = torch.optim.Adam(self.network.parameters(), lr=p.lr, betas=(0.9, 0.98))
        val = self._val_sample(eval_targets, rng) if eval_targets is not None else None

        best_state, best_ndcg, stale, started = None, -1.0, 0, time.monotonic()
        for epoch in range(1, p.max_epochs + 1):
            loss = self._train_epoch(sequences, optimizer, rng)
            record = {"epoch": epoch, "loss": loss, "seconds": round(time.monotonic() - started, 1)}
            if val is not None:
                ndcg = evaluate(export_candidates(self, val, VAL), val, sample=True).ndcg
                record["val_ndcg"] = ndcg
                if ndcg > best_ndcg:
                    best_state, best_ndcg, stale = copy.deepcopy(self.network.state_dict()), ndcg, 0
                else:
                    stale += 1
            self.history.append(record)
            if stale >= p.patience or time.monotonic() - started > p.minutes * 60:
                break
        if best_state is not None:
            self.network.load_state_dict(best_state)
        self.network.eval()
        return self

    def _val_sample(self, eval_targets: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
        val = eval_targets[eval_targets["split"] == VAL]
        n = min(self.params.val_sessions, len(val))
        return val.iloc[np.sort(rng.choice(len(val), size=n, replace=False))]

    def _train_epoch(self, sequences: list[np.ndarray], optimizer: torch.optim.Optimizer, rng: np.random.Generator) -> float:
        p = self.params
        inputs, targets = training_windows(sequences, p.maxlen, rng)
        order = rng.permutation(len(inputs))
        n_items = len(self.index)
        self.network.train()
        losses = []
        for start in range(0, len(order), p.batch_size):
            batch = order[start : start + p.batch_size]
            seqs = torch.as_tensor(inputs[batch], device=self.device)
            nexts = torch.as_tensor(targets[batch], device=self.device)
            features = self.network(seqs)[nexts != PAD]  # (M, hidden): mọi vị trí có bài kế tiếp
            positives = nexts[nexts != PAD]
            negatives = torch.randint(1, n_items + 1, (p.negatives,), device=self.device)
            item_emb = self.network.item_emb
            pos_logits = (features * item_emb(positives)).sum(-1, keepdim=True)
            neg_logits = features @ item_emb(negatives).T
            neg_logits = neg_logits.masked_fill(negatives[None, :] == positives[:, None], float("-inf"))  # mẫu âm trùng bài đúng
            logits = torch.cat([pos_logits, neg_logits], dim=1)
            loss = torch.nn.functional.cross_entropy(logits, torch.zeros(len(logits), dtype=torch.long, device=self.device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        self.network.eval()
        return float(np.mean(losses))

    def candidates(self, session_track_ids: Sequence[str], k: int = N_CANDIDATES) -> tuple[list[str], np.ndarray]:
        """Đặc trưng ở vị trí cuối của phiên (bỏ bài không có trong train) · embedding mọi bài; không có bài nào thì Popularity."""
        session = list(session_track_ids)
        codes = self.index.get_indexer(session)
        known = codes[codes >= 0][-self.params.maxlen :] + 1
        if len(known) == 0:
            return self.popularity.candidates(session, k=k)
        seq = np.full((1, self.params.maxlen), PAD, dtype=np.int64)
        seq[0, -len(known) :] = known
        with torch.no_grad():
            last = self.network(torch.as_tensor(seq, device=self.device))[0, -1]
            scores = (self.network.item_emb.weight[1:] @ last).float().cpu().numpy()
        return top_k_by_score(self.index.to_numpy(), scores, session, k)

    def save(self, path: str | Path) -> Path:
        """Checkpoint: trọng số, tham số, danh mục bài (mã = vị trí + 1) và lịch sử huấn luyện."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.network.state_dict(),
                "params": asdict(self.params),
                "track_ids": self.index.tolist(),
                "random_seed": self.random_seed,
                "history": self.history,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, path: str | Path, sessions: pd.DataFrame, device: str | None = None) -> SASRec:
        """Nạp checkpoint để gợi ý; sessions dùng cho mô hình dự phòng Popularity."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = cls(SASRecParams(**checkpoint["params"]), checkpoint["random_seed"], device)
        model._build(np.asarray(checkpoint["track_ids"], dtype=object), sessions)
        model.network.load_state_dict(checkpoint["state_dict"])
        model.network.eval()
        model.history = checkpoint["history"]
        return model
