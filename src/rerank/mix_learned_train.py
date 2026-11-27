"""Huấn luyện điểm mix học bằng InfoNCE (ticket #23, spec §6), rồi xuất trọng số numpy cho app.

    python -m src.rerank.mix_learned_train --data <thư mục mixnext-splited hoặc data/fake> --out <thư mục ra> [--run 0]

Ghi mix_learned.npz (trọng số f_out, f_in cho LearnedMix.load, chép vào artifacts/ cho app) và mix_learned.json
(tham số, số cặp, loss và Recall@10 trong batch trên cặp val theo từng epoch). In ra JSON đó.

Mẫu dương là cặp bài liền nhau trong playlist train (a, b⁺); mẫu âm là các bài kế tiếp khác trong cùng batch:
    L = −log exp(s_learn(a, b⁺) / τ) / Σ_j exp(s_learn(a, b_j) / τ)
Bài kế tiếp trùng track với b⁺ không bị tính là mẫu âm. Dừng sớm theo loss trên cặp val; cặp val chỉ dùng để
theo dõi, không dùng để học. Module này cần torch; app chỉ cần mix_learned.py (numpy).
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import itertools
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from src.eval.run import VAL, K
from src.features.track_features import TrackFeatures
from src.rerank.mix_learned import FEATURE_NAMES, MixEncoders, audio_features, consecutive_pairs

TRAIN = "train"
WEIGHTS_FILE = "mix_learned.npz"
REPORT_FILE = "mix_learned.json"


@dataclass(frozen=True)
class MixTrainParams:
    hidden: tuple[int, ...] = (64, 64)  # số nơ-ron các lớp ẩn của f_out và f_in
    dim: int = 32                      # số chiều đầu ra (trước khi chuẩn hóa L2)
    temperature: float = 0.1           # τ
    batch_size: int = 1024             # cũng là số mẫu (1 dương, còn lại âm) cho mỗi bài đang phát
    lr: float = 1e-3
    weight_decay: float = 1e-4
    max_epochs: int = 30
    patience: int = 3                  # dừng khi loss val không giảm sau chừng này epoch
    max_val_pairs: int = 50_000        # mẫu cặp val cố định để theo dõi
    minutes: float = 30.0              # giới hạn thời gian huấn luyện


def _mlp(n_in: int, hidden: tuple[int, ...], dim: int) -> nn.Sequential:
    sizes = (n_in, *hidden, dim)
    layers: list[nn.Module] = []
    for i, (a, b) in enumerate(itertools.pairwise(sizes)):
        layers.append(nn.Linear(a, b))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class MixModel(nn.Module):
    """f_out, f_in trên ma trận đặc trưng của cả danh mục; bài được gọi theo chỉ số dòng."""

    def __init__(self, x: np.ndarray, params: MixTrainParams):
        super().__init__()
        self.x: torch.Tensor
        self.register_buffer("x", torch.as_tensor(x, dtype=torch.float32))
        self.f_out = _mlp(x.shape[1], params.hidden, params.dim)
        self.f_in = _mlp(x.shape[1], params.hidden, params.dim)
        self.temperature = params.temperature

    def s_learn_rows(self, prev_rows: torch.Tensor, next_rows: torch.Tensor) -> torch.Tensor:
        """Ma trận s_learn shape (len(prev_rows), len(next_rows)), trong [−1, 1]."""
        out = F.normalize(self.f_out(self.x[prev_rows]), dim=1)
        inn = F.normalize(self.f_in(self.x[next_rows]), dim=1)
        return out @ inn.T

    def export(self) -> MixEncoders:
        def layers(seq: nn.Sequential):
            return [(m.weight.detach().cpu().numpy().astype(np.float32), m.bias.detach().cpu().numpy().astype(np.float32)) for m in seq if isinstance(m, nn.Linear)]

        return MixEncoders(f_out=layers(self.f_out), f_in=layers(self.f_in), temperature=self.temperature)


def _batch_stats(model: MixModel, prev_rows: torch.Tensor, next_rows: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
    """Loss InfoNCE trung bình của batch, kèm Recall@K của bài kế tiếp thật trong batch và mức ngẫu nhiên."""
    logits = model.s_learn_rows(prev_rows, next_rows) / model.temperature
    n = len(prev_rows)
    eye = torch.eye(n, dtype=torch.bool, device=logits.device)
    duplicate = (next_rows[:, None] == next_rows[None, :]) & ~eye  # cùng bài với b⁺: không phải mẫu âm
    logits = logits.masked_fill(duplicate, float("-inf"))
    loss = F.cross_entropy(logits, torch.arange(n, device=logits.device))
    with torch.no_grad():
        positive = logits.diagonal()[:, None]
        # hòa điểm tính là thua (bộ mã hóa hằng số có Recall = 0, không phải 1)
        rank = ((logits >= positive) & ~eye).sum(dim=1)
        n_candidates = (~duplicate).sum(dim=1).float()
        stats = {
            "loss": float(loss) * n,
            "hits": float((rank < K).sum()),
            "random": float((torch.clamp(n_candidates, max=K) / n_candidates).sum()),
            "n": n,
        }
    return loss, stats


def _evaluate(model: MixModel, batches: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    model.eval()
    total = {"loss": 0.0, "hits": 0.0, "random": 0.0, "n": 0}
    with torch.no_grad():
        for prev_rows, next_rows in batches:
            _, stats = _batch_stats(model, prev_rows, next_rows)
            for key in total:
                total[key] += stats[key]
    n = max(total["n"], 1)
    return {"val_loss": total["loss"] / n, "val_recall_at_10": total["hits"] / n, "random_recall_at_10": total["random"] / n}


def _rows(features: TrackFeatures, pairs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return features.rows(pairs["prev_track_id"].tolist()), features.rows(pairs["next_track_id"].tolist())


def _batches(prev: np.ndarray, nxt: np.ndarray, size: int, device: torch.device) -> list[tuple[torch.Tensor, torch.Tensor]]:
    return [
        (torch.as_tensor(prev[i : i + size], device=device), torch.as_tensor(nxt[i : i + size], device=device))
        for i in range(0, len(prev), size)
        if len(prev[i : i + size]) >= 2
    ]


@dataclass
class TrainResult:
    model: MixModel
    encoders: MixEncoders
    params: MixTrainParams
    history: list[dict[str, Any]] = field(default_factory=list)  # epoch 0 = bộ mã hóa chưa học
    best_epoch: int = 0
    n_train_pairs: int = 0
    n_val_pairs: int = 0
    seconds: float = 0.0

    @property
    def best_epoch_stats(self) -> dict:
        return next(h for h in self.history if h["epoch"] == self.best_epoch)

    def report(self) -> dict:
        return {
            "params": {**self.params.__dict__, "hidden": list(self.params.hidden)},
            "feature_names": list(FEATURE_NAMES),
            "n_train_pairs": self.n_train_pairs,
            "n_val_pairs": self.n_val_pairs,
            "best_epoch": self.best_epoch,
            "val": self.best_epoch_stats,
            "history": self.history,
            "seconds": round(self.seconds, 1),
        }


def train(tracks: pd.DataFrame, sessions: pd.DataFrame, params: MixTrainParams = MixTrainParams(), random_seed: int = 0) -> TrainResult:
    """Học f_out, f_in từ các cặp liền nhau trong phiên train; theo dõi và dừng sớm trên cặp val."""
    t0 = time.time()
    rng = np.random.default_rng(random_seed)
    torch.manual_seed(random_seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    features = TrackFeatures.from_tracks(tracks)
    train_prev, train_next = _rows(features, consecutive_pairs(sessions, TRAIN))
    val_pairs = consecutive_pairs(sessions, VAL)
    if len(val_pairs) > params.max_val_pairs:
        val_pairs = val_pairs.iloc[np.sort(rng.choice(len(val_pairs), params.max_val_pairs, replace=False))]
    val_order = rng.permutation(len(val_pairs))  # batch val trộn nhiều phiên, như batch train
    val_prev, val_next = (r[val_order] for r in _rows(features, val_pairs))
    val_batches = _batches(val_prev, val_next, params.batch_size, device)
    if len(train_prev) < 2 or not val_batches:
        raise ValueError(f"cần ≥ 2 cặp train và ≥ 2 cặp val, đang có {len(train_prev)} và {len(val_prev)}")

    model = MixModel(audio_features(features), params).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=params.lr, weight_decay=params.weight_decay)
    val = _evaluate(model, val_batches)
    history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, **val}]
    best_loss, best_epoch, best_state = val["val_loss"], 0, copy.deepcopy(model.state_dict())

    for epoch in range(1, params.max_epochs + 1):
        model.train()
        order = rng.permutation(len(train_prev))
        losses, n = 0.0, 0.0
        for prev_rows, next_rows in _batches(train_prev[order], train_next[order], params.batch_size, device):
            loss, stats = _batch_stats(model, prev_rows, next_rows)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses, n = losses + stats["loss"], n + stats["n"]
        val = _evaluate(model, val_batches)
        history.append({"epoch": epoch, "train_loss": losses / n, **val})
        if val["val_loss"] < best_loss:
            best_loss, best_epoch, best_state = val["val_loss"], epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= params.patience or time.time() - t0 > params.minutes * 60:
            break

    model.load_state_dict(best_state)
    model.eval()
    return TrainResult(
        model=model.cpu(),
        encoders=model.export(),
        params=params,
        history=history,
        best_epoch=best_epoch,
        n_train_pairs=len(train_prev),
        n_val_pairs=len(val_pairs),
        seconds=time.time() - t0,
    )


def run(data_dir: str | Path, out_dir: str | Path, run_number: int = 0, params: MixTrainParams = MixTrainParams()) -> dict:
    """Lần chạy run_number (cũng là random_seed): học, ghi mix_learned.npz và mix_learned.json vào out_dir."""
    data, out = Path(data_dir), Path(out_dir)
    result = train(pd.read_parquet(data / "tracks.parquet"), pd.read_parquet(data / "sessions.parquet"), params, random_seed=run_number)
    result.encoders.save(out / WEIGHTS_FILE)
    report = {"run": run_number, **result.report()}
    (out / REPORT_FILE).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description="Điểm mix học bằng InfoNCE: huấn luyện và xuất trọng số numpy (#23)")
    ap.add_argument("--data", required=True, help="Thư mục có tracks.parquet, sessions.parquet")
    ap.add_argument("--out", required=True, help="Thư mục ra")
    ap.add_argument("--run", type=int, default=0, help="Số lần chạy, cũng là random_seed")
    ap.add_argument("--max-epochs", type=int, help="Ghi đè MixTrainParams.max_epochs")
    ap.add_argument("--minutes", type=float, help="Ghi đè MixTrainParams.minutes (giới hạn thời gian huấn luyện)")
    args = ap.parse_args(argv)
    overrides = {"max_epochs": args.max_epochs, "minutes": args.minutes}
    params = dataclasses.replace(MixTrainParams(), **{key: value for key, value in overrides.items() if value is not None})
    report = run(args.data, args.out, args.run, params)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


if __name__ == "__main__":
    main()
