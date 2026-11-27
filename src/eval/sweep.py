"""Quét α: độ chính xác và độ mượt theo α, ra đường đánh đổi (ticket #19, spec §7).

    python -m src.eval.sweep --data <thư mục tracks/sessions/eval_targets> --candidates <itemknn_val_run0.parquet> --out reports/
    python -m src.eval.sweep --plot-only reports/sweep_v1.csv --out reports/     # chỉ vẽ lại hình từ bảng đã có

Độ chính xác: xếp lại toàn bộ candidates val theo điểm cuối (giữ cả danh sách để MRR so được với file gốc).
Độ mượt: sinh chuỗi tham lam 10 bài (#18) trên một mẫu phiên val cố định theo random_seed, rồi đo (#13).
"""
from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.schema import N_CANDIDATES
from src.eval.run import VAL, evaluate
from src.eval.smoothness import smoothness
from src.features.track_features import TrackFeatures
from src.models.base import TasteModel
from src.models.item_knn import ItemKNN
from src.models.popularity import Popularity
from src.rerank.combine import MixScorer, rerank_candidates
from src.rerank.generate import generate_chains
from src.rerank.mix_rule import RuleMix

ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))  # 0, 0.1, …, 1
N_SMOOTH = 2000      # số phiên val để sinh chuỗi đo độ mượt (cùng cỡ mẫu với grid search #25)
RANDOM_SEED = 42
MAX_NDCG_DROP = 0.10  # tiêu chí thành công (spec §7): NDCG@10 giảm ≤ 10% so với α = 1
MODELS = ("itemknn", "popularity")  # mô hình gu học lại nhanh trên train để sinh chuỗi; ALS/SASRec cần notebook riêng


def smoothness_seeds(eval_targets: pd.DataFrame, n: int = N_SMOOTH, random_seed: int = RANDOM_SEED) -> list[list[str]]:
    """Seed của một mẫu cố định n phiên val (ít hơn n phiên thì lấy hết), theo thứ tự session_id."""
    val = eval_targets[eval_targets["split"] == VAL].sort_values("session_id")
    if len(val) > n:
        val = val.iloc[np.sort(np.random.default_rng(random_seed).choice(len(val), n, replace=False))]
    return [list(seed) for seed in val["seed_track_ids"]]


def sweep(
    candidates: pd.DataFrame,
    eval_targets: pd.DataFrame,
    model: TasteModel,
    mix: MixScorer,
    features: TrackFeatures,
    alphas: tuple[float, ...] = ALPHAS,
    n_smooth: int = N_SMOOTH,
    random_seed: int = RANDOM_SEED,
) -> pd.DataFrame:
    """Mỗi α một dòng: hit_rate, ndcg, mrr, n_sessions (cả val) và các chỉ số độ mượt (mẫu n_smooth phiên)."""
    seeds = smoothness_seeds(eval_targets, n_smooth, random_seed)
    rows = []
    for alpha in alphas:
        accuracy = evaluate(rerank_candidates(candidates, eval_targets, mix, alpha, k=None), eval_targets)
        smooth = smoothness(generate_chains(model, seeds, mix, alpha, n_candidates=N_CANDIDATES), features)
        rows.append(
            {
                "alpha": alpha,
                "hit_rate": accuracy.hit_rate,
                "ndcg": accuracy.ndcg,
                "mrr": accuracy.mrr,
                "n_sessions": accuracy.n_sessions,
                **{key: value for key, value in asdict(smooth).items()},
            }
        )
    return pd.DataFrame(rows)


def plot_tradeoff(table: pd.DataFrame, path: str | Path, title: str = "") -> Path:
    """Hình đường đánh đổi: trái NDCG@10 theo % cặp mix được (mỗi điểm một α), phải hai chỉ số theo α."""
    import matplotlib

    matplotlib.use("Agg")  # không cần màn hình (Kaggle, test)
    import matplotlib.pyplot as plt

    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(table["mixable_pct"], table["ndcg"], marker="o", color="#3b6ea5")
    for _, row in table.iterrows():
        left.annotate(f"α={row['alpha']:.1f}", (row["mixable_pct"], row["ndcg"]), textcoords="offset points", xytext=(4, 4), fontsize=8)
    threshold = (1 - MAX_NDCG_DROP) * float(table.loc[table["alpha"].idxmax(), "ndcg"])
    for ax in (left, right):
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"NDCG giảm {MAX_NDCG_DROP:.0%} so với α = 1")
    left.legend(loc="lower left", fontsize=8)
    left.set_xlabel("% cặp mix được (chuỗi 10 bài)")
    left.set_ylabel("NDCG@10 (val)")
    left.set_title("Đường đánh đổi")
    left.grid(alpha=0.3)

    right.plot(table["alpha"], table["ndcg"], marker="o", color="#3b6ea5", label="NDCG@10")
    right.set_xlabel("α (1 = chỉ theo gu, 0 = chỉ theo mix)")
    right.set_ylabel("NDCG@10", color="#3b6ea5")
    twin = right.twinx()
    twin.plot(table["alpha"], table["mixable_pct"], marker="s", color="#c0504d", label="% cặp mix được")
    twin.set_ylabel("% cặp mix được", color="#c0504d")
    right.set_title("Theo α")
    right.grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Quét α: độ chính xác và độ mượt, ra đường đánh đổi (#19)")
    ap.add_argument("--data", help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--candidates", help="File candidates val của mô hình gu (ví dụ itemknn_val_run0.parquet)")
    ap.add_argument("--plot-only", help="Chỉ vẽ lại hình từ một sweep_<tag>.csv đã có, không chạy quét")
    ap.add_argument("--model", choices=MODELS, default="itemknn", help="Mô hình gu để sinh chuỗi; nên trùng với file candidates")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-smooth", type=int, default=N_SMOOTH)
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--tag", default="v1", help="Hậu tố tên file: sweep_<tag>.csv, fig_tradeoff_<tag>.png")
    args = ap.parse_args(argv)
    out = Path(args.out)
    title = f"{args.model} + mix theo luật (w 0.4/0.4/0.2)"
    if args.plot_only:
        table = pd.read_csv(args.plot_only)
        plot_tradeoff(table, out / f"fig_tradeoff_{args.tag}.png", title=title)
        return table
    if not (args.data and args.candidates):
        ap.error("cần --data và --candidates (hoặc --plot-only)")

    data = Path(args.data)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")
    candidates = pd.read_parquet(args.candidates)
    train = sessions[sessions["split"] == "train"]
    model: TasteModel = ItemKNN().fit(train) if args.model == "itemknn" else Popularity().fit(train)
    mix = RuleMix(tracks)

    t0 = time.perf_counter()
    table = sweep(candidates, eval_targets, model, mix, mix.features, n_smooth=args.n_smooth, random_seed=args.random_seed)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"sweep_{args.tag}.csv", index=False)
    plot_tradeoff(table, out / f"fig_tradeoff_{args.tag}.png", title=title)
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"{time.perf_counter() - t0:.0f} giây")
    return table


if __name__ == "__main__":
    main()
