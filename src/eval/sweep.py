"""Quét α: độ chính xác và độ mượt theo α, ra đường đánh đổi (ticket #19, spec §6–§7).

    python -m src.eval.sweep --data <thư mục tracks/sessions/eval_targets> --candidates <itemknn_val_run0.parquet> --out reports/
    python -m src.eval.sweep --plot-only reports/sweep_v1.csv --out reports/     # chỉ vẽ lại hình từ bảng đã có

Độ chính xác: xếp lại toàn bộ candidates val theo điểm cuối (giữ cả danh sách để MRR so được với file gốc).
Độ mượt: sinh chuỗi tham lam 10 bài (#18) trên một mẫu phiên val cố định theo random_seed, rồi đo (#13).
"""
from __future__ import annotations

import argparse
import re
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.run import VAL, evaluate
from src.eval.smoothness import smoothness
from src.features.track_features import TrackFeatures
from src.models.base import TasteModel
from src.models.item_knn import ItemKNN
from src.models.popularity import Popularity
from src.rerank.combine import MixScorer, rerank_candidates
from src.rerank.generate import generate_chains
from src.rerank.mix_rule import MixWeights, RuleMix

ALPHAS = tuple(round(0.1 * i, 1) for i in range(11))  # 0, 0.1, …, 1
N_SMOOTH = 2000      # số phiên val để sinh chuỗi đo độ mượt (cùng cỡ mẫu với grid search #25)
RANDOM_SEED = 42
MAX_NDCG_DROP = 0.10  # spec §6: chọn α có % cặp mix được cao nhất trong các α mà NDCG@10 giảm ≤ 10% so với α = 1
# Mô hình gu học lại nhanh trên train để sinh chuỗi. ALS/item2vec/SASRec cần tham số và môi trường riêng (notebook 05–07).
MODELS: dict[str, type] = {"itemknn": ItemKNN, "popularity": Popularity}


def val_sample(eval_targets: pd.DataFrame, n: int = N_SMOOTH, random_seed: int = RANDOM_SEED) -> pd.DataFrame:
    """Một mẫu cố định n phiên val của eval_targets (ít hơn n phiên thì lấy hết), theo thứ tự session_id."""
    val = eval_targets[eval_targets["split"] == VAL].sort_values("session_id")
    if len(val) > n:
        val = val.iloc[np.sort(np.random.default_rng(random_seed).choice(len(val), n, replace=False))]
    return val


def smoothness_seeds(eval_targets: pd.DataFrame, n: int = N_SMOOTH, random_seed: int = RANDOM_SEED) -> list[list[str]]:
    """Seed của val_sample: các phiên dùng để sinh chuỗi đo độ mượt."""
    return [list(seed) for seed in val_sample(eval_targets, n, random_seed)["seed_track_ids"]]


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
    """Mỗi α một dòng: hit_rate, ndcg, mrr, n_sessions (cả val) và các chỉ số độ mượt (mẫu n_smooth phiên).

    features là đặc trưng âm thanh để đo độ mượt, tách khỏi mix vì điểm mix học (#23) không có đặc trưng riêng.
    """
    seeds = smoothness_seeds(eval_targets, n_smooth, random_seed)
    rows = []
    for alpha in alphas:
        accuracy = evaluate(rerank_candidates(candidates, eval_targets, mix, alpha, k=None), eval_targets)
        smooth = smoothness(generate_chains(model, seeds, mix, alpha), features)
        rows.append(
            {"alpha": alpha, "hit_rate": accuracy.hit_rate, "ndcg": accuracy.ndcg, "mrr": accuracy.mrr,
             "n_sessions": accuracy.n_sessions, **asdict(smooth)}
        )
    return pd.DataFrame(rows)


def ndcg_threshold(table: pd.DataFrame) -> float:
    """NDCG@10 tối thiểu được chấp nhận: (1 − MAX_NDCG_DROP) × NDCG@10 ở α = 1."""
    at_one = table.loc[np.isclose(table["alpha"], 1.0), "ndcg"]
    if at_one.empty:
        raise ValueError("bảng quét phải có dòng α = 1 làm mốc NDCG")
    return (1 - MAX_NDCG_DROP) * float(at_one.iloc[0])


def best_row(table: pd.DataFrame):
    """Nhãn dòng có % cặp mix được cao nhất trong các dòng mà NDCG@10 ≥ ngưỡng (spec §6).

    Hòa thì lấy α lớn hơn (ít đổi gu hơn), rồi dòng đứng trước trong bảng (grid search #25: bộ w đứng trước trong lưới).
    """
    ok = table[table["ndcg"] >= ndcg_threshold(table) - 1e-12]
    return ok.sort_values(["mixable_pct", "alpha"], ascending=[False, False], kind="stable").index[0]


def choose_alpha(table: pd.DataFrame) -> float:
    """α của best_row: α có % cặp mix được cao nhất mà NDCG@10 giảm ≤ 10% so với α = 1."""
    return float(table.loc[best_row(table), "alpha"])


def _title(table: pd.DataFrame) -> str:
    model = table["model"].iloc[0] if "model" in table else "mô hình gu"
    weights = f" (w {table['mix_weights'].iloc[0]})" if "mix_weights" in table else ""
    return f"{model} + mix theo luật{weights}"


def plot_tradeoff(table: pd.DataFrame, path: str | Path) -> Path:
    """Hình đường đánh đổi: trái NDCG@10 theo % cặp mix được (mỗi điểm một α), phải hai chỉ số theo α.

    Đường ngang: ngưỡng NDCG −10%; điểm khoanh: α được chọn theo spec §6.
    """
    import matplotlib

    matplotlib.use("Agg")  # không cần màn hình (Kaggle, test)
    import matplotlib.pyplot as plt

    threshold, chosen = ndcg_threshold(table), choose_alpha(table)
    pick = table.loc[np.isclose(table["alpha"], chosen)].iloc[0]
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 4.2))
    left.plot(table["mixable_pct"], table["ndcg"], marker="o", color="#3b6ea5")
    for _, row in table.iterrows():  # nhãn cách một α: các α gần 1 nằm sát nhau; đủ mọi α ở hình phải
        if round(row["alpha"] * 10) % 2 == 0:
            left.annotate(f"α={row['alpha']:.1f}", (row["mixable_pct"], row["ndcg"]), textcoords="offset points",
                          xytext=(5, 4), fontsize=8)
    left.scatter([pick["mixable_pct"]], [pick["ndcg"]], s=180, facecolors="none", edgecolors="#2e8b57", linewidths=2,
                 label=f"α được chọn = {chosen:.1f}", zorder=3)
    left.set_xlabel("% cặp mix được (chuỗi 10 bài)")
    left.set_ylabel("NDCG@10 (val)")
    left.set_title("Đường đánh đổi")

    right.plot(table["alpha"], table["ndcg"], marker="o", color="#3b6ea5", label="NDCG@10")
    twin = right.twinx()
    twin.plot(table["alpha"], table["mixable_pct"], marker="s", color="#c0504d", label="% cặp mix được")
    right.set_xlabel("α (1 = chỉ theo gu, 0 = chỉ theo mix)")
    right.set_ylabel("NDCG@10", color="#3b6ea5")
    twin.set_ylabel("% cặp mix được", color="#c0504d")
    right.set_title("Theo α")
    for ax in (left, right):
        ax.axhline(threshold, color="gray", linestyle="--", linewidth=1, label=f"NDCG giảm {MAX_NDCG_DROP:.0%} so với α = 1")
        ax.grid(alpha=0.3)
    left.legend(loc="lower left", fontsize=8)
    handles = right.get_legend_handles_labels()[0] + twin.get_legend_handles_labels()[0]
    right.legend(handles, [h.get_label() for h in handles], loc="center left", fontsize=8)
    fig.suptitle(_title(table))
    fig.tight_layout()
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _model_in_filename(path: str | Path) -> str | None:
    """'itemknn_val_run0.parquet' → 'itemknn' (None nếu tên file không theo bố cục data-contract)."""
    match = re.fullmatch(r"(.+)_(?:val|test)_run\d+\.parquet", Path(path).name)
    return match.group(1) if match else None


def run_sweep(
    data_dir: str | Path,
    candidates_path: str | Path,
    model_name: str,
    out_dir: str | Path,
    n_smooth: int = N_SMOOTH,
    random_seed: int = RANDOM_SEED,
    tag: str = "v1",
) -> pd.DataFrame:
    """Đọc dữ liệu, học lại mô hình gu trên train, quét α, ghi sweep_<tag>.csv và fig_tradeoff_<tag>.png."""
    in_file = _model_in_filename(candidates_path)
    if in_file in MODELS and in_file != model_name:
        raise ValueError(f"file ứng viên là của {in_file} nhưng mô hình sinh chuỗi là {model_name}: bảng sẽ trộn hai mô hình")
    data = Path(data_dir)
    tracks = pd.read_parquet(data / "tracks.parquet")
    sessions = pd.read_parquet(data / "sessions.parquet")
    eval_targets = pd.read_parquet(data / "eval_targets.parquet")
    model = MODELS[model_name]().fit(sessions)  # fit tự lọc phiên train
    mix = RuleMix(tracks)

    table = sweep(pd.read_parquet(candidates_path), eval_targets, model, mix, mix.features, n_smooth=n_smooth, random_seed=random_seed)
    w: MixWeights = mix.weights
    table["model"] = model_name
    table["mix_weights"] = f"{w.bpm:g}/{w.key:g}/{w.energy:g}"
    table["chosen"] = np.isclose(table["alpha"], choose_alpha(table))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / f"sweep_{tag}.csv", index=False)
    plot_tradeoff(table, out / f"fig_tradeoff_{tag}.png")
    return table


def main(argv: list[str] | None = None) -> pd.DataFrame:
    ap = argparse.ArgumentParser(description="Quét α: độ chính xác và độ mượt, ra đường đánh đổi (#19)")
    ap.add_argument("--data", help="Thư mục có tracks, sessions, eval_targets (.parquet)")
    ap.add_argument("--candidates", help="File candidates val của mô hình gu (ví dụ itemknn_val_run0.parquet)")
    ap.add_argument("--model", choices=sorted(MODELS), default="itemknn", help="Mô hình gu để sinh chuỗi; phải trùng với file candidates")
    ap.add_argument("--plot-only", help="Chỉ vẽ lại hình từ một sweep_<tag>.csv đã có, không chạy quét")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-smooth", type=int, default=N_SMOOTH)
    ap.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    ap.add_argument("--tag", default="v1", help="Hậu tố tên file: sweep_<tag>.csv, fig_tradeoff_<tag>.png")
    args = ap.parse_args(argv)
    if args.plot_only:
        table = pd.read_csv(args.plot_only)
        plot_tradeoff(table, Path(args.out) / f"fig_tradeoff_{args.tag}.png")
        return table
    if not (args.data and args.candidates):
        ap.error("cần --data và --candidates (hoặc --plot-only)")
    t0 = time.perf_counter()
    try:
        table = run_sweep(args.data, args.candidates, args.model, args.out, args.n_smooth, args.random_seed, args.tag)
    except ValueError as e:
        ap.error(str(e))
    print(table.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"α được chọn (spec §6): {choose_alpha(table):.1f} · {time.perf_counter() - t0:.0f} giây")
    return table


if __name__ == "__main__":
    main()
