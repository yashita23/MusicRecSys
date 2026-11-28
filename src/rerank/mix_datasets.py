"""Thí nghiệm điểm mix trên hai bộ dữ liệu khác bản chất: playlist MPD và lịch sử nghe last.fm (ticket #57).

Ba thí nghiệm, chạy cùng một cách trên mỗi bộ (run_dataset), gộp thành một bảng (summary_table):
- E1: cặp liền nhau thật có mượt hơn cặp ngẫu nhiên không, kèm khoảng tin cậy bootstrap theo phiên (cặp trong một phiên
  không độc lập, nên bootstrap rút lại cả phiên). Báo trên mọi phiên và trên phiên ≤ độ dài p99 của chính bộ đó.
- E2: điểm mix học (#23) huấn luyện lại trên cặp train của bộ đó; loss val, Recall@10 trong batch, đặt cạnh Recall@10 của
  điểm mix theo luật trên cùng các batch val.
- E3: tương quan Spearman giữa điểm học và luật trên cặp ngẫu nhiên (cùng cách đo với #24).
Chỉ học từ cặp train của mỗi bộ; không đem mô hình học trên bộ này áp sang bộ kia.
"""
from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from src.eval.run import VAL
from src.eval.smoothness import BPM_TOLERANCE, MIXABLE_MAX_BPM_DELTA
from src.features.track_features import TrackFeatures
from src.rerank.mix_compare import in_batch_recall, sample_pairs, score_pairs
from src.rerank.mix_learned import LearnedMix, consecutive_pairs
from src.rerank.mix_rule import RuleMix

if TYPE_CHECKING:
    from src.rerank.mix_learned_train import MixTrainParams

E1_METRICS = ("mixable_pct", "bpm_delta_pct", "energy_gap")  # như Smoothness (#13): % cặp mix được, ΔBPM %, |Δenergy|
CONFIDENCE = 0.95
REAL, RANDOM, DIFFERENCE = "thật", "ngẫu nhiên", "thật − ngẫu nhiên"
TRAIN = "train"  # không lấy từ mix_learned_train (#23) vì module đó cần torch
LONG_SESSION_QUANTILE = 0.99  # E1 báo thêm kết quả khi bỏ phiên dài hơn p99 của chính bộ dữ liệu


def _split_pairs(sessions: pd.DataFrame, split: str, max_session_length: int | None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(mã phiên 0…, bài đang phát, bài kế tiếp) của mọi cặp liền nhau trong split; bỏ cặp A → A như consecutive_pairs.

    Chép cách lọc của consecutive_pairs (#23) nhưng giữ thêm mã phiên cho bootstrap; không sửa #23 vì là file của A.

    max_session_length: bỏ cả phiên dài hơn số bài này (đếm trên split), ví dụ để bỏ phiên dài bất thường.
    """
    s = sessions.loc[sessions["split"] == split, ["session_id", "pos", "track_id"]].sort_values(["session_id", "pos"])
    if max_session_length is not None:
        length = s.groupby("session_id")["session_id"].transform("size")
        s = s[length <= max_session_length]
    session = pd.factorize(s["session_id"])[0]
    track = s["track_id"].to_numpy()
    keep = (session[1:] == session[:-1]) & (track[1:] != track[:-1])
    return session[1:][keep], track[:-1][keep], track[1:][keep]


def _pair_values(features: TrackFeatures, prev: np.ndarray, nxt: np.ndarray) -> dict[str, np.ndarray]:
    rows_a, rows_b = features.rows(prev), features.rows(nxt)
    delta = features.bpm_delta(rows_a, rows_b)
    mixable = (delta <= MIXABLE_MAX_BPM_DELTA + BPM_TOLERANCE) & features.compatible_keys(rows_a, rows_b)
    return {"mixable_pct": mixable * 100.0, "bpm_delta_pct": delta * 100.0, "energy_gap": features.energy_gap(rows_a, rows_b)}


def real_vs_random_ci(
    sessions: pd.DataFrame,
    tracks: pd.DataFrame,
    split: str,
    rng: np.random.Generator,
    n_boot: int = 200,
    max_session_length: int | None = None,
) -> pd.DataFrame:
    """Độ mượt của cặp liền nhau thật trong split so với cặp ngẫu nhiên, kèm khoảng tin cậy 95% bootstrap theo phiên.

    Cặp ngẫu nhiên: xáo (hoán vị) các bài kế tiếp của split, mỗi bài đang phát giữ phiên của nó; bỏ cặp rút trúng chính
    bài đang phát. Khác real_vs_random của #24 (rút có hoàn lại): hoán vị giữ đúng số lần mỗi bài làm bài kế tiếp, như
    #57 yêu cầu ("xáo bài kế tiếp trong cùng split"). Phép xáo rút một lần nên khoảng tin cậy không tính dao động do
    xáo; với hàng triệu cặp phần này không đáng kể. Mỗi lần rút, trọng số Poisson(1) cho từng phiên dùng chung cho cặp thật và cặp ngẫu
    nhiên, nên có cả khoảng tin cậy cho chênh lệch. Dòng: "thật", "ngẫu nhiên", "thật − ngẫu nhiên"; cột: mỗi chỉ số kèm
    _low, _high, và n_pairs, n_sessions.
    """
    features = TrackFeatures.from_tracks(tracks)
    session, prev, nxt = _split_pairs(sessions, split, max_session_length)
    if len(prev) == 0:
        raise ValueError(f"split {split} không có cặp liền nhau nào để đo")
    shuffled = nxt[rng.permutation(len(nxt))]
    keep = prev != shuffled
    groups = {REAL: (session, _pair_values(features, prev, nxt)), RANDOM: (session[keep], _pair_values(features, prev[keep], shuffled[keep]))}
    n_sessions = int(session.max()) + 1 if len(session) else 0

    sums = {
        name: (np.bincount(codes, minlength=n_sessions).astype(float), {m: np.bincount(codes, weights=v[m], minlength=n_sessions) for m in E1_METRICS})
        for name, (codes, v) in groups.items()
    }
    draws = {name: {m: np.empty(n_boot) for m in E1_METRICS} for name in groups}
    for b in range(n_boot):
        weight = rng.poisson(1.0, n_sessions).astype(float)
        for name, (count, metric_sums) in sums.items():
            denominator = weight @ count
            for m in E1_METRICS:
                draws[name][m][b] = (weight @ metric_sums[m]) / denominator if denominator else np.nan

    tail = (1 - CONFIDENCE) / 2 * 100
    rows = {}
    for name in (REAL, RANDOM, DIFFERENCE):
        row: dict[str, float] = {}
        for m in E1_METRICS:
            if name == DIFFERENCE:
                estimate = rows[REAL][m] - rows[RANDOM][m]
                samples = draws[REAL][m] - draws[RANDOM][m]
            else:
                count, metric_sums = sums[name]
                estimate = metric_sums[m].sum() / count.sum()
                samples = draws[name][m]
            low, high = np.nanpercentile(samples, [tail, 100 - tail])
            row.update({m: float(estimate), f"{m}_low": float(low), f"{m}_high": float(high)})
        if name != DIFFERENCE:
            row.update({"n_pairs": int(sums[name][0].sum()), "n_sessions": int(np.count_nonzero(sums[name][0]))})
        rows[name] = row
    return pd.DataFrame(rows).T


def load_sessions_for_mix(path: str | Path, target_train_pairs: int, rng: np.random.Generator) -> tuple[pd.DataFrame, dict]:
    """Đọc sessions.parquet của last.fm (119 triệu dòng) cho E1–E3 mà vừa RAM.

    Đọc track_id và split dạng Categorical, giữ mọi phiên val (E1, E2 cần), lấy ngẫu nhiên phiên train đến khi đủ
    ≥ target_train_pairs cặp, bỏ phiên test (không dùng), rồi mới đổi track_id về chuỗi trên phần còn lại.
    Phiên last.fm đã gộp lặp liên tiếp (#55), nên một phiên n bài có đúng n − 1 cặp.
    """
    table = pq.read_table(path, columns=["session_id", "split", "pos", "track_id"], read_dictionary=["split", "track_id"])
    frame = table.to_pandas()
    lengths = frame.groupby("session_id", sort=False)["split"].agg(["size", "first"])
    val_ids = lengths.index[lengths["first"] == VAL]
    train = lengths[lengths["first"] == TRAIN]
    order = rng.permutation(train.index.to_numpy())
    cumulative = (train.loc[order, "size"] - 1).cumsum().to_numpy()
    n_keep = min(int(np.searchsorted(cumulative, target_train_pairs)) + 1, len(cumulative))  # vượt quá: lấy hết train
    train_ids = order[:n_keep]
    keep = frame["session_id"].isin(np.concatenate([val_ids.to_numpy(), train_ids]))
    sessions = frame[keep].reset_index(drop=True)
    sessions = sessions.assign(split=sessions["split"].astype(str), track_id=sessions["track_id"].astype(str))
    info = {"train_sessions": int(n_keep), "train_pairs": int(cumulative[n_keep - 1]) if n_keep else 0, "val_sessions": int(len(val_ids))}
    return sessions, info


def _session_length_p99(sessions: pd.DataFrame, split: str) -> int:
    lengths = sessions.loc[sessions["split"] == split].groupby("session_id").size()
    return int(np.ceil(lengths.quantile(LONG_SESSION_QUANTILE)))


def run_dataset(
    tracks: pd.DataFrame,
    sessions: pd.DataFrame,
    params: MixTrainParams,
    random_seed: int = 0,
    notes: dict | None = None,
    n_boot: int = 200,
    recall_pairs: int = 20_480,
    recall_batch: int = 1024,
    spearman_current: int = 2_000,
    spearman_candidates: int = 100,
) -> dict:
    """E1–E3 trên một bộ dữ liệu (tracks có track_id, bpm, camelot, energy; sessions có split train/val).

    params là MixTrainParams của #23; học trên mọi phiên train được đưa vào (lấy mẫu trước bằng load_sessions_for_mix).
    notes: ghi chú kèm theo kết quả (ví dụ thông tin lấy mẫu), để notebook không phải sửa kết quả.
    """
    from src.rerank.mix_learned_train import train  # nhập muộn: cần torch

    rng = np.random.default_rng(random_seed)
    p99 = _session_length_p99(sessions, VAL)
    e1 = {
        "all": {"table": real_vs_random_ci(sessions, tracks, VAL, rng, n_boot).to_dict(orient="index")},
        "max_p99": {
            "max_session_length": p99,
            "table": real_vs_random_ci(sessions, tracks, VAL, rng, n_boot, max_session_length=p99).to_dict(orient="index"),
        },
    }

    result = train(tracks, sessions, params, random_seed=random_seed)
    scorers = {"learned": LearnedMix(tracks, result.encoders), "rule": RuleMix(tracks)}

    val_pairs = consecutive_pairs(sessions, VAL)
    val_pairs = val_pairs.iloc[rng.permutation(len(val_pairs))[:recall_pairs]]
    recall = {name: in_batch_recall(scorer, val_pairs, batch_size=recall_batch) for name, scorer in scorers.items()}
    e2 = {
        "n_train_pairs": result.n_train_pairs,
        "best_epoch": result.best_epoch,
        "val_loss": result.best_epoch_stats["val_loss"],
        "history": result.history,
        "recall_at_10": {name: r["recall"] for name, r in recall.items()},
        "random_recall_at_10": recall["rule"]["random_recall"],
        "recall_pairs": recall["rule"]["n"],
    }

    n_current = min(spearman_current, len(tracks))  # danh mục nhỏ (dữ liệu giả) thì lấy hết
    n_candidates = min(spearman_candidates, len(tracks) - 1)
    scored = score_pairs(sample_pairs(tracks["track_id"], n_current, n_candidates, rng), tracks, scorers)
    corr = scored[["learned", "rule", "bpm_delta", "camelot_distance", "energy_gap"]].corr(method="spearman")
    e3 = {
        "n_pairs": int(len(scored)),
        "spearman_learned_rule": float(corr.loc["learned", "rule"]),
        "learned_vs_features": {c: float(corr.loc["learned", c]) for c in ("bpm_delta", "camelot_distance", "energy_gap")},
        "rule_vs_features": {c: float(corr.loc["rule", c]) for c in ("bpm_delta", "camelot_distance", "energy_gap")},
    }
    return {"e1": e1, "e2": e2, "e3": e3, "notes": notes or {}, "encoders": result.encoders}


def summary_table(results: dict[str, dict]) -> pd.DataFrame:
    """Một dòng cho mỗi bộ dữ liệu: các con số chính của E1 (mọi phiên), E2, E3."""
    rows = {}
    for name, r in results.items():
        e1 = r["e1"]["all"]["table"]
        capped = r["e1"]["max_p99"]["table"]
        rows[name] = {
            "E1 mix được thật (%)": e1[REAL]["mixable_pct"],
            "E1 mix được ngẫu nhiên (%)": e1[RANDOM]["mixable_pct"],
            "E1 chênh (%)": e1[DIFFERENCE]["mixable_pct"],
            "E1 chênh, CI 95%": f"[{e1[DIFFERENCE]['mixable_pct_low']:.2f}; {e1[DIFFERENCE]['mixable_pct_high']:.2f}]",
            "E1 chênh, bỏ phiên > p99 (%)": capped[DIFFERENCE]["mixable_pct"],
            "E1 ΔBPM thật / ngẫu nhiên (%)": f"{e1[REAL]['bpm_delta_pct']:.2f} / {e1[RANDOM]['bpm_delta_pct']:.2f}",
            "E1 |Δenergy| thật / ngẫu nhiên": f"{e1[REAL]['energy_gap']:.3f} / {e1[RANDOM]['energy_gap']:.3f}",
            "E2 cặp train": r["e2"]["n_train_pairs"],
            "E2 loss val": r["e2"]["val_loss"],
            "E2 Recall@10 học": r["e2"]["recall_at_10"]["learned"],
            "E2 Recall@10 luật": r["e2"]["recall_at_10"]["rule"],
            "E2 Recall@10 ngẫu nhiên": r["e2"]["random_recall_at_10"],
            "E3 Spearman học–luật": r["e3"]["spearman_learned_rule"],
        }
    return pd.DataFrame(rows).T
