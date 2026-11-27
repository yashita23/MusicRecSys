"""So sánh các điểm mix trên cùng các cặp bài (ticket #24, spec §6 và §13 Tranh luận).

Câu hỏi: điểm mix học từ các cặp liền nhau trong playlist (#23) có "học lại" được luật DJ (#12) không, và cặp
liền nhau thật có mượt hơn cặp ngẫu nhiên không. Mọi hàm chỉ dùng numpy/pandas; notebook 12 chỉ gọi các hàm này.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict

import numpy as np
import pandas as pd

from src.eval.run import K
from src.eval.smoothness import smoothness
from src.features.track_features import TrackFeatures
from src.rerank.combine import MixScorer
from src.rerank.mix_learned import consecutive_pairs

BPM_BANDS = (0.0, 0.03, 0.06, 0.09, 0.12, 0.20, np.inf)  # khoảng phải đóng: lệch đúng 6% vẫn ở ô "3–6%" (mix được)
BPM_BAND_LABELS = ("≤ 3%", "3–6%", "6–9%", "9–12%", "12–20%", "> 20%")
MAX_CAMELOT_DISTANCE = 7


def sample_pairs(track_ids: Sequence[str], n_current: int, n_candidates: int, rng: np.random.Generator) -> pd.DataFrame:
    """n_current bài đang phát ngẫu nhiên, mỗi bài ghép với n_candidates bài khác nhau (khác chính nó)."""
    ids = np.asarray(track_ids)
    rows = []
    for current in rng.choice(len(ids), n_current, replace=False):
        drawn = rng.choice(len(ids), n_candidates + 1, replace=False)
        drawn = drawn[drawn != current][:n_candidates]
        rows.append(pd.DataFrame({"prev_track_id": ids[current], "next_track_id": ids[drawn]}))
    return pd.concat(rows, ignore_index=True)


def score_pairs(pairs: pd.DataFrame, tracks: pd.DataFrame, scorers: Mapping[str, MixScorer]) -> pd.DataFrame:
    """Mỗi cặp (prev_track_id → next_track_id) kèm ΔBPM, khoảng cách Camelot, |Δenergy| và một cột cho mỗi hàm mix.

    Hàm mix được gọi một lần cho mỗi bài đang phát (cùng giao diện với RuleMix), giữ đúng thứ tự dòng đầu vào.
    """
    features = TrackFeatures.from_tracks(tracks)
    prev, nxt = pairs["prev_track_id"].to_numpy(), pairs["next_track_id"].to_numpy()
    rows_a, rows_b = features.rows(prev.tolist()), features.rows(nxt.tolist())
    out = pairs[["prev_track_id", "next_track_id"]].reset_index(drop=True).assign(
        bpm_delta=features.bpm_delta(rows_a, rows_b),
        camelot_distance=features.camelot_distance(rows_a, rows_b),
        energy_gap=features.energy_gap(rows_a, rows_b),
    )
    groups = pd.Series(prev).groupby(prev, sort=False).indices
    for name, scorer in scorers.items():
        scores = np.empty(len(out))
        for current, idx in groups.items():
            scores[idx] = scorer(current, nxt[idx].tolist())
        out[name] = scores
    return out


def in_batch_recall(scorer: MixScorer, pairs: pd.DataFrame, batch_size: int = 1024, k: int = K) -> dict[str, float]:
    """Recall@k của bài kế tiếp thật trong batch, cho một hàm mix bất kỳ; cùng định nghĩa với lúc huấn luyện #23.

    pairs chia thành các batch liên tiếp theo đúng thứ tự dòng. Với mỗi cặp (a, b⁺), ứng viên là mọi bài kế tiếp
    trong batch trừ những bài trùng b⁺ (không phải mẫu âm); hòa điểm tính là thua. random_recall là mức ngẫu nhiên
    min(k, số ứng viên) / số ứng viên, trung bình trên các cặp.
    """
    prev, nxt = pairs["prev_track_id"].to_numpy(), pairs["next_track_id"].to_numpy()
    hits = random = 0.0
    for start in range(0, len(prev), batch_size):
        batch_prev, batch_next = prev[start : start + batch_size], nxt[start : start + batch_size]
        candidates = batch_next.tolist()
        for i, current in enumerate(batch_prev):
            scores = np.asarray(scorer(current, candidates), dtype=float)
            others = batch_next != batch_next[i]  # bỏ chính b⁺ và các bản trùng của nó
            hits += int((scores[others] >= scores[i]).sum() < k)
            n_candidates = int(others.sum()) + 1
            random += min(k, n_candidates) / n_candidates
    n = len(prev)
    return {"recall": hits / n, "random_recall": random / n, "n": n}


def real_vs_random(sessions: pd.DataFrame, tracks: pd.DataFrame, split: str, rng: np.random.Generator, repeats: int = 1) -> pd.DataFrame:
    """Độ mượt của các cặp liền nhau thật trong split, so với cùng các bài đang phát ghép với bài kế tiếp ngẫu nhiên.

    Bài kế tiếp ngẫu nhiên rút có hoàn lại từ chính các bài kế tiếp của split, nên giữ nguyên độ phổ biến của bài;
    mỗi cặp thật cho repeats cặp ngẫu nhiên. Dòng "thật" và "ngẫu nhiên", cột như Smoothness (#13).
    """
    pairs = consecutive_pairs(sessions, split)
    features = TrackFeatures.from_tracks(tracks)
    prev, nxt = pairs["prev_track_id"].to_numpy(), pairs["next_track_id"].to_numpy()
    random_next = nxt[rng.integers(0, len(nxt), len(nxt) * repeats)]
    rows = {
        "thật": smoothness([[a, b] for a, b in zip(prev, nxt)], features),
        "ngẫu nhiên": smoothness([[a, b] for a, b in zip(np.repeat(prev, repeats), random_next)], features),
    }
    return pd.DataFrame({name: asdict(s) for name, s in rows.items()}).T


def heatmap_table(scored: pd.DataFrame, value: str, aggfunc: str = "mean") -> pd.DataFrame:
    """Gộp cột value (mặc định trung bình; "count" để đếm cặp) theo khoảng cách Camelot (dòng, 0–7) × dải ΔBPM (cột).

    Ô không có cặp nào là NaN.
    """
    band = pd.cut(scored["bpm_delta"], BPM_BANDS, labels=BPM_BAND_LABELS, right=True, include_lowest=True)
    table = scored.assign(bpm_band=band).pivot_table(index="camelot_distance", columns="bpm_band", values=value, aggfunc=aggfunc, observed=False)
    return table.reindex(index=range(MAX_CAMELOT_DISTANCE + 1), columns=list(BPM_BAND_LABELS))
