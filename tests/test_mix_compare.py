import numpy as np
import pandas as pd
import pytest

from src.rerank.mix_compare import heatmap_table, in_batch_recall, real_vs_random, sample_pairs, score_pairs
from src.rerank.mix_rule import RuleMix

A, B, C = "a" * 22, "b" * 22, "c" * 22


def tracks() -> pd.DataFrame:
    return pd.DataFrame(
        {"track_id": [A, B, C], "bpm": [120.0, 128.0, 60.0], "camelot": ["8A", "8B", "3A"], "energy": [0.7, 0.5, 0.7]}
    )


def pairs(*ab: tuple[str, str]) -> pd.DataFrame:
    return pd.DataFrame(ab, columns=["prev_track_id", "next_track_id"])


def test_score_pairs_puts_each_scorer_next_to_the_pair_properties():
    scored = score_pairs(pairs((A, B), (A, C), (B, A)), tracks(), {"rule": RuleMix(tracks())})
    assert scored[["prev_track_id", "next_track_id"]].values.tolist() == [[A, B], [A, C], [B, A]]
    # A → B: lệch 6,67%, 8A → 8B (1 bước), |Δenergy| 0,2; A → C: nhịp một nửa (0%), 8A → 3A (5 bước), 0
    # B → A: lệch |120 − 128| / 128 = 6,25%, s_bpm (12 − 6,25) / 6 = 0,9583 → 0,4·0,9583 + 0,4·0,8 + 0,2·0,8
    np.testing.assert_allclose(scored["bpm_delta"], [0.06667, 0.0, 0.0625], atol=1e-4)
    assert scored["camelot_distance"].tolist() == [1, 5, 1]
    np.testing.assert_allclose(scored["energy_gap"], [0.2, 0.0, 0.2], atol=1e-9)
    np.testing.assert_allclose(scored["rule"], [0.83556, 0.6, 0.86333], atol=1e-4)


def test_score_pairs_with_several_scorers_gives_one_column_each():
    constant = lambda current, candidates: np.full(len(candidates), 0.5)  # noqa: E731
    scored = score_pairs(pairs((A, B), (B, C)), tracks(), {"rule": RuleMix(tracks()), "half": constant})
    assert scored["half"].tolist() == [0.5, 0.5]


def test_heatmap_averages_a_score_per_camelot_distance_and_bpm_band():
    scored = score_pairs(pairs((A, B), (A, C), (B, A)), tracks(), {"rule": RuleMix(tracks())})
    table = heatmap_table(scored, "rule")
    # 6,67% và 6,25% cùng ô "6–9%" ở khoảng cách 1: (0,83556 + 0,86333) / 2; lệch 0% ở ô "≤ 3%", khoảng cách 5
    assert table.loc[1, "6–9%"] == pytest.approx(0.84944, abs=1e-4)
    assert table.loc[5, "≤ 3%"] == pytest.approx(0.6)
    assert list(table.index) == list(range(8))  # đủ khoảng cách 0–7, ô trống là NaN
    assert np.isnan(table.loc[0, "≤ 3%"])


def test_heatmap_can_count_pairs_per_cell():
    scored = score_pairs(pairs((A, B), (A, C), (B, A)), tracks(), {"rule": RuleMix(tracks())})
    counts = heatmap_table(scored, "rule", aggfunc="count")
    assert counts.loc[1, "6–9%"] == 2 and counts.loc[5, "≤ 3%"] == 1


def test_exactly_six_percent_falls_in_the_mixable_band():
    t = pd.DataFrame({"track_id": [A, B], "bpm": [100.0, 106.0], "camelot": ["8A", "8A"], "energy": [0.5, 0.5]})
    table = heatmap_table(score_pairs(pairs((A, B)), t, {"rule": RuleMix(t)}), "rule")
    assert table.loc[0, "3–6%"] == pytest.approx(1.0)


def test_sample_pairs_draws_distinct_tracks_for_each_current_track():
    ids = [f"{i:022d}" for i in range(50)]
    sampled = sample_pairs(ids, n_current=5, n_candidates=10, rng=np.random.default_rng(0))
    assert len(sampled) == 50
    assert sampled["prev_track_id"].nunique() == 5
    assert (sampled["prev_track_id"] != sampled["next_track_id"]).all()
    assert not sampled.duplicated().any()


# ---------- Recall@k trong batch cho một hàm chấm bất kỳ ----------

P1, P2, P3, P4, N1, N2, N3 = (f"{c}".ljust(22, "x") for c in ("p1", "p2", "p3", "p4", "n1", "n2", "n3"))
TABLE = {  # điểm của (bài đang phát, ứng viên)
    P1: {N1: 0.9, N2: 0.1, N3: 0.2},   # bài kế tiếp thật N1 đứng đầu
    P2: {N1: 0.8, N2: 0.3, N3: 0.5},   # N2 đứng thứ 3
    P3: {N1: 0.4, N2: 0.1, N3: 0.4},   # N3 hòa với N1: hòa tính là thua
    P4: {N1: 0.2, N2: 0.9, N3: 0.1},   # bài kế tiếp thật cũng là N3 như P3
}


def lookup(current: str, candidates: list[str]) -> np.ndarray:
    return np.array([TABLE[current][c] for c in candidates])


def test_in_batch_recall_ranks_each_true_next_track_against_the_rest_of_the_batch():
    batch = pairs((P1, N1), (P2, N2), (P3, N3), (P4, N3))
    stats = in_batch_recall(lookup, batch, batch_size=4, k=1)
    # P1: N1 đứng đầu → trúng. P2: N1 và hai cột N3 cao hơn → trượt. P3: N1 hòa → trượt. P4: N2 cao hơn → trượt.
    # Ứng viên: P1, P2 so với cả 4 cột; P3, P4 có bài kế tiếp N3 nên cột N3 của dòng kia không phải mẫu âm → 3.
    # Mức ngẫu nhiên với k = 1: (1/4 + 1/4 + 1/3 + 1/3) / 4 = 7/24
    assert stats["recall"] == pytest.approx(1 / 4)
    assert stats["random_recall"] == pytest.approx(7 / 24)
    assert stats["n"] == 4


def test_in_batch_recall_with_k_covering_the_batch_always_hits():
    stats = in_batch_recall(lookup, pairs((P1, N1), (P2, N2), (P3, N3)), batch_size=3, k=10)
    assert stats["recall"] == 1.0 and stats["random_recall"] == 1.0


def test_in_batch_recall_of_learned_mix_matches_the_metric_used_while_training():
    pytest.importorskip("torch", reason="cần PyTorch để huấn luyện")
    from src.rerank.mix_learned import LearnedMix, consecutive_pairs
    from src.rerank.mix_learned_train import MixTrainParams, train
    from tests.test_mix_learned_train import smooth_playlists

    tracks_, sessions = smooth_playlists()
    params = MixTrainParams(hidden=(32,), dim=16, batch_size=2048, max_epochs=3, lr=3e-3)
    val = consecutive_pairs(sessions, "val")
    assert len(val) < params.batch_size  # một batch duy nhất: thứ tự trộn không đổi được tập so sánh
    result = train(tracks_, sessions, params, random_seed=0)
    stats = in_batch_recall(LearnedMix(tracks_, result.encoders), val, batch_size=params.batch_size)
    best = result.best_epoch_stats
    assert stats["recall"] == pytest.approx(best["val_recall_at_10"], abs=2 / len(val))
    assert stats["random_recall"] == pytest.approx(best["random_recall_at_10"], abs=1e-9)


# ---------- cặp liền nhau thật so với cặp ngẫu nhiên ----------

X1, X2, Y1, Y2, V1 = (f"{c}".ljust(22, "x") for c in ("x1", "x2", "y1", "y2", "v1"))


def two_scene_catalog() -> pd.DataFrame:
    # nhóm X: 120 BPM, 8A; nhóm Y: 150 BPM, 2A. Trong nhóm thì mix được, khác nhóm thì không.
    return pd.DataFrame(
        {"track_id": [X1, X2, Y1, Y2, V1], "bpm": [120.0, 121.0, 150.0, 151.0, 120.0],
         "camelot": ["8A", "8A", "2A", "2A", "8A"], "energy": [0.5, 0.6, 0.5, 0.6, 0.5]}
    )


def scene_sessions() -> pd.DataFrame:
    playlists = {1: ("train", [X1, X2, X1, X2]), 2: ("train", [Y1, Y2, Y1, Y2]), 3: ("val", [V1, Y1])}
    return pd.DataFrame(
        [{"session_id": s, "split": sp, "pos": p, "track_id": t} for s, (sp, ts) in playlists.items() for p, t in enumerate(ts)]
    )


def test_real_consecutive_pairs_are_compared_with_the_same_pairs_given_a_random_next_track():
    table = real_vs_random(scene_sessions(), two_scene_catalog(), split="train", rng=np.random.default_rng(0), repeats=400)
    assert table.loc["thật", "n_pairs"] == 6            # chỉ phiên train: 3 cặp mỗi phiên
    assert table.loc["thật", "mixable_pct"] == pytest.approx(100.0)
    # bài kế tiếp rút từ chính các bài kế tiếp của split: một nửa thuộc nhóm còn lại → khoảng 50%
    assert table.loc["ngẫu nhiên", "n_pairs"] == 6 * 400
    assert 40 < table.loc["ngẫu nhiên", "mixable_pct"] < 60


def test_val_split_only_uses_val_playlists():
    table = real_vs_random(scene_sessions(), two_scene_catalog(), split="val", rng=np.random.default_rng(0))
    assert table.loc["thật", "n_pairs"] == 1
    assert table.loc["thật", "mixable_pct"] == pytest.approx(0.0)  # 8A, 120 → 2A, 150
