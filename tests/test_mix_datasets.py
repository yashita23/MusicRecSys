"""Thí nghiệm điểm mix trên hai bộ dữ liệu (ticket #57): E1 cặp thật so với ngẫu nhiên, có khoảng tin cậy theo phiên."""
import numpy as np
import pandas as pd
import pytest

from src.rerank.mix_datasets import real_vs_random_ci


def tid(name: str) -> str:
    return name.ljust(16, "x")


A, B, C, D = (tid(c) for c in "ABCD")
# A, B, C: 120 BPM, tông 8A, energy 0,5 → mọi cặp giữa chúng mix được, |Δenergy| 0.
# D: 150 BPM (lệch 25%), tông 2B, energy 0,9 → cặp có D không mix được, |Δenergy| 0,4.
TRACKS = pd.DataFrame(
    {
        "track_id": [A, B, C, D],
        "bpm": [120.0, 120.0, 120.0, 150.0],
        "camelot": ["8A", "8A", "8A", "2B"],
        "energy": [0.5, 0.5, 0.5, 0.9],
    }
)


def sessions_frame(playlists: list[list[str]], split: str = "val") -> pd.DataFrame:
    return pd.DataFrame(
        [{"session_id": sid, "split": split, "pos": p, "track_id": t} for sid, tracks in enumerate(playlists) for p, t in enumerate(tracks)]
    )


# 4 cặp thật trong 2 phiên: A→B, B→C (mix được), C→D, D→A (không) → 50% mix được, |Δenergy| trung bình 0,2
SESSIONS = sessions_frame([[A, B, C], [C, D, A]])


def test_the_real_pair_estimate_is_the_plain_pair_average():
    table = real_vs_random_ci(SESSIONS, TRACKS, "val", np.random.default_rng(0), n_boot=200)
    real = table.loc["thật"]
    assert real["mixable_pct"] == pytest.approx(50.0)
    assert real["energy_gap"] == pytest.approx(0.2)
    assert real["n_pairs"] == 4 and real["n_sessions"] == 2


def test_confidence_intervals_contain_the_estimate_and_shrink_with_more_sessions():
    small = real_vs_random_ci(SESSIONS, TRACKS, "val", np.random.default_rng(1), n_boot=300)
    many = real_vs_random_ci(sessions_frame([[A, B, C], [C, D, A]] * 50), TRACKS, "val", np.random.default_rng(1), n_boot=300)
    for table in (small, many):
        for row in ("thật", "ngẫu nhiên"):
            assert table.loc[row, "mixable_pct_low"] <= table.loc[row, "mixable_pct"] <= table.loc[row, "mixable_pct_high"]
    width = lambda t: t.loc["thật", "mixable_pct_high"] - t.loc["thật", "mixable_pct_low"]  # noqa: E731
    assert width(many) < width(small)


def test_random_pairs_shuffle_next_tracks_within_the_split_and_never_pair_a_track_with_itself():
    # Bài kế tiếp của split chỉ có A, B, C (cùng đặc trưng): xáo kiểu gì cũng mix được 100%, và không có cặp X → X
    table = real_vs_random_ci(sessions_frame([[A, B, C], [B, C, A], [C, A, B]]), TRACKS, "val", np.random.default_rng(2), n_boot=50)
    assert table.loc["ngẫu nhiên", "mixable_pct"] == pytest.approx(100.0)
    assert table.loc["ngẫu nhiên", "n_pairs"] <= table.loc["thật", "n_pairs"]


def test_only_the_requested_split_is_used_and_long_sessions_can_be_left_out():
    sessions = pd.concat([SESSIONS, sessions_frame([[D, A, D, A, D]], split="train").assign(session_id=9)])
    table = real_vs_random_ci(sessions, TRACKS, "val", np.random.default_rng(0), n_boot=50)
    assert table.loc["thật", "n_pairs"] == 4  # phiên train không được đếm
    # Phiên val thứ ba dài 6 bài, toàn cặp mix được; bỏ phiên > 3 bài thì về lại đúng 50%
    longer = pd.concat([SESSIONS, sessions_frame([[A, B, A, B, A, B]]).assign(session_id=5)])
    assert real_vs_random_ci(longer, TRACKS, "val", np.random.default_rng(0), n_boot=50).loc["thật", "mixable_pct"] > 50
    capped = real_vs_random_ci(longer, TRACKS, "val", np.random.default_rng(0), n_boot=50, max_session_length=3)
    assert capped.loc["thật", "mixable_pct"] == pytest.approx(50.0) and capped.loc["thật", "n_sessions"] == 2


def test_sampling_train_sessions_stops_near_the_target_pair_count_and_keeps_other_splits():
    from src.rerank.mix_datasets import sample_train_sessions

    train = sessions_frame([[A, B, C, D]] * 30, split="train")  # 30 phiên train, mỗi phiên 3 cặp
    val = sessions_frame([[A, B]], split="val").assign(session_id=100)
    sampled, n_pairs = sample_train_sessions(pd.concat([train, val]), target_pairs=20, rng=np.random.default_rng(0))
    assert n_pairs == 21 and sampled.loc[sampled["split"] == "train", "session_id"].nunique() == 7  # 7 phiên × 3 ≥ 20
    assert (sampled["split"] == "val").sum() == 2  # split khác giữ nguyên


@pytest.fixture(scope="module")
def fake():
    from src.data.fake import make_fake_dataset

    return make_fake_dataset(n_sessions=200, n_tracks=400, random_seed=4)


def test_one_dataset_run_reports_e1_e2_e3_and_learns_only_from_train_pairs(fake):
    pytest.importorskip("torch")
    from src.rerank.mix_datasets import run_dataset, summary_table
    from src.rerank.mix_learned import consecutive_pairs
    from src.rerank.mix_learned_train import MixTrainParams

    params = MixTrainParams(hidden=(8,), dim=4, batch_size=32, max_epochs=2, max_val_pairs=200)
    result = run_dataset(fake.tracks, fake.sessions, params, random_seed=0, n_boot=30, recall_pairs=128, recall_batch=32, spearman_current=20, spearman_candidates=10)

    assert set(result["e1"]) == {"all", "max_p99"} and result["e1"]["max_p99"]["max_session_length"] > 0
    assert set(result["e1"]["all"]["table"]) == {"thật", "ngẫu nhiên", "thật − ngẫu nhiên"}
    assert result["e2"]["n_train_pairs"] == len(consecutive_pairs(fake.sessions, "train"))  # chỉ cặp train
    assert set(result["e2"]["recall_at_10"]) == {"learned", "rule"}
    assert -1 <= result["e3"]["spearman_learned_rule"] <= 1

    table = summary_table({"MPD": result, "last.fm": result})
    assert table.index.tolist() == ["MPD", "last.fm"]
    for column in ("E1 mix được thật (%)", "E1 mix được ngẫu nhiên (%)", "E2 Recall@10 học", "E2 Recall@10 luật", "E3 Spearman học–luật"):
        assert column in table.columns


def test_loading_lastfm_for_the_mix_runs_keeps_val_samples_train_and_drops_test(tmp_path):
    from src.rerank.mix_datasets import load_sessions_for_mix

    parts = [
        sessions_frame([[A, B, C, D]] * 30, split="train"),                          # 30 phiên × 3 cặp
        sessions_frame([[A, B, C]] * 4, split="val").assign(session_id=lambda d: d["session_id"] + 100),
        sessions_frame([[A, B]] * 3, split="test").assign(session_id=lambda d: d["session_id"] + 200),
    ]
    pd.concat(parts).assign(user_id=1)[["session_id", "user_id", "split", "pos", "track_id"]].to_parquet(tmp_path / "sessions.parquet", index=False)
    sessions, info = load_sessions_for_mix(tmp_path / "sessions.parquet", target_train_pairs=20, rng=np.random.default_rng(0))

    assert set(sessions["split"]) == {"train", "val"}
    assert sessions.loc[sessions["split"] == "val", "session_id"].nunique() == 4
    assert info == {"train_sessions": 7, "train_pairs": 21, "val_sessions": 4}  # 7 × 3 ≥ 20
    assert sessions["track_id"].map(type).eq(str).all()  # đã đổi về chuỗi cho TrackFeatures
