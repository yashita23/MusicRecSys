"""Kiểm tra liên bảng: các file bàn giao phải khớp với nhau, không chỉ đúng từng file."""
import numpy as np
import pandas as pd
import pytest

from src.data.schema import SchemaError, validate_dataset

T = [f"{i:022d}" for i in range(12)]  # 12 bài; Spotify ID giả 22 ký tự


def small_dataset():
    """Phiên 1 (train): T0…T9. Phiên 2 (val): T0…T10, seed T0…T4, bài đích T5."""
    train = T[:10]
    val = T[:11]
    sessions = pd.DataFrame(
        [{"session_id": 1, "split": "train", "pos": p, "track_id": t} for p, t in enumerate(train)]
        + [{"session_id": 2, "split": "val", "pos": p, "track_id": t} for p, t in enumerate(val)]
    )
    tracks = pd.DataFrame(
        {
            "track_id": T,
            "track_name": [f"Song {i}" for i in range(12)],
            "artist_name": ["X"] * 12,
            "tempo": [120.0] * 12,
            "bpm": [120.0] * 12,
            "key": [9] * 12,
            "mode": [0] * 12,
            "camelot": ["8A"] * 12,
            "energy": [0.5] * 12,
            "n_occurrences": [1] * 10 + [0, 0],  # chỉ đếm trong train
        }
    )
    eval_targets = pd.DataFrame(
        {"session_id": [2], "split": ["val"], "seed_track_ids": [T[:5]], "target_track_id": [T[5]]}
    )
    candidates = pd.DataFrame(
        {"session_id": [2, 2, 2], "rank": [1, 2, 3], "track_id": [T[5], T[6], T[11]], "taste_score": [3.0, 2.0, 1.0]}
    )
    return tracks, sessions, eval_targets, candidates


def check(tracks, sessions, eval_targets, candidates, embeddings=None):
    validate_dataset(tracks, sessions, eval_targets, candidates={"val": candidates}, embeddings=embeddings, k=3)


def test_consistent_dataset_passes():
    check(*small_dataset())


def test_session_track_missing_from_tracks_is_reported():
    tracks, sessions, eval_targets, candidates = small_dataset()
    with pytest.raises(SchemaError, match="tracks"):
        check(tracks[tracks["track_id"] != T[3]], sessions, eval_targets, candidates)


def test_seed_must_be_first_five_tracks_of_session():
    tracks, sessions, eval_targets, candidates = small_dataset()
    eval_targets.at[0, "seed_track_ids"] = [T[0], T[1], T[2], T[3], T[9]]
    with pytest.raises(SchemaError, match="seed"):
        check(tracks, sessions, eval_targets, candidates)


def test_target_must_be_sixth_track_of_session():
    tracks, sessions, eval_targets, candidates = small_dataset()
    eval_targets.at[0, "target_track_id"] = T[7]
    with pytest.raises(SchemaError, match="bài đích"):
        check(tracks, sessions, eval_targets, candidates)


def test_eval_target_split_must_match_session_split():
    tracks, sessions, eval_targets, candidates = small_dataset()
    eval_targets.at[0, "split"] = "test"
    with pytest.raises(SchemaError, match="split"):
        validate_dataset(tracks, sessions, eval_targets, candidates={"test": candidates}, k=3)


def test_candidates_must_not_contain_seed_tracks():
    tracks, sessions, eval_targets, candidates = small_dataset()
    candidates.at[2, "track_id"] = T[0]
    with pytest.raises(SchemaError, match="seed"):
        check(tracks, sessions, eval_targets, candidates)


def test_candidates_session_must_be_in_eval_targets_of_same_split():
    tracks, sessions, eval_targets, candidates = small_dataset()
    candidates["session_id"] = 1  # phiên train
    with pytest.raises(SchemaError, match="eval_targets"):
        check(tracks, sessions, eval_targets, candidates)


def test_n_occurrences_must_count_train_sessions_only():
    tracks, sessions, eval_targets, candidates = small_dataset()
    tracks.loc[tracks["track_id"] == T[10], "n_occurrences"] = 1  # T10 chỉ có trong phiên val
    with pytest.raises(SchemaError, match="n_occurrences"):
        check(tracks, sessions, eval_targets, candidates)


def add_val_session(tracks, sessions, session_id, track_ids):
    """Thêm một phiên val; giữ n_occurrences đúng vì phiên val không được tính."""
    extra = pd.DataFrame([{"session_id": session_id, "split": "val", "pos": p, "track_id": t} for p, t in enumerate(track_ids)])
    return tracks, pd.concat([sessions, extra], ignore_index=True)


def test_eligible_val_session_missing_from_eval_targets_is_reported():
    tracks, sessions, eval_targets, candidates = small_dataset()
    tracks, sessions = add_val_session(tracks, sessions, 3, T[1:11])  # bài đích T6 ∉ seed → phải có trong eval_targets
    with pytest.raises(SchemaError, match="eval_targets thiếu"):
        check(tracks, sessions, eval_targets, candidates)


def test_val_session_whose_target_is_in_seed_may_be_left_out():
    tracks, sessions, eval_targets, candidates = small_dataset()
    tracks, sessions = add_val_session(tracks, sessions, 3, T[:5] + [T[0]] + T[5:9])  # bài đích T0 nằm trong seed
    check(tracks, sessions, eval_targets, candidates)


def test_every_eval_session_needs_candidates():
    tracks, sessions, eval_targets, candidates = small_dataset()
    with pytest.raises(SchemaError, match="không có ứng viên"):
        check(tracks, sessions, eval_targets, candidates.iloc[0:0])


def test_embeddings_only_hold_tracks_learned_from_train():
    tracks, sessions, eval_targets, candidates = small_dataset()
    index = pd.DataFrame({"row": [0, 1], "track_id": [T[0], T[10]]})  # T10 không có trong train
    vectors = np.ones((2, 4), dtype=np.float32)
    with pytest.raises(SchemaError, match="train"):
        check(tracks, sessions, eval_targets, candidates, embeddings={"als": (vectors, index)})


def test_embedding_tracks_must_exist_in_tracks():
    tracks, sessions, eval_targets, candidates = small_dataset()
    index = pd.DataFrame({"row": [0, 1], "track_id": [T[0], "z" * 22]})
    vectors = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(SchemaError, match="als"):
        check(tracks, sessions, eval_targets, candidates, embeddings={"als": (vectors, index)})
