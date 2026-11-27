import numpy as np
import pandas as pd
import pytest

from src.rerank.combine import rerank
from src.rerank.mix_learned import FEATURE_NAMES, LearnedMix, MixEncoders, consecutive_pairs


def tid(name: str) -> str:
    return name.ljust(22, "x")


A, B, C, D, V, W = (tid(c) for c in "ABCDVW")


def sessions_frame(playlists: dict[int, tuple[str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"session_id": sid, "split": split, "pos": p, "track_id": t} for sid, (split, tracks) in playlists.items() for p, t in enumerate(tracks)]
    )


def test_train_pairs_are_consecutive_tracks_of_train_playlists_only():
    frame = sessions_frame({1: ("train", [A, B, C]), 2: ("val", [V, W, A]), 3: ("train", [D, A]), 4: ("test", [W, V])})
    pairs = consecutive_pairs(frame.sample(frac=1, random_state=0), split="train")  # thứ tự dòng không quan trọng
    assert list(zip(pairs["prev_track_id"], pairs["next_track_id"])) == [(A, B), (B, C), (D, A)]


def test_pairs_never_cross_from_one_playlist_into_the_next():
    frame = sessions_frame({1: ("train", [A, B]), 2: ("train", [C, D])})
    pairs = consecutive_pairs(frame, split="train")
    assert (B, C) not in set(zip(pairs["prev_track_id"], pairs["next_track_id"]))
    assert len(pairs) == 2


def test_a_track_repeated_back_to_back_is_not_a_transition():
    frame = sessions_frame({1: ("train", [A, A, B])})
    pairs = consecutive_pairs(frame, split="train")
    assert list(zip(pairs["prev_track_id"], pairs["next_track_id"])) == [(A, B)]


def test_val_pairs_come_from_val_playlists():
    frame = sessions_frame({1: ("train", [A, B]), 2: ("val", [V, W, A])})
    pairs = consecutive_pairs(frame, split="val")
    assert list(zip(pairs["prev_track_id"], pairs["next_track_id"])) == [(V, W), (W, A)]


# ---------- LearnedMix: điểm mix học bằng numpy, cắm vào rerank ----------


def catalog() -> pd.DataFrame:
    # Camelot 8A, 11A (cách 3 bước = 1/4 vòng), 2A (cách 6 bước = nửa vòng); BPM và energy khác nhau tùy ý
    return pd.DataFrame(
        {"track_id": [A, B, C], "bpm": [120.0, 95.0, 170.0], "camelot": ["8A", "11A", "2A"], "energy": [0.2, 0.9, 0.5]}
    )


def key_angle_encoders() -> MixEncoders:
    """f_out = f_in = (cos, sin) của góc trên vòng Camelot: s_learn là cos của góc lệch giữa hai tông."""
    w = np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32)
    w[0, FEATURE_NAMES.index("key_cos")] = 1.0
    w[1, FEATURE_NAMES.index("key_sin")] = 1.0
    layer = [(w, np.zeros(2, dtype=np.float32))]
    return MixEncoders(f_out=layer, f_in=layer, temperature=0.1)


def test_learned_mix_maps_s_learn_from_minus_one_one_onto_zero_one_in_candidate_order():
    mix = LearnedMix(catalog(), key_angle_encoders())
    # cùng tông: cos 0 = 1 → 1; lệch 1/4 vòng: cos 90° = 0 → 0,5; lệch nửa vòng: cos 180° = −1 → 0
    np.testing.assert_allclose(mix(A, [A, B, C]), [1.0, 0.5, 0.0], atol=1e-6)
    np.testing.assert_allclose(mix(A, [C, A]), [0.0, 1.0], atol=1e-6)


def test_learned_mix_unknown_track_is_an_error():
    with pytest.raises(KeyError, match="z" * 22):
        LearnedMix(catalog(), key_angle_encoders())(A, [B, "z" * 22])


def test_learned_mix_plugs_into_rerank():
    # α = 0: chỉ còn điểm mix, nên bài cùng tông lên đầu dù điểm gu thấp nhất
    out = rerank([C, B, A], [3.0, 2.0, 1.0], current_track_id=A, mix=LearnedMix(catalog(), key_angle_encoders()), alpha=0.0)
    assert out["track_id"].tolist() == [A, B, C]


def test_encoders_saved_to_npz_score_the_same_after_loading(tmp_path):
    path = tmp_path / "mix_learned.npz"
    key_angle_encoders().save(path)
    loaded = LearnedMix.load(catalog(), path)
    np.testing.assert_allclose(loaded(A, [A, B, C]), [1.0, 0.5, 0.0], atol=1e-6)
    assert loaded.encoders.temperature == pytest.approx(0.1)
