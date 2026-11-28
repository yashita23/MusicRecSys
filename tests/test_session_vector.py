import numpy as np
import pandas as pd
import pytest

from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste, session_vector

# Ba bài có vector 2 chiều; X không có vector (không có trong train)
VECTORS = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)  # A, B, C
A, B, C, X = 0, 1, 2, -1


def test_weights_fall_with_distance_to_the_last_track_and_missing_tracks_keep_their_place():
    # Phiên [A, B, X, C]: khoảng cách tới bài cuối C 0, X 1, B 2, A 3; X không đóng góp nhưng vẫn giữ chỗ
    # decay 0,5 → C + 0,25·B + 0,125·A = (1,125, 1,25)
    v = session_vector(VECTORS, np.array([A, B, X, C]), decay=0.5)
    assert v == pytest.approx([1.125, 1.25])


def test_a_repeated_track_counts_at_each_position():
    # Phiên [A, A], decay 0,7 → 0,7·A + A = (1,7, 0)
    assert session_vector(VECTORS, np.array([A, A]), decay=0.7) == pytest.approx([1.7, 0.0])


def test_a_session_without_any_known_track_has_a_zero_vector():
    assert session_vector(VECTORS, np.array([X, X]), decay=0.9).tolist() == [0.0, 0.0]


def tid(name: str) -> str:
    return name * 22


T_A, T_B, T_C, T_D, T_E, T_V = (tid(c) for c in "ABCDEV")  # V không có vector
TRACK_IDS = [T_A, T_B, T_C, T_D, T_E]
TRACK_VECTORS = np.array([[1, 0], [0, 1], [1, 1], [2, 0], [-1, 0]], dtype=np.float32)
# Train cho mô hình dự phòng: E 3 lượt, D 2, C 1
TRAIN = pd.DataFrame(
    [{"session_id": s, "split": "train", "pos": p, "track_id": t} for s, tracks in enumerate([[T_E, T_D, T_C], [T_E, T_D], [T_E]]) for p, t in enumerate(tracks)]
)


@pytest.fixture
def taste():
    return SessionVectorTaste("toy", TRACK_VECTORS, TRACK_IDS, decay=0.5, fallback=Popularity().fit(TRAIN))


def test_candidates_rank_by_dot_product_with_the_session_vector_and_skip_session_tracks(taste):
    # Phiên [B, A]: vector phiên = A + 0,5·B = (1; 0,5). Điểm: A 1, B 0,5 (đều trong phiên, bị loại), C 1,5, D 2, E −1
    ids, scores = taste.candidates([T_B, T_A], k=3)
    assert ids == [T_D, T_C, T_E]
    assert list(scores) == pytest.approx([2.0, 1.5, -1.0])


def test_a_session_without_known_tracks_falls_back_to_the_fallback_model(taste):
    # V không có vector → vector phiên 0 → lấy theo Popularity: E 3 lượt, D 2
    ids, scores = taste.candidates([T_V], k=2)
    assert ids == [T_E, T_D]
    assert list(scores) == [3, 2]


def test_normalized_vectors_rank_by_cosine_and_are_what_gets_exported():
    # Chuẩn hóa: A (1, 0), B (0, 1), C (0,707; 0,707), D (1, 0), E (−1, 0). Phiên [B, A], decay 0,5 → vector phiên (1; 0,5)
    # Điểm: C 0,707 + 0,354 = 1,061, D 1, E −1. Không chuẩn hóa thì D (2) đứng trước C (1,5)
    taste = SessionVectorTaste("toy", TRACK_VECTORS, TRACK_IDS, decay=0.5, fallback=Popularity().fit(TRAIN), normalize=True)
    ids, scores = taste.candidates([T_B, T_A], k=3)
    assert ids == [T_C, T_D, T_E]
    assert list(scores) == pytest.approx([1.0607, 1.0, -1.0], abs=1e-4)
    vectors, _ = taste.embeddings()
    assert np.linalg.norm(vectors, axis=1) == pytest.approx(np.ones(5))


def test_batch_candidates_match_scoring_each_session_alone():
    # Chấm theo lô (grid search #25) phải ra đúng như từng phiên: loại bài trong phiên, phiên không bài nào có vector
    # thì dùng mô hình dự phòng, phiên dài ngắn khác nhau.
    from src.data.fake import FAKE_MODEL, make_fake_dataset

    fake = make_fake_dataset(n_sessions=120, n_tracks=300, random_seed=7)
    vectors, index = fake.embeddings[FAKE_MODEL]
    taste = SessionVectorTaste(FAKE_MODEL, vectors, index["track_id"].tolist(), 0.7, Popularity().fit(fake.sessions))
    known = set(index["track_id"])
    unknown = [t for t in fake.tracks["track_id"] if t not in known][:2]
    sessions = [list(s) for s in fake.eval_targets["seed_track_ids"][:30]]
    sessions += [unknown, sessions[0] + sessions[1], [sessions[2][0]] * 3]
    batch = taste.candidates_batch(sessions, k=20)
    assert len(batch) == len(sessions)
    for session, (ids, scores) in zip(sessions, batch):
        alone_ids, alone_scores = taste.candidates(session, k=20)
        assert ids == alone_ids
        np.testing.assert_allclose(scores, alone_scores, rtol=1e-5, atol=1e-6)
