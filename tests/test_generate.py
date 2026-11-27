from collections.abc import Sequence

import numpy as np
import pytest

from src.data.fake import make_fake_dataset
from src.models.item_knn import ItemKNN
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste
from src.rerank.generate import generate_chains
from src.rerank.mix_rule import RuleMix

SEED = [f"s{i}" for i in range(5)]
CATALOG = [f"t{i}" for i in range(40)]


class RankedTaste:
    """Mô hình gu giả: luôn thích t0 > t1 > t2 > …, bỏ bài đã có trong phiên; ghi lại các phiên đã được hỏi."""

    def __init__(self):
        self.asked: list[list[str]] = []

    def candidates(self, session_track_ids: Sequence[str], k: int = 100) -> tuple[list[str], np.ndarray]:
        self.asked.append(list(session_track_ids))
        ids = [t for t in CATALOG if t not in set(session_track_ids)][:k]
        return ids, np.arange(len(ids), 0, -1, dtype=float)


def no_mix(current_track_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
    return np.zeros(len(candidate_ids))


def test_alpha_one_follows_taste_order_and_starts_from_last_seed_track():
    chains = generate_chains(RankedTaste(), [SEED], no_mix, alpha=1.0, length=10, n_candidates=20)
    assert chains == [["s4"] + [f"t{i}" for i in range(10)]]


def test_alpha_zero_follows_mix_from_the_track_just_added():
    best_next = {"s4": "t5", "t5": "t3", "t3": "t9"}  # điểm mix 1 cho đúng một bài kế, còn lại 0

    def chained_mix(current_track_id: str, candidate_ids: Sequence[str]) -> np.ndarray:
        return np.array([1.0 if c == best_next.get(current_track_id) else 0.0 for c in candidate_ids])

    chains = generate_chains(RankedTaste(), [SEED], chained_mix, alpha=0.0, length=3, n_candidates=20)
    assert chains == [["s4", "t5", "t3", "t9"]]


class BatchOnlyTaste(RankedTaste):
    """Như SASRec: chấm nhiều phiên một lượt; gọi từng phiên là lỗi."""

    def __init__(self):
        super().__init__()
        self.batch_sizes: list[int] = []

    def candidates(self, session_track_ids: Sequence[str], k: int = 100) -> tuple[list[str], np.ndarray]:
        raise AssertionError("mô hình có candidates_batch thì phải được chấm theo lô")

    def candidates_batch(self, sessions: Sequence[Sequence[str]], k: int = 100) -> list[tuple[list[str], np.ndarray]]:
        self.batch_sizes.append(len(sessions))
        return [RankedTaste.candidates(self, s, k) for s in sessions]


def test_model_with_candidates_batch_is_scored_in_batches_with_same_result():
    seeds = [SEED, ["s9", "s8", "s7", "s6", "t0"], ["t1", "t2", "t3", "t4", "t5"]]
    batched = BatchOnlyTaste()
    result = generate_chains(batched, seeds, no_mix, alpha=1.0, length=4, n_candidates=20)
    assert result == generate_chains(RankedTaste(), seeds, no_mix, alpha=1.0, length=4, n_candidates=20)
    assert batched.batch_sizes == [3, 3, 3, 3]  # mỗi bước một lượt cho cả 3 phiên


def test_each_step_asks_the_taste_model_with_the_session_grown_so_far():
    model = RankedTaste()
    generate_chains(model, [SEED], no_mix, alpha=1.0, length=3, n_candidates=20)
    assert model.asked == [SEED, SEED + ["t0"], SEED + ["t0", "t1"]]



def build_model(name: str, data):
    train = data.sessions[data.sessions["split"] == "train"]
    if name == "session_vector":  # khung chung của ALS và item2vec, dùng embedding giả của fake.py
        vectors, index = data.embeddings["fake"]
        return SessionVectorTaste("fake", vectors, index["track_id"].tolist(), decay=0.5, fallback=Popularity().fit(train))
    if name == "itemknn":
        return ItemKNN().fit(train)
    return Popularity().fit(train)


@pytest.mark.parametrize("name", ["popularity", "itemknn", "session_vector"])
def test_real_taste_models_generate_full_chains_without_repeats(name):
    data = make_fake_dataset(n_sessions=80, n_tracks=400, random_seed=3)
    model = build_model(name, data)
    seeds = [list(s) for s in data.eval_targets["seed_track_ids"]]
    chains = generate_chains(model, seeds, RuleMix(data.tracks), alpha=0.5)
    assert len(chains) == len(seeds)
    for seed, chain in zip(seeds, chains):
        generated = chain[1:]
        assert chain[0] == seed[-1]
        assert len(generated) == 10
        assert len(set(generated)) == 10, "không lặp bài trong chuỗi"
        assert not set(generated) & set(seed), "không gợi ý lại bài trong seed"
