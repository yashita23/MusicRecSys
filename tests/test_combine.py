import numpy as np
import pandas as pd
import pytest

from src.data.fake import make_fake_dataset
from src.data.schema import SchemaError
from src.eval.run import evaluate
from src.rerank.combine import rerank, rerank_candidates
from src.rerank.mix_rule import RuleMix

NOW = "n" * 22  # bài đang phát
A, B, C = "a" * 22, "b" * 22, "c" * 22


def fixed_mix(scores: dict[str, float]):
    """Hàm điểm mix giả: điểm cố định theo ứng viên, không phụ thuộc bài đang phát."""
    return lambda current, cands: np.array([scores[t] for t in cands])


# Gu: A > B > C (gu_norm 1, 0.5, 0). Mix: B > C > A. Ba mức α cho ba thứ tự khác nhau.
TASTE = [3.0, 2.0, 1.0]
MIX = fixed_mix({A: 0.1, B: 0.9, C: 0.5})


def test_alpha_1_keeps_taste_order():
    out = rerank([A, B, C], TASTE, NOW, MIX, alpha=1.0)
    assert list(out["track_id"]) == [A, B, C]
    assert list(out["rank"]) == [1, 2, 3]


def test_alpha_0_orders_by_mix_only():
    out = rerank([A, B, C], TASTE, NOW, MIX, alpha=0.0)
    assert list(out["track_id"]) == [B, C, A]


def test_alpha_half_hand_computed_final_scores():
    # final = 0.5·gu_norm + 0.5·mix: A = 0.5 + 0.05 = 0.55, B = 0.25 + 0.45 = 0.70, C = 0 + 0.25 = 0.25
    out = rerank([A, B, C], TASTE, NOW, MIX, alpha=0.5).set_index("track_id")
    assert list(out.index) == [B, A, C]
    assert out.loc[A, "gu_norm"] == pytest.approx(1.0)
    assert out.loc[B, "gu_norm"] == pytest.approx(0.5)
    assert out.loc[C, "gu_norm"] == pytest.approx(0.0)
    assert out.loc[B, "mix"] == pytest.approx(0.9)
    assert list(out["final"]) == pytest.approx([0.70, 0.55, 0.25])
    assert out.loc[A, "taste_score"] == 3.0  # giữ điểm gu gốc để app hiện


def test_rule_mix_plugs_in_and_scores_against_the_current_track():
    tracks = pd.DataFrame(
        {
            "track_id": [NOW, A, B, C],
            "bpm": [120.0, 120.0, 121.2, 200.0],
            "camelot": ["8A", "3B", "8A", "9A"],
            "energy": [0.5, 0.5, 0.5, 0.5],
        }
    )
    # So với NOW (120 BPM, 8A, 0.5), w = 0.4/0.4/0.2:
    # A: BPM 1, tông 3B không hợp 0, năng lượng 1 → 0.6
    # B: lệch 1% → 1, cùng 8A → 1, năng lượng 1   → 1.0
    # C: 200 (×0.5 = 100) lệch 16,7% → 0, 9A cạnh 8A → 0.8, năng lượng 1 → 0.52
    out = rerank([A, B, C], TASTE, NOW, RuleMix(tracks), alpha=0.0).set_index("track_id")
    assert list(out.index) == [B, A, C]
    assert list(out["mix"]) == pytest.approx([1.0, 0.6, 0.52])


def test_k_cuts_the_list_and_none_keeps_every_candidate():
    assert list(rerank([A, B, C], TASTE, NOW, MIX, alpha=0.5, k=2)["track_id"]) == [B, A]
    assert len(rerank([A, B, C], TASTE, NOW, MIX, alpha=0.5, k=None)) == 3


def test_equal_taste_scores_give_gu_norm_1_and_mix_decides():
    out = rerank([A, B, C], [2.0, 2.0, 2.0], NOW, MIX, alpha=0.5)
    assert list(out["gu_norm"]) == [1.0, 1.0, 1.0]
    assert list(out["track_id"]) == [B, C, A]


@pytest.fixture(scope="module")
def fake():
    return make_fake_dataset(n_sessions=80, n_tracks=300, random_seed=3)


def test_reranked_file_with_alpha_1_scores_like_the_original_candidates(fake):
    cands, mix = fake.candidates["val"], RuleMix(fake.tracks)
    reranked = rerank_candidates(cands, fake.eval_targets, mix, alpha=1.0, k=None)
    assert evaluate(reranked, fake.eval_targets) == evaluate(cands, fake.eval_targets)


def test_reranked_top_10_is_scorable_and_mix_sees_the_last_seed_track(fake):
    seen: dict[str, int] = {}

    def spy_mix(current, cands):
        seen[current] = seen.get(current, 0) + 1
        return np.zeros(len(cands))

    targets = fake.eval_targets[fake.eval_targets["split"] == "val"]
    reranked = rerank_candidates(fake.candidates["val"], fake.eval_targets, spy_mix, alpha=0.3)
    assert reranked.groupby("session_id").size().eq(10).all()
    assert evaluate(reranked, fake.eval_targets).list_length == 10
    last_seed = targets["seed_track_ids"].map(lambda s: s[-1])
    assert seen == last_seed.value_counts().to_dict()


@pytest.mark.parametrize(
    "taste, mix, alpha, k, ids, message",
    [
        ([3.0, np.nan, 1.0], MIX, 0.5, 10, [A, B, C], "điểm gu"),
        ([3.0, np.inf, 1.0], MIX, 0.5, 10, [A, B, C], "điểm gu"),
        (TASTE, fixed_mix({A: 0.1, B: np.nan, C: 0.5}), 1.0, 10, [A, B, C], "điểm mix"),  # NaN đảo thứ tự cả ở α = 1
        (TASTE, lambda cur, cands: np.array([0.1, 0.2]), 0.5, 10, [A, B, C], "độ dài"),
        (TASTE, lambda cur, cands: 0.5, 0.5, 10, [A, B, C], "độ dài"),  # trả một số thay vì mảng
        (TASTE, MIX, 1.5, 10, [A, B, C], "α"),
        (TASTE, MIX, -0.1, 10, [A, B, C], "α"),
        (TASTE, MIX, 0.5, 0, [A, B, C], "k"),
        (TASTE, MIX, 0.5, 10, [A, A, C], "trùng"),
    ],
)
def test_invalid_inputs_are_rejected_with_a_clear_message(taste, mix, alpha, k, ids, message):
    with pytest.raises(ValueError, match=message):
        rerank(ids, taste, NOW, mix, alpha=alpha, k=k)


def test_candidates_of_a_session_missing_from_eval_targets_are_rejected(fake):
    targets = fake.eval_targets[fake.eval_targets["session_id"] != fake.candidates["val"]["session_id"].iloc[0]]
    with pytest.raises(SchemaError, match="không có trong eval_targets"):
        rerank_candidates(fake.candidates["val"], targets, zero_mix, alpha=0.5)


def test_empty_candidates_are_rejected(fake):
    with pytest.raises(SchemaError, match="rỗng"):
        rerank_candidates(fake.candidates["val"].iloc[:0], fake.eval_targets, zero_mix, alpha=0.5)


def zero_mix(current, cands):
    return np.zeros(len(cands))
