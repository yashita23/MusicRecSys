import json

import pandas as pd
import pytest

from src.data.fake import make_fake_dataset, write_fake
from src.data.schema import SchemaError
from src.eval.run import evaluate, main

SEED = ["s" * 21 + str(i) for i in range(5)]


def tid(i: int) -> str:
    return f"t{i:021d}"


def targets(target_of: dict[int, str], split: str = "val") -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session_id": list(target_of),
            "split": split,
            "seed_track_ids": [SEED] * len(target_of),
            "target_track_id": list(target_of.values()),
        }
    )


def candidates(lists: dict[int, list[str]]) -> pd.DataFrame:
    rows = [
        {"session_id": sid, "rank": r, "track_id": t, "taste_score": float(len(ids) - r)}
        for sid, ids in lists.items()
        for r, t in enumerate(ids, start=1)
    ]
    return pd.DataFrame(rows)


def test_target_at_rank_1_scores_perfectly():
    acc = evaluate(candidates({1: [tid(0), tid(1), tid(2)]}), targets({1: tid(0)}))
    assert acc.n_sessions == 1
    assert acc.hit_rate == 1.0
    assert acc.ndcg == 1.0
    assert acc.mrr == 1.0


def test_hand_computed_mix_of_ranks_1_3_11_and_missing():
    # Hạng 1:  hit 1, NDCG 1,            RR 1
    # Hạng 3:  hit 1, NDCG 1/log2(4)=0.5, RR 1/3
    # Hạng 11: hit 0 (ngoài top-10), NDCG 0, nhưng RR 1/11 (MRR tính trên cả danh sách)
    # Không có: 0, 0, 0
    ids = [tid(i) for i in range(12)]
    cands = candidates({1: ids, 2: ids, 3: ids, 4: ids})
    acc = evaluate(cands, targets({1: ids[0], 2: ids[2], 3: ids[10], 4: tid(99)}))
    assert acc.n_sessions == 4
    assert acc.hit_rate == pytest.approx(0.5)
    assert acc.ndcg == pytest.approx(0.375)
    assert acc.mrr == pytest.approx(47 / 132)  # (1 + 1/3 + 1/11) / 4
    assert acc.list_length == 12  # MRR tính trên 12 bài; chỉ so MRR giữa các danh sách cùng độ dài


def test_sessions_with_lists_of_different_lengths_are_rejected():
    # MRR trên 100 bài và trên 10 bài không so được với nhau, nên một lần chấm phải cùng độ dài
    uneven = candidates({1: [tid(0), tid(1), tid(2)], 2: [tid(0), tid(1)]})
    with pytest.raises(SchemaError, match="cùng độ dài"):
        evaluate(uneven, targets({1: tid(0), 2: tid(0)}))


def test_candidates_containing_a_seed_track_are_rejected():
    leaked = candidates({1: [tid(0), SEED[2], tid(1)]})
    with pytest.raises(SchemaError, match="seed"):
        evaluate(leaked, targets({1: tid(0)}))


def test_eval_session_without_candidates_is_rejected():
    # Bỏ qua phiên 2 sẽ làm điểm cao giả tạo, nên phải báo lỗi chứ không chấm trên phần còn lại
    with pytest.raises(SchemaError, match="không có ứng viên"):
        evaluate(candidates({1: [tid(0)]}), targets({1: tid(0), 2: tid(0)}))


def test_a_sample_of_sessions_can_be_scored_when_asked_explicitly():
    # Grid search (#25) chỉ chạy trên một mẫu phiên val; mặc định vẫn bắt buộc đủ mọi phiên
    acc = evaluate(candidates({1: [tid(0), tid(1)]}), targets({1: tid(0), 2: tid(0)}), sample=True)
    assert acc.n_sessions == 1
    assert acc.hit_rate == 1.0


def test_only_sessions_of_the_candidates_split_are_scored():
    # eval_targets.parquet thật chứa cả val lẫn test; file candidates chỉ thuộc một split
    both = pd.concat([targets({1: tid(0)}, split="val"), targets({2: tid(0)}, split="test")], ignore_index=True)
    acc = evaluate(candidates({1: [tid(0)]}), both)
    assert acc.split == "val"
    assert acc.n_sessions == 1
    assert acc.hit_rate == 1.0


def test_candidates_mixing_val_and_test_sessions_are_rejected():
    both = pd.concat([targets({1: tid(0)}, split="val"), targets({2: tid(0)}, split="test")], ignore_index=True)
    with pytest.raises(SchemaError, match="một split"):
        evaluate(candidates({1: [tid(0)], 2: [tid(0)]}), both)


@pytest.mark.parametrize(
    "ranks, track_ids",
    [
        ([0, 1, 2], [tid(0), tid(1), tid(2)]),  # đánh hạng từ 0: bài đích ở hạng 0 sẽ bị tính là trượt
        ([1, 1, 2], [tid(0), tid(1), tid(2)]),  # trùng hạng
        ([1, 2, 4], [tid(0), tid(1), tid(2)]),  # hạng bị nhảy
        ([1, 2, 3], [tid(0), tid(0), tid(2)]),  # một bài xuất hiện hai lần trong phiên
    ],
)
def test_lists_with_broken_ranks_or_repeated_tracks_are_rejected(ranks, track_ids):
    ranked = pd.DataFrame({"session_id": 1, "rank": ranks, "track_id": track_ids})
    with pytest.raises(SchemaError, match="rank|trùng"):
        evaluate(ranked, targets({1: tid(0)}))


def test_empty_list_is_rejected_with_a_clear_message():
    empty = pd.DataFrame({"session_id": pd.Series(dtype=int), "rank": pd.Series(dtype=int), "track_id": pd.Series(dtype=str)})
    with pytest.raises(SchemaError, match="rỗng"):
        evaluate(empty, targets({1: tid(0)}))


def test_cli_scores_a_candidates_file_of_the_fake_dataset(tmp_path, capsys):
    data = make_fake_dataset(n_sessions=60, n_tracks=300, random_seed=1)
    write_fake(data, tmp_path)
    main(["--data", str(tmp_path), "--candidates", str(tmp_path / "candidates" / "fake_val_run0.parquet")])
    out = json.loads(capsys.readouterr().out)
    assert out["split"] == "val"
    assert out["n_sessions"] == int((data.eval_targets["split"] == "val").sum())
    assert {"hit_rate", "ndcg", "mrr"} <= set(out)
