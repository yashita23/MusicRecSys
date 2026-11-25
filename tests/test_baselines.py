import json

import numpy as np
import pandas as pd
import pytest

from src.data.fake import make_fake_dataset, write_fake
from src.data.schema import validate_dataset
from src.models.baselines import main
from src.models.item_knn import ItemKNN
from src.models.popularity import Popularity


def tid(name: str) -> str:
    return name * 22 if len(name) == 1 else name.ljust(22, "x")


A, B, C, D, E, F, G, H, V = (tid(c) for c in "ABCDEFGHV")


def train(*playlists: list[str]) -> pd.DataFrame:
    """Phiên train tự dựng; thêm một phiên val để chắc mô hình chỉ học từ train."""
    rows = [
        {"session_id": sid, "split": "train", "pos": p, "track_id": t}
        for sid, playlist in enumerate(playlists)
        for p, t in enumerate(playlist)
    ]
    rows += [{"session_id": 999, "split": "val", "pos": p, "track_id": V} for p in range(10)]
    return pd.DataFrame(rows)


# Số lượt trong train: A 4, B 3, C 2, D 1. V chỉ có ở val.
SESSIONS = train([A, B, C, A], [A, B, C, D], [A, B])


def test_popularity_ranks_by_train_occurrences_and_skips_session_tracks():
    ids, scores = Popularity().fit(SESSIONS).candidates([B], k=3)
    assert ids == [A, C, D]
    assert list(scores) == [4, 2, 1]


# Phiên [A, C], bài cuối C.
# Tầng 1, liền kề C→?: D (P1, P2) 2 lần, E (P3) 1 lần. Cặp ngược B→C, A→C không được tính.
#   Điểm 2 + đếm/đếm lớn nhất: D 3, E 2,5
# Tầng 2, đồng xuất hiện với A hoặc C (cộng số playlist): B = P4 (với A) + P2 (với C) = 2, F = P4 + P2 = 2
#   Điểm 1 + 2/2 = 2; hòa thì theo track_id: B trước F
# Tầng 3, Popularity phần còn lại: G 2 lượt, H 1 lượt → 0 + 2/2 = 1, 0 + 1/2 = 0,5
KNN_SESSIONS = train([A, C, D, E], [B, C, D, F], [C, E, A], [A, B, F], [G, G, H])


def test_item_knn_fills_from_adjacency_then_cooccurrence_then_popularity():
    ids, scores = ItemKNN().fit(KNN_SESSIONS).candidates([A, C], k=6)
    assert ids == [D, E, B, F, G, H]
    assert list(scores) == pytest.approx([3.0, 2.5, 2.0, 2.0, 1.0, 0.5])


def test_item_knn_with_a_last_track_unseen_in_train_falls_back_to_the_other_tiers():
    # V chỉ có ở val nên không có tầng liền kề. Playlist chứa A: P1, P3, P4
    # → đồng xuất hiện C 2, E 2, B 1, D 1, F 1 (điểm 2, 2, 1,5, 1,5, 1,5); còn lại G 2, H 1 lượt (điểm 1, 0,5)
    ids, scores = ItemKNN().fit(KNN_SESSIONS).candidates([A, V], k=7)
    assert ids == [C, E, B, D, F, G, H]
    assert list(scores) == pytest.approx([2.0, 2.0, 1.5, 1.5, 1.5, 1.0, 0.5])


@pytest.fixture(scope="module")
def fake():
    return make_fake_dataset(n_sessions=120, n_tracks=400, random_seed=2)


@pytest.mark.parametrize("model_cls", [Popularity, ItemKNN])
def test_every_eval_session_gets_k_candidates_without_seed_tracks(fake, model_cls):
    model = model_cls().fit(fake.sessions)
    for seed in fake.eval_targets["seed_track_ids"]:
        ids, scores = model.candidates(list(seed), k=100)
        assert len(ids) == 100 and len(set(ids)) == 100
        assert not set(ids) & set(seed)
        assert np.isfinite(scores).all() and (np.diff(scores) <= 0).all()


def test_cli_writes_contract_files_and_reports_val_accuracy(tmp_path, capsys):
    data_dir, out_dir = tmp_path / "data", tmp_path / "out"
    d = make_fake_dataset(n_sessions=120, n_tracks=400, random_seed=2)
    write_fake(d, data_dir)
    main(["--data", str(data_dir), "--out", str(out_dir)])

    for model in ("popularity", "itemknn"):
        cands = {split: pd.read_parquet(out_dir / "candidates" / f"{model}_{split}_run0.parquet") for split in ("val", "test")}
        validate_dataset(d.tracks, d.sessions, d.eval_targets, candidates=cands)

    report = json.loads(capsys.readouterr().out)
    assert set(report) == {"popularity", "itemknn"}
    assert report["itemknn"]["split"] == "val"
    assert report["itemknn"]["n_sessions"] == int((d.eval_targets["split"] == "val").sum())
    assert {"hit_rate", "ndcg", "mrr"} <= set(report["popularity"])
