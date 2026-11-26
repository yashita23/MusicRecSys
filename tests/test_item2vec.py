import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("gensim", reason="item2vec cần gensim (pip install gensim==4.4.0)")

from src.data.fake import make_fake_dataset, write_fake  # noqa: E402
from src.data.schema import validate_dataset, validate_item_embeddings  # noqa: E402
from src.models.item2vec import Item2Vec, Item2VecParams, main  # noqa: E402
from tests.toy_sessions import JAZZ, ROCK, two_clusters  # noqa: E402


@pytest.fixture(scope="module")
def item2vec():
    # Dữ liệu đồ chơi chỉ khoảng 2.400 bài mỗi epoch: 5 epoch chưa tách được hai cụm, 50 epoch thì rõ
    params = Item2VecParams(window=5, vector_size=8, decay=0.9)
    return Item2Vec(params, random_seed=0, workers=1, epochs=50).fit(two_clusters(n_per_cluster=200))


def test_a_rock_session_gets_rock_candidates_first(item2vec):
    ids, scores = item2vec.candidates(ROCK[:5], k=7)
    assert set(ids) == set(ROCK[5:])  # 7 bài rock còn lại, không bài jazz nào chen vào
    assert np.all(np.diff(scores) <= 0)


def test_embeddings_follow_the_contract_and_cover_only_train_tracks(item2vec):
    vectors, index = item2vec.taste.embeddings()
    validate_item_embeddings(vectors, index)
    assert vectors.shape == (24, 8) and vectors.dtype == np.float32
    assert index["track_id"].tolist() == sorted(ROCK + JAZZ)  # sắp theo track_id; bài chỉ có ở val không có vector


def test_cli_tunes_window_size_and_decay_on_val_then_writes_files(tmp_path, capsys):
    data = make_fake_dataset(n_sessions=150, n_tracks=400, random_seed=5)
    write_fake(data, tmp_path / "data")
    main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "out")])

    out = tmp_path / "out"
    cands = {split: pd.read_parquet(out / "candidates" / f"item2vec_{split}_run0.parquet") for split in ("val", "test")}
    vectors = np.load(out / "item_embeddings" / "item2vec.npy")
    index = pd.read_parquet(out / "item_embeddings" / "item2vec_tracks.parquet")
    validate_dataset(data.tracks, data.sessions, data.eval_targets, candidates=cands, embeddings={"item2vec": (vectors, index)})

    chosen = json.loads((out / "item2vec_params.json").read_text(encoding="utf-8"))
    grid = chosen["grid"]
    assert len(grid) == 16  # window {3, 10} × vector_size {128, 256} × decay {0,5; 0,7; 0,9; 1,0}
    best = max(grid, key=lambda row: row["ndcg"])
    assert chosen["params"] == {key: best[key] for key in ("window", "vector_size", "decay")}
    assert vectors.shape[1] == chosen["params"]["vector_size"]
    assert json.loads(capsys.readouterr().out)["params"] == chosen["params"]
