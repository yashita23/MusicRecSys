import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("implicit", reason="ALS cần implicit (pip install implicit==0.7.3)")

from src.data.fake import make_fake_dataset, write_fake  # noqa: E402
from src.data.schema import validate_dataset, validate_item_embeddings  # noqa: E402
from src.models.als import ALS, ALSParams, main  # noqa: E402


def tid(prefix: str, i: int) -> str:
    return f"{prefix}{i}".ljust(22, "x")


ROCK = [tid("rock", i) for i in range(12)]
JAZZ = [tid("jazz", i) for i in range(12)]
NEVER_TRAIN = tid("val", 0)


def two_clusters(n_per_cluster: int = 40, length: int = 6, random_seed: int = 0) -> pd.DataFrame:
    """Playlist train chỉ chứa bài rock hoặc chỉ bài jazz; thêm một phiên val để chắc ALS chỉ học từ train."""
    rng = np.random.default_rng(random_seed)
    rows = []
    for sid in range(2 * n_per_cluster):
        cluster = ROCK if sid % 2 == 0 else JAZZ
        for pos, t in enumerate(rng.choice(cluster, size=length, replace=False)):
            rows.append({"session_id": sid, "split": "train", "pos": pos, "track_id": t})
    rows += [{"session_id": 999, "split": "val", "pos": p, "track_id": NEVER_TRAIN} for p in range(6)]
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def als():
    # 2 chiều cho 2 cụm: thừa chiều so với dữ liệu đồ chơi 24 bài thì ALS khớp từng playlist và vector bài nhiễu
    return ALS(ALSParams(factors=2, regularization=0.1, decay=0.9), random_seed=0).fit(two_clusters())


def test_a_rock_session_gets_rock_candidates_first(als):
    seed = ROCK[:5]
    ids, scores = als.candidates(seed, k=7)
    assert set(ids) == set(ROCK[5:])  # 7 bài rock còn lại, không bài jazz nào chen vào
    assert np.all(np.diff(scores) <= 0)


def test_embeddings_follow_the_contract_and_cover_only_train_tracks(als):
    vectors, index = als.taste.embeddings()
    validate_item_embeddings(vectors, index)
    assert vectors.shape == (24, 2) and vectors.dtype == np.float32
    assert set(index["track_id"]) == set(ROCK + JAZZ)  # bài chỉ có ở val không có vector


def test_same_random_seed_gives_the_same_vectors():
    params = ALSParams(factors=2, regularization=0.1)
    a = ALS(params, random_seed=3).fit(two_clusters()).taste.embeddings()[0]
    b = ALS(params, random_seed=3).fit(two_clusters()).taste.embeddings()[0]
    np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def fake_dir(tmp_path_factory):
    data = make_fake_dataset(n_sessions=150, n_tracks=400, random_seed=5)
    root = tmp_path_factory.mktemp("fake")
    write_fake(data, root)
    return data, root


def test_cli_tunes_on_val_then_writes_candidates_embeddings_and_chosen_params(fake_dir, tmp_path, capsys):
    data, root = fake_dir
    main(["--data", str(root), "--out", str(tmp_path)])

    cands = {split: pd.read_parquet(tmp_path / "candidates" / f"als_{split}_run0.parquet") for split in ("val", "test")}
    vectors = np.load(tmp_path / "item_embeddings" / "als.npy")
    index = pd.read_parquet(tmp_path / "item_embeddings" / "als_tracks.parquet")
    validate_dataset(data.tracks, data.sessions, data.eval_targets, candidates=cands, embeddings={"als": (vectors, index)})

    chosen = json.loads((tmp_path / "als_params.json").read_text(encoding="utf-8"))
    grid = chosen["grid"]
    assert len(grid) == 24  # factors {128, 256} × regularization {0,1; 1; 10} × decay {0,5; 0,7; 0,9; 1,0}
    best = max(grid, key=lambda row: row["ndcg"])
    assert chosen["params"] == {key: best[key] for key in ("factors", "regularization", "decay")}
    assert vectors.shape[1] == chosen["params"]["factors"]

    report = json.loads(capsys.readouterr().out)
    assert report["params"] == chosen["params"] and report["val"]["ndcg"] == pytest.approx(best["ndcg"])


def test_a_later_run_reuses_the_chosen_params_without_retuning(fake_dir, tmp_path, capsys):
    _, root = fake_dir
    params = tmp_path / "params.json"
    params.write_text(json.dumps({"params": {"factors": 64, "regularization": 0.1, "decay": 0.7}}), encoding="utf-8")
    main(["--data", str(root), "--out", str(tmp_path / "out"), "--run", "1", "--params", str(params)])

    assert (tmp_path / "out" / "candidates" / "als_test_run1.parquet").is_file()
    assert not (tmp_path / "out" / "item_embeddings").exists()  # embedding chỉ xuất ở lần chạy 0
    assert "grid" not in json.loads(capsys.readouterr().out)


def test_a_later_run_without_the_chosen_params_is_refused(fake_dir, tmp_path):
    # Lần chạy 1, 2 phải dùng lại tham số của lần chạy 0; tự chọn lại với random_seed khác sẽ ghi đè als_params.json
    _, root = fake_dir
    with pytest.raises(ValueError, match="--params"):
        main(["--data", str(root), "--out", str(tmp_path), "--run", "1"])


def test_fitting_without_train_sessions_says_so():
    only_val = two_clusters().assign(split="val")
    with pytest.raises(ValueError, match="train"):
        ALS().fit(only_val)
