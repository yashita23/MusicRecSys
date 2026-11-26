import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="SASRec cần PyTorch")

from src.data.fake import make_fake_dataset, write_fake  # noqa: E402
from src.data.schema import validate_dataset  # noqa: E402
from src.models.sasrec.cli import main  # noqa: E402
from src.models.sasrec.data import seed_input, train_sequences, training_windows  # noqa: E402
from src.models.sasrec.model import SASRec, SASRecParams  # noqa: E402


def tid(name: str) -> str:
    return name.ljust(22, "x")


def sessions_frame(playlists: dict[int, tuple[str, list[str]]]) -> pd.DataFrame:
    return pd.DataFrame(
        [{"session_id": sid, "split": split, "pos": p, "track_id": t} for sid, (split, tracks) in playlists.items() for p, t in enumerate(tracks)]
    )


A, B, C, D, V = (tid(c) for c in "ABCDV")


def test_train_sequences_encode_train_playlists_in_listening_order_with_codes_from_1():
    # Mã bài theo track_id đã sắp xếp: A 1, B 2, C 3, D 4; 0 dành cho đệm. Phiên val (V) không được học.
    frame = sessions_frame({5: ("train", [C, A, B]), 2: ("train", [D, A]), 9: ("val", [V, A, B])})
    track_ids, sequences = train_sequences(frame.sample(frac=1, random_state=0))  # thứ tự dòng không quan trọng
    assert track_ids.tolist() == [A, B, C, D]
    assert [s.tolist() for s in sequences] == [[4, 1], [3, 1, 2]]  # theo session_id: phiên 2 rồi phiên 5


def test_a_short_playlist_becomes_one_right_padded_window_of_inputs_and_next_tracks():
    # [1, 2, 3] với maxlen 4: đầu vào [1, 2, 0, 0], bài kế tiếp [2, 3, 0, 0]; chuỗi luôn bắt đầu ở vị trí 1
    inputs, targets = training_windows([np.array([1, 2, 3])], maxlen=4, rng=np.random.default_rng(0))
    assert inputs.tolist() == [[1, 2, 0, 0]]
    assert targets.tolist() == [[2, 3, 0, 0]]


def test_a_seed_is_laid_out_exactly_like_the_first_five_tracks_of_a_training_window():
    # Playlist 6 bài: dự đoán bài 6 từ 5 bài đầu ở vị trí 1–5. Seed 5 bài khi gợi ý phải có đúng bố cục đó.
    inputs, _ = training_windows([np.array([1, 2, 3, 4, 5, 6])], maxlen=8, rng=np.random.default_rng(0))
    row, last = seed_input(np.array([1, 2, 3, 4, 5]), maxlen=8)
    assert inputs[0].tolist() == row.tolist() == [1, 2, 3, 4, 5, 0, 0, 0]
    assert last == 4  # đặc trưng lấy ở vị trí của bài seed cuối


def test_a_seed_longer_than_maxlen_keeps_its_last_tracks():
    row, last = seed_input(np.array([1, 2, 3, 4, 5]), maxlen=3)
    assert row.tolist() == [3, 4, 5] and last == 2


def test_a_long_playlist_contributes_a_random_contiguous_window_of_maxlen():
    playlist = np.arange(1, 21)  # 20 bài
    inputs, targets = training_windows([playlist], maxlen=5, rng=np.random.default_rng(1))
    window = np.concatenate([inputs[0], targets[0][-1:]])  # 6 bài liên tiếp (cửa sổ đầy, không có ô đệm)
    start = int(window[0]) - 1
    assert window.tolist() == playlist[start : start + 6].tolist()
    assert (targets[0][:-1] == inputs[0][1:]).all()


def test_playlists_with_a_single_track_give_no_training_window():
    inputs, targets = training_windows([np.array([7]), np.array([1, 2])], maxlen=3, rng=np.random.default_rng(0))
    assert inputs.tolist() == [[1, 0, 0]] and targets.tolist() == [[2, 0, 0]]


# 12 bài nối thành một vòng theo thứ tự ngẫu nhiên (không theo track_id); mỗi phiên train là 8 bài liên tiếp trên vòng
CYCLE = [tid(f"c{i:02d}") for i in np.random.default_rng(3).permutation(12)]


def cycle_sessions(n: int = 300, length: int = 8) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    playlists = {}
    for sid in range(n):
        start = int(rng.integers(0, len(CYCLE)))
        playlists[sid] = ("train", [CYCLE[(start + i) % len(CYCLE)] for i in range(length)])
    return sessions_frame(playlists)


TOY = SASRecParams(hidden=16, blocks=1, heads=1, dropout=0.0, lr=0.01, batch_size=64, maxlen=10, negatives=8, max_epochs=30)


@pytest.fixture(scope="module")
def sasrec():
    return SASRec(TOY, random_seed=0).fit(cycle_sessions())


def test_the_next_track_on_the_cycle_comes_first(sasrec):
    # Đồng xuất hiện không phân biệt bài trước với bài sau (c4 và c6 đều đứng cạnh c5); thứ tự thì có
    ids, scores = sasrec.candidates(CYCLE[3:6], k=3)
    assert ids[0] == CYCLE[6]
    assert np.all(np.diff(scores) <= 0)


def test_candidates_never_include_session_tracks(sasrec):
    ids, _ = sasrec.candidates(CYCLE[:5], k=7)
    assert len(ids) == 7 and not set(ids) & set(CYCLE[:5])


def test_a_seed_of_unknown_tracks_falls_back_to_popularity(sasrec):
    # Seed chỉ có bài không có trong train → ứng viên giống hệt Popularity học trên cùng dữ liệu
    from src.models.popularity import Popularity

    expected = Popularity().fit(cycle_sessions()).candidates([V], k=4)
    ids, scores = sasrec.candidates([V], k=4)
    assert ids == expected[0] and list(scores) == list(expected[1])


def test_scoring_many_sessions_at_once_matches_one_at_a_time_including_fallback(sasrec):
    sessions = [CYCLE[3:6], [V], CYCLE[:5], [V, CYCLE[8]]]  # phiên thứ hai chỉ có bài lạ → Popularity
    batched = sasrec.candidates_batch(sessions, k=4)
    for session, (ids, scores) in zip(sessions, batched):
        one_ids, one_scores = sasrec.candidates(session, k=4)
        assert ids == one_ids
        assert list(scores) == pytest.approx(list(one_scores), rel=1e-5)


def test_cli_trains_with_early_stopping_on_val_then_writes_candidates_and_a_reloadable_checkpoint(tmp_path, capsys):
    data = make_fake_dataset(n_sessions=150, n_tracks=400, random_seed=5)
    write_fake(data, tmp_path / "data")
    main(["--data", str(tmp_path / "data"), "--out", str(tmp_path / "out"), "--max-epochs", "3"])

    out = tmp_path / "out"
    cands = {split: pd.read_parquet(out / "candidates" / f"sasrec_{split}_run0.parquet") for split in ("val", "test")}
    validate_dataset(data.tracks, data.sessions, data.eval_targets, candidates=cands)

    report = json.loads(capsys.readouterr().out)
    assert [r["epoch"] for r in report["history"]] == [1, 2, 3]
    assert all("val_ndcg" in r for r in report["history"])
    assert report["val"]["n_sessions"] == int((data.eval_targets["split"] == "val").sum())

    reloaded = SASRec.load(out / "sasrec" / "sasrec_run0.pt", data.sessions)
    first = cands["test"][cands["test"]["session_id"] == cands["test"]["session_id"].min()]
    seed = data.eval_targets.set_index("session_id").loc[first["session_id"].iloc[0], "seed_track_ids"]
    ids, _ = reloaded.candidates(list(seed))
    assert ids == first.sort_values("rank")["track_id"].tolist()
