"""Cắt phiên nghe last.fm từ lịch sử nghe (ticket #55): cắt theo khoảng nghỉ, gộp lặp, lọc, chia theo user."""
import json

import pandas as pd
import pytest

from src.data.lastfm import assign_user_splits, cut_sessions, main, usable_tracks, validate_lastfm_sessions
from src.data.schema import SchemaError


def tid(name: str) -> str:
    return name.ljust(16, "x")  # track_id của Music4All-Onion: 16 ký tự


A, B, C, D, E, F, G, H, I, J, X = (tid(c) for c in "ABCDEFGHIJX")  # X: không có đặc trưng dùng được


def ev(user: int, track: str, when: str) -> dict:
    return {"user_id": user, "track_id": track, "timestamp": pd.Timestamp(when)}


EVENTS = pd.DataFrame(
    [
        # user 1, phiên 1: nghỉ đúng 30 phút vẫn cùng phiên; C C liên tiếp gộp thành C
        ev(1, A, "2013-01-01 10:00"), ev(1, B, "2013-01-01 10:30"), ev(1, C, "2013-01-01 10:31"),
        ev(1, C, "2013-01-01 10:32"), ev(1, D, "2013-01-01 10:33"), ev(1, E, "2013-01-01 10:34"),
        # user 1, phiên 2: nghỉ 31 phút → phiên mới; X không có đặc trưng bị bỏ
        ev(1, F, "2013-01-01 11:05"), ev(1, G, "2013-01-01 11:06"), ev(1, H, "2013-01-01 11:07"),
        ev(1, I, "2013-01-01 11:08"), ev(1, X, "2013-01-01 11:09"), ev(1, J, "2013-01-01 11:10"),
        # timestamp hỏng (epoch 0): bỏ trước khi cắt
        ev(1, A, "1970-01-01 01:00:36"),
        # user 2: chỉ 3 bài → phiên ngắn bị bỏ
        ev(2, A, "2013-01-02 09:00"), ev(2, B, "2013-01-02 09:01"), ev(2, C, "2013-01-02 09:02"),
        # user 3: B và A cùng thời điểm → xếp theo track_id (A trước B)
        ev(3, B, "2013-01-03 12:00"), ev(3, A, "2013-01-03 12:00"), ev(3, C, "2013-01-03 12:01"),
        ev(3, D, "2013-01-03 12:02"), ev(3, E, "2013-01-03 12:03"),
    ]
)
USABLE = {A, B, C, D, E, F, G, H, I, J}


def test_sessions_are_cut_on_gaps_collapsed_filtered_and_numbered():
    sessions, _ = cut_sessions(EVENTS.sample(frac=1, random_state=0), USABLE)  # đầu vào không sắp xếp
    lists = sessions.groupby("session_id")["track_id"].agg(list)
    assert lists.tolist() == [[A, B, C, D, E], [F, G, H, I, J], [A, B, C, D, E]]
    assert sessions.groupby("session_id")["user_id"].first().tolist() == [1, 1, 3]
    assert sessions.groupby("session_id")["pos"].agg(list).tolist() == [list(range(5))] * 3
    assert sessions.columns.tolist() == ["session_id", "user_id", "pos", "track_id"]


def test_the_funnel_counts_events_sessions_tracks_and_users_after_each_step():
    _, funnel = cut_sessions(EVENTS, USABLE)
    steps = {row["step"]: row for row in funnel}
    assert steps["đầu vào"]["events"] == 21
    assert steps["bỏ timestamp hỏng"]["events"] == 20
    assert steps["cắt theo khoảng nghỉ"]["sessions"] == 4  # user 1: 2 phiên, user 2: 1, user 3: 1
    assert steps["bỏ bài không có đặc trưng"]["events"] == 19
    assert steps["gộp lặp liên tiếp"]["events"] == 18
    assert steps["giữ phiên ≥ 5 bài"] == {"step": "giữ phiên ≥ 5 bài", "events": 15, "sessions": 3, "tracks": 10, "users": 2}


def test_usable_tracks_need_essentia_features_with_a_positive_bpm():
    essentia = pd.DataFrame({"id": [A, B, C, D], "rhythm.bpm": [120.0, 0.0, float("nan"), -3.0]})
    assert usable_tracks(essentia) == {A}


def test_categorical_track_ids_give_the_same_sessions_as_strings():
    # Trên Kaggle track_id được mã hóa Categorical (categories sắp xếp) để vừa RAM
    as_category = EVENTS.assign(track_id=pd.Categorical(EVENTS["track_id"], categories=sorted(EVENTS["track_id"].unique())))
    expected, _ = cut_sessions(EVENTS, USABLE)
    actual, _ = cut_sessions(as_category.sample(frac=1, random_state=1), USABLE)
    pd.testing.assert_frame_equal(actual, expected)


def many_users(n_users: int = 200) -> pd.DataFrame:
    """Mỗi user 1–3 phiên 5 bài; session_id liền mạch."""
    rows, sid = [], 0
    for user in range(n_users):
        for _ in range(1 + user % 3):
            rows += [{"session_id": sid, "user_id": user, "pos": p, "track_id": tid(str(p))} for p in range(5)]
            sid += 1
    return pd.DataFrame(rows)


def test_splits_are_by_user_80_10_10_so_no_user_appears_in_two_splits():
    sessions = assign_user_splits(many_users(), random_seed=42)
    assert sessions.columns.tolist() == ["session_id", "user_id", "split", "pos", "track_id"]
    assert (sessions.groupby("user_id")["split"].nunique() == 1).all()
    users_per_split = sessions.groupby("split")["user_id"].nunique().to_dict()
    assert users_per_split == {"train": 160, "val": 20, "test": 20}  # 10% val, 10% test của 200 user


def test_the_same_random_seed_gives_the_same_user_split():
    a = assign_user_splits(many_users(), random_seed=42)
    b = assign_user_splits(many_users().sample(frac=1, random_state=3).sort_values(["session_id", "pos"]), random_seed=42)
    pd.testing.assert_frame_equal(a.reset_index(drop=True), b.reset_index(drop=True))


@pytest.fixture
def onion_dir(tmp_path):
    """Output giả của notebook 09: listening_events.parquet và essentia.parquet (X không có trong essentia)."""
    root = tmp_path / "onion"
    root.mkdir()
    EVENTS.astype({"user_id": "int32"}).to_parquet(root / "listening_events.parquet", index=False)
    pd.DataFrame({"id": sorted(USABLE), "rhythm.bpm": 120.0}).to_parquet(root / "essentia.parquet", index=False)
    return root


def test_cli_writes_split_sessions_and_a_log_with_funnel_and_gap_sensitivity(onion_dir, tmp_path, capsys):
    out = tmp_path / "out" / "lastfm"
    main(["--onion", str(onion_dir), "--out", str(out)])

    sessions = pd.read_parquet(out / "sessions.parquet")
    validate_lastfm_sessions(sessions)
    assert sessions["session_id"].nunique() == 3

    log = json.loads((out / "lastfm_log.json").read_text(encoding="utf-8"))
    assert log == json.loads(capsys.readouterr().out)
    assert log["gap_minutes"] == 30 and log["sample_users"] == 1.0 and log["min_timestamp"] == "2002-01-01 00:00:00"
    assert log["funnel"][-1]["sessions"] == 3
    # 15 phút: phiên đầu của user 1 tách sau A, còn 4 bài → bỏ; 60 phút: user 1 thành một phiên 10 bài
    assert log["sensitivity"] == {
        "15": {"sessions": 2, "pairs": 8},
        "30": {"sessions": 3, "pairs": 12},
        "60": {"sessions": 2, "pairs": 13},
    }
    assert set(log["splits"]) <= {"train", "val", "test"}


def test_validation_rejects_a_user_in_two_splits_and_broken_positions():
    good = assign_user_splits(many_users(10))
    validate_lastfm_sessions(good)
    two_splits = good.copy()  # user 1 có hai phiên (1, 2): đổi split của riêng phiên 1
    own = good.loc[good["session_id"] == 1, "split"].iloc[0]
    two_splits.loc[two_splits["session_id"] == 1, "split"] = "test" if own != "test" else "val"
    with pytest.raises(SchemaError, match="split"):
        validate_lastfm_sessions(two_splits)
    gap_in_pos = good.assign(pos=good["pos"].where(good["pos"] != 2, 7))
    with pytest.raises(SchemaError, match="pos"):
        validate_lastfm_sessions(gap_in_pos)


def test_large_user_ids_take_the_fallback_sort_and_give_the_same_sessions():
    # user_id ≥ 2^17 không gộp được vào một khóa int64 → lexsort; kết quả phải như nhau
    shifted = EVENTS.assign(user_id=EVENTS["user_id"] + 2**20)
    expected, _ = cut_sessions(EVENTS, USABLE)
    actual, _ = cut_sessions(shifted, USABLE)
    pd.testing.assert_frame_equal(actual.drop(columns="user_id"), expected.drop(columns="user_id"))
    assert (actual["user_id"] - 2**20).tolist() == expected["user_id"].tolist()


def test_an_empty_track_id_is_reported_instead_of_corrupting_the_sort():
    with pytest.raises(ValueError, match="rỗng"):
        cut_sessions(EVENTS.assign(track_id=EVENTS["track_id"].where(EVENTS.index != 3)), USABLE)


def test_validation_rejects_session_ids_that_skip_a_number():
    good = assign_user_splits(many_users(10))
    with pytest.raises(SchemaError, match="session_id"):
        validate_lastfm_sessions(good.assign(session_id=good["session_id"] * 2))


def test_sampling_users_happens_on_read_and_the_log_keeps_the_full_event_count(onion_dir, tmp_path):
    log = main(["--onion", str(onion_dir), "--out", str(tmp_path / "out"), "--sample-users", "0.5"])
    assert log["sample_users"] == 0.5 and log["events_total"] == 21
    assert log["funnel"][0]["users"] == 2  # round(0,5 × 3 user) = 2, lấy ngay khi đọc parquet
