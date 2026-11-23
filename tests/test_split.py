import json

import numpy as np
import pandas as pd

from src.data.schema import validate_dataset
from src.data.split import main, make_eval_targets, prepare, split_sessions


def tid(name: str) -> str:
    return name.ljust(22, "0")  # Spotify ID giả 22 ký tự


def unique_target_raw(n_sessions: int = 20):
    """Mỗi phiên: 5 bài chung, bài đích riêng của phiên (không xuất hiện ở đâu khác), 4 bài chung."""
    common = [tid(f"c{i}") for i in range(9)]
    rows = []
    for sid in range(n_sessions):
        tracks = common[:5] + [tid(f"u{sid}")] + common[5:]
        rows += [{"session_id": sid, "pos": p, "track_id": t} for p, t in enumerate(tracks)]
    sessions_raw = pd.DataFrame(rows)
    ids = sorted(sessions_raw["track_id"].unique())
    tracks_raw = pd.DataFrame(
        {"track_id": ids, "track_name": "n", "artist_name": "a", "tempo": 120.0, "key": 9, "mode": 0, "energy": 0.5}
    )
    return sessions_raw, tracks_raw


def raw_sessions(n_sessions: int, length: int = 10) -> pd.DataFrame:
    """sessions_raw giả: phiên i có các bài f"t{i}_{p}"."""
    return pd.DataFrame(
        [{"session_id": sid, "pos": p, "track_id": f"t{sid}_{p}"} for sid in range(n_sessions) for p in range(length)]
    )


def split_of(df: pd.DataFrame) -> pd.Series:
    return df.groupby("session_id")["split"].agg(lambda s: tuple(sorted(set(s))))


def test_every_session_lands_in_exactly_one_split():
    out = split_sessions(raw_sessions(100), random_seed=1)
    assert (split_of(out).map(len) == 1).all()
    assert len(out) == 100 * 10  # không mất dòng nào


def test_split_is_80_10_10_by_session():
    counts = split_of(split_sessions(raw_sessions(100), random_seed=1)).map(lambda t: t[0]).value_counts()
    assert counts.to_dict() == {"train": 80, "val": 10, "test": 10}


def sessions_with(splits_and_tracks: dict[int, tuple[str, str]]) -> pd.DataFrame:
    """{session_id: (split, "abcdefgh")} → mỗi ký tự là một bài, theo thứ tự pos."""
    return pd.DataFrame(
        [
            {"session_id": sid, "split": split, "pos": p, "track_id": t}
            for sid, (split, tracks) in splits_and_tracks.items()
            for p, t in enumerate(tracks)
        ]
    )


def test_eval_targets_take_first_five_as_seed_and_sixth_as_target_for_val_and_test_only():
    sessions = sessions_with({1: ("train", "abcdefghij"), 2: ("val", "klmnopqrst"), 3: ("test", "uvwxyzabcd")})
    et = make_eval_targets(sessions).set_index("session_id")
    assert list(et.index) == [2, 3]
    assert list(et.loc[2, "seed_track_ids"]) == list("klmno")
    assert et.loc[2, "target_track_id"] == "p"
    assert (et.loc[3, "split"], et.loc[3, "target_track_id"]) == ("test", "z")


def test_session_whose_target_repeats_a_seed_track_is_left_out():
    sessions = sessions_with({2: ("val", "klmnolqrst"), 3: ("val", "uvwxyzabcd")})  # phiên 2: bài đích "l" ∈ seed
    assert list(make_eval_targets(sessions)["session_id"]) == [3]


def test_prepare_reports_targets_unseen_in_train():
    r = prepare(*unique_target_raw(20), random_seed=3)
    assert r.report["n_sessions"] == {"train": 16, "val": 2, "test": 2}
    assert r.report["n_eval_targets"] == {"val": 2, "test": 2}
    assert r.report["n_dropped_target_in_seed"] == {"val": 0, "test": 0}
    assert r.report["share_target_unseen_in_train"] == {"val": 1.0, "test": 1.0}


def test_cli_writes_files_that_satisfy_the_contract(tmp_path):
    sessions_raw, tracks_raw = unique_target_raw(20)
    raw, out = tmp_path / "raw", tmp_path / "out"
    raw.mkdir()
    sessions_raw.to_parquet(raw / "sessions_raw.parquet", index=False)
    tracks_raw.to_parquet(raw / "tracks_raw.parquet", index=False)
    main(["--raw-dir", str(raw), "--out", str(out), "--random-seed", "3"])
    tracks, sessions, eval_targets = (pd.read_parquet(out / f"{n}.parquet") for n in ("tracks", "sessions", "eval_targets"))
    validate_dataset(tracks, sessions, eval_targets)
    report = json.loads((out / "split_report.json").read_text(encoding="utf-8"))
    assert report["random_seed"] == 3
    assert np.isclose(report["share_target_unseen_in_train"]["val"], 1.0)


def test_same_random_seed_same_split_and_different_seed_differs():
    a = split_sessions(raw_sessions(100), random_seed=1)
    b = split_sessions(raw_sessions(100), random_seed=1)
    c = split_sessions(raw_sessions(100), random_seed=2)
    pd.testing.assert_frame_equal(a, b)
    assert not a["split"].equals(c["split"])
