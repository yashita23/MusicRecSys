import json
import sqlite3

import pandas as pd
import pytest

from src.data.build import build, filter_sessions, main

ID = {c: c * 22 for c in "abcdmt"}  # Spotify ID giả 22 ký tự


@pytest.fixture
def toy_mpd(tmp_path):
    """3 slice, 6 playlist. t có tempo = 0, m không có dòng features → cả hai phải bị loại."""
    slices = {
        "mpd.slice.0-999.json": {0: "abcd", 1: "abc"},
        "mpd.slice.1000-1999.json": {2: "abct", 3: "abm"},
        "mpd.slice.2000-2999.json": {4: "abc", 5: "dabc"},
    }
    mpd = tmp_path / "mpd" / "data"
    mpd.mkdir(parents=True)
    for name, playlists in slices.items():
        data = {
            "playlists": [
                {
                    "pid": pid,
                    "tracks": [
                        {"pos": p, "track_uri": f"spotify:track:{ID[c]}", "track_name": f"Song {c}", "artist_name": f"Artist {c}"}
                        for p, c in enumerate(tracks)
                    ],
                }
                for pid, tracks in playlists.items()
            ]
        }
        (mpd / name).write_text(json.dumps(data), encoding="utf-8")
    db = tmp_path / "extracted.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE extracted(track_uri TEXT PRIMARY KEY, tempo REAL, key INTEGER, mode INTEGER, energy REAL)")
    rows = [(f"spotify:track:{ID[c]}", 120.0 + i, i, i % 2, 0.5) for i, c in enumerate("abcd")]
    rows.append((f"spotify:track:{ID['t']}", 0.0, 3, 1, 0.5))
    con.executemany("INSERT INTO extracted VALUES (?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return tmp_path / "mpd", db


def occurrences(sessions: dict[int, str]) -> pd.DataFrame:
    """{session_id: "abc"} → mỗi ký tự là một bài, theo đúng thứ tự pos."""
    return pd.DataFrame(
        [{"session_id": sid, "pos": p, "track_id": t} for sid, tracks in sessions.items() for p, t in enumerate(tracks)]
    )


def as_strings(df: pd.DataFrame) -> dict[int, str]:
    return {sid: "".join(g.sort_values("pos")["track_id"]) for sid, g in df.groupby("session_id")}


def test_filter_repeats_until_stable():
    # Lượt 1: bỏ d, e, x, y (1 lượt) → phiên 3 = "a", phiên 5 = "f" quá ngắn → bỏ.
    # Lượt 2: f chỉ còn 1 lượt → bỏ → phiên 4 = "ab" quá ngắn → bỏ. Lọc một lượt sẽ giữ sai phiên 4.
    occ = occurrences({1: "abc", 2: "abc", 3: "ade", 4: "fab", 5: "fxy"})
    kept, _ = filter_sessions(occ, min_track_occurrences=2, min_session_length=3)
    assert as_strings(kept) == {1: "abc", 2: "abc"}


def test_log_counts_every_filter_step():
    occ = occurrences({1: "abc", 2: "abc", 3: "ade", 4: "fab", 5: "fxy"})
    kept, log = filter_sessions(occ, min_track_occurrences=2, min_session_length=3)
    # 15 lượt → bỏ d,e,x,y (11) → bỏ phiên 3, 5 (9) → bỏ f (8) → bỏ phiên 4 (6) → vòng 3 không đổi
    assert [s.n_occurrences for s in log] == [15, 11, 9, 8, 6, 6, 6]
    assert [s.n_sessions for s in log] == [5, 5, 3, 3, 2, 2, 2]
    assert [s.n_tracks for s in log] == [8, 4, 4, 3, 3, 3, 3]
    assert log[1].name == "vòng 1: bỏ bài < 2 lượt"
    assert log[2].name == "vòng 1: bỏ phiên < 3 bài"


def test_build_drops_tracks_without_usable_features_before_filtering(toy_mpd):
    mpd_dir, db = toy_mpd
    r = build(mpd_dir, db, n_slices=None, min_track_occurrences=2, min_session_length=3)
    # t (tempo 0) và m (không có features) bị loại → phiên 3 còn "ab" < 3 bài → bị bỏ
    assert as_strings(r.sessions.assign(track_id=r.sessions["track_id"].str[0])) == {0: "abcd", 1: "abc", 2: "abc", 4: "abc", 5: "dabc"}
    assert sorted(r.tracks["track_id"]) == [ID[c] for c in "abcd"]
    a = r.tracks.set_index("track_id").loc[ID["a"]]
    assert (a["track_name"], a["artist_name"], a["tempo"], a["key"], a["mode"], a["energy"]) == ("Song a", "Artist a", 120.0, 0, 0, 0.5)


def test_same_random_seed_samples_the_same_slices(toy_mpd):
    mpd_dir, db = toy_mpd
    runs = [build(mpd_dir, db, n_slices=2, random_seed=7, min_track_occurrences=1, min_session_length=3) for _ in range(2)]
    pd.testing.assert_frame_equal(runs[0].sessions, runs[1].sessions)
    assert runs[0].sessions["session_id"].nunique() < 6  # chỉ 2/3 slice được lấy


def test_cli_writes_raw_files_and_log(toy_mpd, tmp_path):
    mpd_dir, db = toy_mpd
    out = tmp_path / "out"
    main(["--mpd-dir", str(mpd_dir), "--features", str(db), "--out", str(out), "--n-slices", "3",
          "--min-track-occurrences", "2", "--min-session-length", "3"])
    sessions = pd.read_parquet(out / "sessions_raw.parquet")
    tracks = pd.read_parquet(out / "tracks_raw.parquet")
    log = json.loads((out / "build_log.json").read_text(encoding="utf-8"))
    assert sessions["session_id"].nunique() == 5
    assert len(tracks) == 4
    assert log["steps"][-1]["n_occurrences"] == len(sessions)
    assert log["config"]["random_seed"] == 42


def test_pos_is_renumbered_without_gaps_and_keeps_order():
    occ = occurrences({1: "azbc", 2: "abc"})  # z chỉ có 1 lượt → bị bỏ khỏi giữa phiên 1
    kept, _ = filter_sessions(occ, min_track_occurrences=2, min_session_length=3)
    s1 = kept[kept["session_id"] == 1].sort_values("pos")
    assert list(s1["pos"]) == [0, 1, 2]
    assert list(s1["track_id"]) == ["a", "b", "c"]
    assert list(kept.columns) == ["session_id", "pos", "track_id"]
