import json
import sqlite3

import pytest

from src.data.coverage import (
    coverage_report,
    list_slices,
    load_features,
    load_mpd_occurrences,
    main,
    normalize_track_id,
    sample_slices,
)

ID = {c: c * 22 for c in "abcdef"}  # 22-char fake Spotify IDs


@pytest.mark.parametrize(
    "raw",
    [f"spotify:track:{ID['a']}", ID["a"], f"https://open.spotify.com/track/{ID['a']}?si=xyz"],
)
def test_normalize_track_id(raw):
    assert normalize_track_id(raw) == ID["a"]


def test_normalize_track_id_rejects_garbage():
    assert normalize_track_id("not-an-id") is None
    assert normalize_track_id(None) is None


@pytest.fixture
def toy(tmp_path):
    mpd = tmp_path / "mpd" / "data"
    mpd.mkdir(parents=True)
    # pl 0: a a b c ; pl 1: a d e   → 7 lượt, 5 bài
    playlists = [
        {"pid": 0, "tracks": [{"pos": i, "track_uri": f"spotify:track:{ID[c]}"} for i, c in enumerate("aabc")]},
        {"pid": 1, "tracks": [{"pos": i, "track_uri": f"spotify:track:{ID[c]}"} for i, c in enumerate("ade")]},
    ]
    (mpd / "mpd.slice.0-999.json").write_text(json.dumps({"playlists": playlists}), encoding="utf-8")

    db = tmp_path / "extracted.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE extracted(track_uri TEXT PRIMARY KEY, track_name TEXT, tempo REAL, key INTEGER, mode INTEGER, energy REAL)")
    rows = [
        (f"spotify:track:{ID['a']}", "A", 120.0, 8, 1, 0.7),  # dùng được
        (f"spotify:track:{ID['b']}", "B", 0.0, 5, 0, 0.5),    # tempo = 0
        (f"spotify:track:{ID['c']}", "C", 100.0, -1, 1, 0.4),  # không có key
        (f"spotify:track:{ID['f']}", "F", 90.0, 2, 1, 0.3),   # không có trong MPD
    ]  # d, e: không có dòng nào (thiếu features)
    con.executemany("INSERT INTO extracted VALUES (?,?,?,?,?,?)", rows)
    con.commit()
    con.close()
    return tmp_path / "mpd", db


def test_coverage_report_counts(toy):
    mpd_dir, db = toy
    occ = load_mpd_occurrences(list_slices(mpd_dir))
    feats = load_features(db)
    r = coverage_report(occ, feats, min_tracks=2, min_occurrences=2)

    assert r["n_playlists"] == 2
    assert r["n_occurrences"] == 7
    assert r["n_unique_tracks"] == 5
    assert r["track_coverage"] == pytest.approx(3 / 5)          # a, b, c
    assert r["occurrence_coverage"] == pytest.approx(5 / 7)     # a a b c a
    assert r["track_coverage_usable"] == pytest.approx(1 / 5)   # chỉ a
    assert r["occurrence_coverage_usable"] == pytest.approx(3 / 7)
    assert r["track_coverage_usable_among_ge2_occ"] == pytest.approx(1.0)  # chỉ a xuất hiện ≥ 2 lần
    assert r["playlists_ge2_usable_tracks"] == pytest.approx(1 / 2)        # pl 0 có a, a
    assert r["missing_among_matched"]["tempo_le_0"] == pytest.approx(1 / 3)
    assert r["missing_among_matched"]["key_minus_1"] == pytest.approx(1 / 3)


def test_load_features_autodetects_table(toy):
    _, db = toy
    feats = load_features(db)
    assert set(feats.columns) == {"track_id", "tempo", "key", "mode", "energy"}
    assert len(feats) == 4


def test_load_features_from_csv(tmp_path):
    csv = tmp_path / "tracks_features.csv"
    csv.write_text(f"id,name,tempo,key,mode,energy\n{ID['a']},A,120,8,1,0.7\n{ID['a']},A dup,121,8,1,0.7\n", encoding="utf-8")
    feats = load_features(csv)
    assert list(feats["track_id"]) == [ID["a"]]


def test_sample_slices_is_deterministic(tmp_path):
    paths = [tmp_path / f"mpd.slice.{i}.json" for i in range(10)]
    assert sample_slices(paths, 3, 7) == sample_slices(paths, 3, 7)
    assert len(sample_slices(paths, 3, 7)) == 3
    assert sample_slices(paths, None) == paths


def test_main_writes_json(toy, tmp_path):
    mpd_dir, db = toy
    out = tmp_path / "coverage.json"
    main(["--mpd-dir", str(mpd_dir), "--features", str(db), "--n-slices", "5", "--out", str(out)])
    saved = json.loads(out.read_text(encoding="utf-8"))
    assert saved["n_unique_tracks"] == 5
    assert saved["config"]["n_slices"] == 1
