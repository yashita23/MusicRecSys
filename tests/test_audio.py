import pandas as pd
import pytest

from src.data.schema import validate_tracks
from src.features.audio import make_tracks

A, B, C = "a" * 22, "b" * 22, "c" * 22


def tracks_raw(**overrides):
    df = pd.DataFrame(
        {
            "track_id": [A, B, C],
            "track_name": ["Song A", "Song B", "Song C"],
            "artist_name": ["X", "Y", "Z"],
            "tempo": [124.0, 126.5, 90.0],
            "key": [9, 0, 6],
            "mode": [0, 1, 1],
            "energy": [0.7, 0.8, 0.3],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def sessions():
    """Train: A A B (A 2 lượt, B 1 lượt). Val: A C → không được tính vào n_occurrences."""
    return pd.DataFrame(
        {
            "session_id": [1, 1, 1, 2, 2],
            "split": ["train", "train", "train", "val", "val"],
            "pos": [0, 1, 2, 0, 1],
            "track_id": [A, A, B, A, C],
        }
    )


def test_make_tracks_produces_contract_table():
    tracks = make_tracks(tracks_raw(), sessions())
    validate_tracks(tracks)
    by_id = tracks.set_index("track_id")
    assert by_id.loc[A, "camelot"] == "8A"   # A thứ
    assert by_id.loc[B, "camelot"] == "8B"   # C trưởng
    assert by_id.loc[C, "camelot"] == "2B"   # F# trưởng
    assert by_id.loc[B, "bpm"] == 126.5
    assert list(by_id.loc[[A, B, C], "n_occurrences"]) == [2, 1, 0]


def test_float_key_and_mode_from_sqlite_become_int():
    validate_tracks(make_tracks(tracks_raw(key=[9.0, 0.0, 6.0], mode=[0.0, 1.0, 1.0]), sessions()))


@pytest.mark.parametrize(
    "overrides",
    [{"tempo": [124.0, 0.0, 90.0]}, {"key": [9, -1, 6]}, {"key": [9.0, float("nan"), 6.0]}],
    ids=["tempo-0", "key-minus-1", "key-missing"],
)
def test_unusable_track_is_an_error_naming_the_track(overrides):
    with pytest.raises(ValueError, match=f"1 bài.*{B}"):
        make_tracks(tracks_raw(**overrides), sessions())


def test_duplicate_track_id_in_raw_features_is_an_error():
    with pytest.raises(ValueError, match="track_id"):
        make_tracks(pd.concat([tracks_raw(), tracks_raw().iloc[:1]], ignore_index=True), sessions())


def test_session_track_missing_from_raw_features_is_an_error():
    with pytest.raises(ValueError, match="tracks_raw"):
        make_tracks(tracks_raw().iloc[:2], sessions())
