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
    t = make_tracks(tracks_raw(), sessions()).set_index("track_id")
    validate_tracks(t.reset_index())
    assert t.loc[A, "camelot"] == "8A"   # A thứ
    assert t.loc[B, "camelot"] == "8B"   # C trưởng
    assert t.loc[C, "camelot"] == "2B"   # F# trưởng
    assert t.loc[B, "bpm"] == 126.5
    assert list(t.loc[[A, B, C], "n_occurrences"]) == [2, 1, 0]


def test_unusable_track_is_an_error_not_silently_dropped():
    with pytest.raises(ValueError, match="1 bài"):
        make_tracks(tracks_raw(tempo=[124.0, 0.0, 90.0]), sessions())


def test_session_track_missing_from_raw_features_is_an_error():
    with pytest.raises(ValueError, match="tracks_raw"):
        make_tracks(tracks_raw().iloc[:2], sessions())
