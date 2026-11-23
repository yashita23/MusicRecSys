import pandas as pd
import pytest

import numpy as np

from src.data.schema import (
    SchemaError,
    validate_candidates,
    validate_item_embeddings,
    validate_eval_targets,
    validate_sessions,
    validate_tracks,
)

A, B, C = "a" * 22, "b" * 22, "c" * 22


def sessions_df(lengths={1: 10, 2: 12}, splits={1: "train", 2: "val"}):
    rows = [
        {"session_id": sid, "split": splits[sid], "pos": p, "track_id": (A, B, C)[p % 3]}
        for sid, n in lengths.items()
        for p in range(n)
    ]
    return pd.DataFrame(rows)


def tracks_df(**overrides):
    df = pd.DataFrame(
        {
            "track_id": [A, B, C],
            "track_name": ["Song A", "Song B", "Song C"],
            "artist_name": ["X", "Y", "Z"],
            "tempo": [124.0, 126.0, 90.0],
            "bpm": [124.0, 126.0, 90.0],
            "key": [9, 0, 2],
            "mode": [0, 1, 1],
            "camelot": ["8A", "8B", "10B"],
            "energy": [0.7, 0.8, 0.3],
            "n_occurrences": [10, 7, 0],
        }
    )
    for col, values in overrides.items():
        df[col] = values
    return df


def test_valid_tracks_pass():
    validate_tracks(tracks_df())


def test_tracks_missing_column_is_reported():
    with pytest.raises(SchemaError, match="energy"):
        validate_tracks(tracks_df().drop(columns="energy"))


@pytest.mark.parametrize(
    "overrides, expected",
    [
        ({"energy": [0.7, 1.5, 0.3]}, "energy"),
        ({"bpm": [124.0, 0.0, 90.0]}, "bpm"),
        ({"key": [9, -1, 2]}, "key"),
        ({"camelot": ["8A", "13B", "10B"]}, "camelot"),
        ({"camelot": ["8B", "8B", "10B"]}, "camelot"),  # 8B nhưng mode = 0
        ({"camelot": ["8A", "8B", "5B"]}, "camelot"),   # D trưởng là 10B, không phải 5B
        ({"track_id": [A, A, C]}, "track_id"),
        ({"track_id": [A, "short", C]}, "track_id"),
        ({"n_occurrences": [10, -1, 0]}, "n_occurrences"),
        ({"key": [9.0, 0.0, 2.0]}, "key"),  # sai kiểu: phải là int
        ({"track_name": ["Song A", None, "Song C"]}, "track_name"),
    ],
)
def test_tracks_bad_values_are_reported(overrides, expected):
    with pytest.raises(SchemaError, match=expected):
        validate_tracks(tracks_df(**overrides))


def test_valid_sessions_pass():
    validate_sessions(sessions_df())


def test_session_shorter_than_10_tracks_is_reported():
    with pytest.raises(SchemaError, match="10"):
        validate_sessions(sessions_df(lengths={1: 9, 2: 12}))


def test_session_with_gap_in_pos_is_reported():
    df = sessions_df()
    df.loc[df.index[3], "pos"] = 50
    with pytest.raises(SchemaError, match="pos"):
        validate_sessions(df)


def test_session_split_across_two_sets_is_reported():
    df = sessions_df()
    df.loc[df.index[0], "split"] = "test"
    with pytest.raises(SchemaError, match="split"):
        validate_sessions(df)


def test_unknown_split_is_reported():
    with pytest.raises(SchemaError, match="split"):
        validate_sessions(sessions_df(splits={1: "train", 2: "dev"}))


D, E = "d" * 22, "e" * 22
SEED = [A, B, C, D, E]
F = "f" * 22


def eval_targets_df(seeds=(SEED,), targets=(F,), splits=("val",)):
    return pd.DataFrame(
        {
            "session_id": list(range(1, len(seeds) + 1)),
            "split": list(splits),
            "seed_track_ids": [list(s) for s in seeds],
            "target_track_id": list(targets),
        }
    )


def test_valid_eval_targets_pass():
    validate_eval_targets(eval_targets_df())


def test_eval_targets_pass_after_parquet_round_trip(tmp_path):
    path = tmp_path / "eval_targets.parquet"
    eval_targets_df().to_parquet(path)
    validate_eval_targets(pd.read_parquet(path))  # list → numpy array khi đọc lại


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"seeds": ([A, B, C, D],)}, "5"),
        ({"targets": (C,)}, "seed"),
        ({"splits": ("train",)}, "split"),
        ({"seeds": ([A, B, C, D, "bad"],)}, "seed_track_ids"),
    ],
)
def test_eval_targets_bad_rows_are_reported(kwargs, expected):
    with pytest.raises(SchemaError, match=expected):
        validate_eval_targets(eval_targets_df(**kwargs))


def test_eval_targets_duplicate_session_is_reported():
    df = pd.concat([eval_targets_df(), eval_targets_df()])
    with pytest.raises(SchemaError, match="session_id"):
        validate_eval_targets(df)


def candidates_df(tracks=(A, B, C), scores=(0.9, 0.5, 0.1), ranks=(1, 2, 3)):
    return pd.DataFrame({"session_id": [7] * len(tracks), "rank": list(ranks), "track_id": list(tracks), "taste_score": list(scores)})


def test_valid_candidates_pass():
    validate_candidates(candidates_df(), k=3)


def test_candidates_need_exactly_k_per_session():
    with pytest.raises(SchemaError, match="100"):
        validate_candidates(candidates_df())  # mặc định k = 100


@pytest.mark.parametrize(
    "kwargs, expected",
    [
        ({"ranks": (1, 2, 4)}, "rank"),
        ({"tracks": (A, A, C)}, "track_id"),
        ({"scores": (0.1, 0.5, 0.9)}, "taste_score"),
        ({"scores": (0.9, float("inf"), 0.1)}, "taste_score"),
    ],
)
def test_candidates_bad_rows_are_reported(kwargs, expected):
    with pytest.raises(SchemaError, match=expected):
        validate_candidates(candidates_df(**kwargs), k=3)


def embedding_index(tracks=(A, B, C)):
    return pd.DataFrame({"row": list(range(len(tracks))), "track_id": list(tracks)})


def test_valid_item_embeddings_pass():
    validate_item_embeddings(np.zeros((3, 8), dtype=np.float32), embedding_index())


@pytest.mark.parametrize(
    "vectors, index, expected",
    [
        (np.zeros((2, 8), dtype=np.float32), embedding_index(), "3"),         # số hàng lệch index
        (np.zeros((3, 8), dtype=np.float64), embedding_index(), "float32"),
        (np.zeros(3, dtype=np.float32), embedding_index(), "2 chiều"),
        (np.zeros((3, 8), dtype=np.float32), embedding_index((A, A, C)), "track_id"),
        (np.full((3, 8), np.nan, dtype=np.float32), embedding_index(), "hữu hạn"),
    ],
)
def test_bad_item_embeddings_are_reported(vectors, index, expected):
    with pytest.raises(SchemaError, match=expected):
        validate_item_embeddings(vectors, index)


def test_all_problems_are_listed_at_once():
    with pytest.raises(SchemaError) as e:
        validate_tracks(tracks_df(energy=[2.0, 0.5, 0.5], bpm=[-1.0, 120.0, 120.0]))
    assert len(e.value.problems) == 2
