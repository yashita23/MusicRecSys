import pandas as pd

from src.data.fake import make_fake_dataset
from src.data.schema import validate_dataset


def test_fake_dataset_satisfies_the_whole_contract():
    d = make_fake_dataset(n_sessions=60, n_tracks=300, random_seed=1)
    validate_dataset(d.tracks, d.sessions, d.eval_targets, candidates=d.candidates, embeddings=d.embeddings)


def test_fake_dataset_has_every_split_and_eval_rows_for_val_and_test():
    d = make_fake_dataset(n_sessions=60, n_tracks=300, random_seed=1)
    assert set(d.sessions["split"]) == {"train", "val", "test"}
    assert set(d.eval_targets["split"]) == {"val", "test"}
    assert set(d.candidates) == {"val", "test"}


def test_fake_dataset_covers_repeated_tracks_and_targets_unseen_in_train():
    d = make_fake_dataset(n_sessions=60, n_tracks=300, random_seed=1)
    repeats = d.sessions.duplicated(["session_id", "track_id"])
    assert repeats.any(), "cần có phiên lặp bài"
    train_tracks = set(d.sessions.loc[d.sessions["split"] == "train", "track_id"])
    assert (~d.eval_targets["target_track_id"].isin(train_tracks)).any(), "cần có bài đích không có trong train"


def test_same_random_seed_gives_same_data():
    a = make_fake_dataset(n_sessions=30, n_tracks=200, random_seed=5)
    b = make_fake_dataset(n_sessions=30, n_tracks=200, random_seed=5)
    c = make_fake_dataset(n_sessions=30, n_tracks=200, random_seed=6)
    pd.testing.assert_frame_equal(a.sessions, b.sessions)
    pd.testing.assert_frame_equal(a.candidates["val"], b.candidates["val"])
    assert not a.sessions.equals(c.sessions)
