from pathlib import Path

import pandas as pd
import pytest

pytest.importorskip("tomllib", reason="app cần Python 3.11+ (đọc config bằng tomllib)")

from app.config import load_config  # noqa: E402
from app.player import Player  # noqa: E402
from src.data.fake import make_fake_dataset, write_fake  # noqa: E402


def tid(name: str) -> str:
    return name.ljust(22, "x")


SEED = [tid(f"S{i}") for i in range(5)]  # bài cuối S4 đang phát: 8A, 124 BPM, năng lượng 0,7
P, Q, R = tid("P"), tid("Q"), tid("R")


def track(track_id: str, camelot: str, bpm: float, energy: float) -> dict:
    return {"track_id": track_id, "track_name": f"Bài {track_id[:2]}", "artist_name": "Nghệ sĩ", "bpm": bpm, "camelot": camelot, "energy": energy}


TRACKS = pd.DataFrame(
    [track(t, "1A", 100, 0.5) for t in SEED[:-1]]
    + [
        track(SEED[-1], "8A", 124, 0.7),
        # Điểm mix so với S4 (0,4·s_bpm + 0,4·s_key + 0,2·s_energy):
        track(P, "3B", 90, 0.1),   # BPM lệch 27% → 0, tông không mix được → 0, năng lượng 0,4 → 0,08
        track(Q, "8A", 126, 0.68),  # lệch 1,6% → 1, cùng tông → 1, năng lượng 0,98 → 0,996
        track(R, "9A", 124, 0.7),   # cùng BPM → 1, tông kề → 0,8, năng lượng 1 → 0,92
    ]
)
EVAL_TARGETS = pd.DataFrame({"session_id": [7], "split": ["test"], "seed_track_ids": [SEED], "target_track_id": [Q]})
CANDIDATES = pd.DataFrame({"session_id": 7, "rank": [1, 2, 3], "track_id": [P, Q, R], "taste_score": [0.9, 0.5, 0.1]})


@pytest.fixture
def player():
    return Player(TRACKS, EVAL_TARGETS, CANDIDATES)


def test_now_playing_is_the_last_seed_track(player):
    assert player.playlists == [7]
    assert player.now_playing(7)["track_id"] == SEED[-1]


def test_queue_follows_taste_at_alpha_1_and_mix_at_alpha_0(player):
    assert player.queue(7, alpha=1.0)["track_id"].tolist() == [P, Q, R]
    at_mix = player.queue(7, alpha=0.0)
    assert at_mix["track_id"].tolist() == [Q, R, P]
    assert at_mix["mix"].tolist() == pytest.approx([0.996, 0.92, 0.08])


def test_queue_rows_carry_what_the_player_shows(player):
    row = player.queue(7, alpha=0.5).iloc[0]
    for column in ("rank", "track_name", "artist_name", "bpm", "camelot", "energy", "gu_norm", "mix", "final"):
        assert column in row.index


def write_config(path: Path, body: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


def test_relative_data_paths_resolve_against_the_config_folder_not_the_working_directory(tmp_path, monkeypatch):
    cfg_path = write_config(
        tmp_path / "app" / "config.toml",
        """
[data]
tracks = "../data/fake/tracks.parquet"
eval_targets = "../data/fake/eval_targets.parquet"
candidates = "/abs/candidates/fake_test_run0.parquet"
""",
    )
    monkeypatch.chdir(tmp_path / "app")
    cfg = load_config(cfg_path)
    assert cfg.tracks == (tmp_path / "data" / "fake" / "tracks.parquet").resolve()
    assert cfg.eval_targets == (tmp_path / "data" / "fake" / "eval_targets.parquet").resolve()
    assert cfg.candidates == Path("/abs/candidates/fake_test_run0.parquet").resolve()


def test_a_missing_data_path_names_the_key_and_the_config_file(tmp_path):
    cfg_path = write_config(tmp_path / "config.toml", '[data]\ntracks = "t.parquet"\neval_targets = "e.parquet"\n')
    with pytest.raises(ValueError, match="candidates") as err:
        load_config(cfg_path)
    assert "config.toml" in str(err.value)


APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


@pytest.fixture
def fake_config(tmp_path, monkeypatch):
    """Dữ liệu giả ghi ra thư mục tạm, cùng một config trỏ tới nó; app đọc config qua MIXNEXT_CONFIG."""
    write_fake(make_fake_dataset(n_sessions=120, n_tracks=400, random_seed=3), tmp_path / "data" / "fake")
    cfg_path = write_config(
        tmp_path / "app" / "config.toml",
        """
[data]
tracks = "../data/fake/tracks.parquet"
eval_targets = "../data/fake/eval_targets.parquet"
candidates = "../data/fake/candidates/fake_test_run0.parquet"
""",
    )
    monkeypatch.setenv("MIXNEXT_CONFIG", str(cfg_path))
    return cfg_path


def test_app_shows_a_ten_track_queue_that_changes_when_alpha_moves(fake_config):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    first = at.dataframe[0].value
    assert len(first) == 10

    at.slider[0].set_value(0.0).run()
    assert not at.exception
    assert at.dataframe[0].value["Bài"].tolist() != first["Bài"].tolist()


def test_app_without_data_explains_how_to_create_it(fake_config, tmp_path):
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    (tmp_path / "data" / "fake" / "tracks.parquet").unlink()
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=30).run()
    assert not at.exception
    assert "python -m src.data.fake" in at.error[0].value
