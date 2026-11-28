"""App MixNext (Demo v1, #21): logic phiên trong app/player.py, config, và giao diện qua AppTest."""
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("tomllib", reason="app cần Python 3.11+ (đọc config bằng tomllib)")

from app.config import load_config  # noqa: E402
from app.player import Player  # noqa: E402
from src.data.fake import make_fake_dataset  # noqa: E402
from src.export_artifacts import build_artifacts, demo_playlists, write_artifacts  # noqa: E402
from src.rerank.explain import explain  # noqa: E402

DECAY = 0.7


@pytest.fixture(scope="module")
def artifacts():
    fake = make_fake_dataset(n_sessions=200, n_tracks=400, random_seed=7)
    vectors, index = fake.embeddings["fake"]
    demo = demo_playlists(fake.eval_targets, n=10)
    return build_artifacts(fake.tracks, vectors, index["track_id"], model="fake", decay=DECAY, demo=demo)


@pytest.fixture
def player(artifacts):
    return Player(artifacts)


def first_playlist(player):
    return player.start(player.playlists[0])


def test_a_session_starts_from_the_seed_with_the_last_seed_track_playing(player, artifacts):
    session = first_playlist(player)
    seed = artifacts.demo.set_index("session_id").loc[player.playlists[0], "seed_track_ids"]
    assert list(session.tracks) == list(seed) and session.chosen == ()
    assert player.now_playing(session)["track_id"] == seed[-1]
    history = player.history(session)
    assert history["track_id"].tolist() == list(seed) and (history["source"] == "seed").all()


def test_the_queue_has_ten_tracks_outside_the_session_with_scores_and_explanations(player):
    session = first_playlist(player)
    queue = player.queue(session, alpha=0.7)
    assert len(queue) == 10 and not set(queue["track_id"]) & set(session.tracks)
    assert (np.diff(queue["final"]) <= 0).all()
    playing = player.now_playing(session)
    assert queue["explanation"].tolist() == [explain(playing, row) for _, row in queue.iterrows()]


def test_choosing_a_queued_track_appends_it_plays_it_and_recomputes_the_queue(player):
    start = first_playlist(player)
    before = player.queue(start, alpha=0.7)
    pick = before["track_id"].iloc[3]
    after = player.choose(start, pick, alpha=0.7)

    assert after.tracks == start.tracks + (pick,) and after.chosen == (pick,)
    assert player.now_playing(after)["track_id"] == pick
    history = player.history(after)
    assert history["track_id"].iloc[-1] == pick and history["source"].iloc[-1] == "chọn"
    assert history["explanation"].iloc[-1] == explain(player.now_playing(start), player.now_playing(after))
    queue = player.queue(after, alpha=0.7)
    assert not set(queue["track_id"]) & set(after.tracks)  # seed và bài đã chọn không quay lại
    assert queue["track_id"].tolist() != before["track_id"].tolist()


def test_only_a_track_from_the_current_queue_can_be_chosen(player):
    session = first_playlist(player)
    outside = sorted(set(player.tracks_with_vectors) - set(player.queue(session, alpha=0.7)["track_id"]) - set(session.tracks))[0]
    with pytest.raises(ValueError, match="hàng đợi"):
        player.choose(session, outside, alpha=0.7)


def test_back_undoes_the_last_choice_and_restart_returns_to_the_seed(player):
    start = first_playlist(player)
    one = player.choose(start, player.queue(start, 0.7)["track_id"].iloc[0], 0.7)
    two = player.choose(one, player.queue(one, 0.7)["track_id"].iloc[0], 0.7)
    assert player.back(two) == one and player.back(one) == start
    assert player.back(start) == start  # đang ở seed: không lùi được nữa
    assert player.restart(two) == start


def test_alpha_only_reorders_the_same_candidates(player):
    session = first_playlist(player)
    by_taste, by_mix = player.queue(session, alpha=1.0, k=None), player.queue(session, alpha=0.0, k=None)
    assert set(by_taste["track_id"]) == set(by_mix["track_id"]) and len(by_taste) == 100
    assert by_taste["track_id"].tolist() != by_mix["track_id"].tolist()


def test_choosing_and_moving_alpha_take_under_a_second_on_a_full_size_catalog(artifacts):
    # Danh mục cỡ artifacts thật: khoảng 130 nghìn bài, vector 256 chiều
    rng = np.random.default_rng(0)
    n, dim = 130_000, 256
    ids = [f"{i:022d}" for i in range(n)]
    big_tracks = pd.DataFrame({
        "track_id": ids, "track_name": "x", "artist_name": "y", "tempo": 120.0, "bpm": rng.uniform(70, 180, n),
        "key": 0, "mode": 1, "camelot": rng.choice(["8A", "9A", "3B"], n), "energy": rng.uniform(0, 1, n), "n_occurrences": 1,
    })
    demo = pd.DataFrame({"session_id": [1], "seed_track_ids": [ids[:5]], "target_track_id": [ids[5]]})
    big = build_artifacts(big_tracks, rng.normal(size=(n, dim)).astype(np.float32), ids, "big", DECAY, demo)
    player = Player(big)
    session = player.start(1)
    player.queue(session, 0.7)  # lần đầu: tính ứng viên
    t = time.perf_counter()
    player.queue(session, 0.3)
    assert time.perf_counter() - t < 1.0  # kéo α
    t = time.perf_counter()
    after = player.choose(session, player.queue(session, 0.3)["track_id"].iloc[0], 0.3)
    player.queue(after, 0.3)
    player.history(after)
    assert time.perf_counter() - t < 1.0  # chọn bài: phiên, lịch sử, hàng đợi mới


def test_the_config_points_at_one_artifacts_folder_relative_to_the_config_file(tmp_path, monkeypatch):
    cfg_path = tmp_path / "app" / "config.toml"
    cfg_path.parent.mkdir()
    cfg_path.write_text("[data]\nartifacts = '../artifacts'\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    cfg = load_config(cfg_path)
    assert cfg.artifacts == (tmp_path / "artifacts").resolve()
    assert {p.name for p in cfg.missing_files()} >= {"manifest.json", "tracks.parquet"}


def test_a_missing_data_path_names_the_key_and_the_config_file(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text("[data]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="artifacts") as err:
        load_config(cfg_path)
    assert "config.toml" in str(err.value)


def test_player_reads_an_artifacts_folder(artifacts, tmp_path):
    write_artifacts(artifacts, tmp_path)
    player = Player.from_dir(tmp_path)
    assert player.playlists == sorted(artifacts.demo["session_id"].tolist())
    assert player.default_alpha == artifacts.mix_params.rule_alpha


APP = Path(__file__).resolve().parents[1] / "app" / "main.py"


@pytest.fixture
def app_config(artifacts, tmp_path, monkeypatch):
    """artifacts/ giả ghi ra thư mục tạm, kèm config trỏ tới; app đọc config qua MIXNEXT_CONFIG."""
    write_artifacts(artifacts, tmp_path / "artifacts")
    cfg_path = tmp_path / "app" / "config.toml"
    cfg_path.parent.mkdir()
    cfg_path.write_text("[data]\nartifacts = '../artifacts'\n", encoding="utf-8")
    monkeypatch.setenv("MIXNEXT_CONFIG", str(cfg_path))
    return tmp_path


def run_app():
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")
    at = streamlit_testing.AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception, at.exception
    return at


def queue_ids(at) -> list[str]:
    return [b.key.removeprefix("choose_") for b in at.button if b.key and b.key.startswith("choose_")]


def test_choosing_from_the_queue_updates_now_playing_history_and_queue(app_config):
    at = run_app()
    first = queue_ids(at)
    assert len(first) == 10
    at.button(key=f"choose_{first[2]}").click().run()
    assert not at.exception
    assert at.session_state["session"].current == first[2]
    assert first[2] not in queue_ids(at) and len(queue_ids(at)) == 10
    assert any(first[2] in str(df.value["track_id"].tolist()) for df in at.dataframe)  # bảng lịch sử có bài vừa chọn

    at.button(key="back").click().run()
    assert at.session_state["session"].chosen == ()
    at.button(key=f"choose_{first[0]}").click().run()
    at.button(key="restart").click().run()
    assert at.session_state["session"].chosen == () and queue_ids(at) == first


def test_moving_alpha_reorders_the_queue(app_config):
    at = run_app()
    before = queue_ids(at)
    at.slider(key="alpha").set_value(0.0).run()
    assert not at.exception and queue_ids(at) != before


def test_missing_artifacts_explain_how_to_build_them(app_config):
    (app_config / "artifacts" / "manifest.json").unlink()
    at = run_app()
    assert "python -m src.export_artifacts" in at.error[0].value


def test_a_broken_config_shows_an_error_instead_of_a_traceback(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXNEXT_CONFIG", str(tmp_path / "missing.toml"))
    at = run_app()
    assert "missing.toml" in at.error[0].value
