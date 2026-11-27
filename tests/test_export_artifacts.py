import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.data.fake import make_fake_dataset
from src.export_artifacts import build_artifacts, demo_playlists, load_artifacts, write_artifacts
from src.models.popularity import Popularity
from src.models.session import SessionVectorTaste
from src.rerank.mix_learned import FEATURE_NAMES, MixEncoders

DECAY = 0.6
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def fake():
    return make_fake_dataset(n_sessions=200, n_tracks=400, random_seed=7)


def export(fake, **kwargs):
    vectors, index = fake.embeddings["fake"]
    demo = demo_playlists(fake.eval_targets, n=10)
    return build_artifacts(fake.tracks, vectors, index["track_id"], model="fake", decay=DECAY, demo=demo, **kwargs)


def test_loaded_artifacts_give_the_same_candidates_as_the_original_model(fake, tmp_path):
    write_artifacts(export(fake), tmp_path)
    loaded = load_artifacts(tmp_path).taste_model()

    vectors, index = fake.embeddings["fake"]
    original = SessionVectorTaste("fake", vectors, index["track_id"].tolist(), DECAY, Popularity().fit(fake.sessions))
    no_vector = sorted(set(fake.tracks["track_id"]) - set(index["track_id"]))[:3]
    assert no_vector, "dữ liệu giả phải có bài không có vector để thử nhánh dự phòng Popularity"
    sessions = [list(seed) for seed in fake.eval_targets["seed_track_ids"][:20]] + [no_vector]
    for session in sessions:
        ids, scores = loaded.candidates(session)
        expected_ids, expected_scores = original.candidates(session)
        assert ids == expected_ids
        np.testing.assert_allclose(scores, expected_scores, rtol=1e-6)


def test_cutting_the_catalog_keeps_demo_tracks_and_the_most_played_tracks(fake):
    full = export(fake)
    cut = export(fake, max_bytes=full.nbytes // 4)
    assert cut.nbytes <= full.nbytes // 4
    assert len(cut.track_index) < len(full.track_index)

    demo_tracks = {t for seed in cut.demo["seed_track_ids"] for t in seed} | set(cut.demo["target_track_id"])
    assert demo_tracks <= set(cut.tracks["track_id"])
    assert demo_tracks & set(full.track_index["track_id"]) <= set(cut.track_index["track_id"])  # bài demo có vector giữ vector
    assert set(cut.track_index["track_id"]) <= set(cut.tracks["track_id"])

    full_rows = full.track_index.set_index("track_id")["row"]
    np.testing.assert_array_equal(cut.vectors, full.vectors[full_rows[cut.track_index["track_id"]].to_numpy()])

    plays = fake.tracks.set_index("track_id")["n_occurrences"]
    kept = set(cut.track_index["track_id"]) - demo_tracks
    dropped = set(full.track_index["track_id"]) - set(cut.track_index["track_id"])
    assert plays[sorted(kept)].min() >= plays[sorted(dropped)].max()


def test_app_loads_artifacts_without_torch(fake, tmp_path):
    rng = np.random.default_rng(0)
    layer = lambda n_out, n_in: (rng.normal(size=(n_out, n_in)).astype(np.float32), np.zeros(n_out, np.float32))  # noqa: E731
    encoders = MixEncoders([layer(8, len(FEATURE_NAMES)), layer(4, 8)], [layer(8, len(FEATURE_NAMES)), layer(4, 8)], 0.1)
    encoders.save(tmp_path / "mix.npz")
    write_artifacts(export(fake), tmp_path / "artifacts", mix_learned=tmp_path / "mix.npz")

    script = (
        "import sys\n"
        "from src.export_artifacts import load_artifacts\n"
        f"a = load_artifacts({str(tmp_path / 'artifacts')!r})\n"
        "seed = list(a.demo['seed_track_ids'].iloc[0])\n"
        "ids, _ = a.taste_model().candidates(seed)\n"
        "a.rule_mix()(seed[-1], ids)\n"
        "a.learned_mix()(seed[-1], ids)\n"
        "assert 'torch' not in sys.modules, 'app không được import torch'\n"
    )
    result = subprocess.run([sys.executable, "-c", script], cwd=REPO, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
