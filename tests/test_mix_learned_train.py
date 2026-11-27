import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch", reason="huấn luyện điểm mix học cần PyTorch")

import torch  # noqa: E402

from src.data.fake import make_fake_dataset, write_fake  # noqa: E402
from src.features.camelot import camelot_code  # noqa: E402
from src.rerank.mix_learned import LearnedMix, consecutive_pairs  # noqa: E402
from src.rerank.mix_learned_train import MixTrainParams, main, train  # noqa: E402

FAST = MixTrainParams(hidden=(32,), dim=16, batch_size=128, max_epochs=6, patience=3, lr=3e-3)


def smooth_playlists(n_tracks: int = 400, n_sessions: int = 300, length: int = 20, seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Danh mục ngẫu nhiên; mỗi bài kế tiếp được chọn trong các bài cùng hoặc kề tông Camelot và lệch BPM ≤ 6%.

    Phiên 0–239 là train, còn lại là val. Cặp liền nhau vì vậy mang đúng tín hiệu "mix được" mà mô hình phải học.
    """
    rng = np.random.default_rng(seed)
    key, mode = rng.integers(0, 12, n_tracks), rng.integers(0, 2, n_tracks)
    camelot = [camelot_code(int(k), int(m)) for k, m in zip(key, mode)]
    num = np.array([int(c[:-1]) for c in camelot])
    letter = np.array([c[-1] for c in camelot])
    bpm = rng.uniform(70, 180, n_tracks)
    ids = [f"t{i:021d}" for i in range(n_tracks)]
    tracks = pd.DataFrame({"track_id": ids, "bpm": bpm, "camelot": camelot, "energy": rng.uniform(0, 1, n_tracks)})

    step = np.abs(num[:, None] - num[None, :]) % 12
    step = np.minimum(step, 12 - step)
    same_letter = letter[:, None] == letter[None, :]
    key_ok = (same_letter & (step <= 1)) | (~same_letter & (step == 0))
    bpm_ok = np.abs(bpm[None, :] - bpm[:, None]) / bpm[:, None] <= 0.06
    nxt = key_ok & bpm_ok
    np.fill_diagonal(nxt, False)

    rows = []
    for sid in range(n_sessions):
        t = int(rng.integers(n_tracks))
        for pos in range(length):
            rows.append({"session_id": sid, "split": "train" if sid < 240 else "val", "pos": pos, "track_id": ids[t]})
            options = np.flatnonzero(nxt[t])
            t = int(rng.choice(options)) if len(options) else int(rng.integers(n_tracks))
    return tracks, pd.DataFrame(rows)


@pytest.fixture(scope="module")
def trained():
    tracks, sessions = smooth_playlists()
    return tracks, sessions, train(tracks, sessions, FAST, random_seed=0)


def test_val_loss_goes_down_from_the_untrained_encoders(trained):
    _, _, result = trained
    untrained, *epochs = result.history
    assert untrained["epoch"] == 0
    assert min(e["val_loss"] for e in epochs) < 0.9 * untrained["val_loss"]


def test_in_batch_recall_at_10_on_val_beats_random(trained):
    _, _, result = trained
    best = result.best_epoch_stats
    assert best["val_recall_at_10"] > 2 * best["random_recall_at_10"]


def test_learns_only_from_train_pairs(trained):
    _, sessions, result = trained
    assert result.n_train_pairs == len(consecutive_pairs(sessions, "train"))
    assert result.n_val_pairs == len(consecutive_pairs(sessions, "val"))


def test_exported_numpy_encoders_score_like_the_torch_model(trained):
    tracks, _, result = trained
    ids = tracks["track_id"].tolist()
    mix = LearnedMix(tracks, result.encoders)
    with torch.no_grad():
        torch_scores = result.model.s_learn_rows(torch.tensor([3]), torch.arange(len(ids))).numpy()[0]
    np.testing.assert_allclose(mix.s_learn(ids[3], ids), torch_scores, atol=1e-5)


def test_learned_mix_prefers_a_mixable_next_track(trained):
    tracks, _, result = trained
    tracks = tracks.set_index("track_id")
    a = tracks.index[0]
    # cùng tông và cùng BPM với a, so với bài lệch nửa vòng Camelot và lệch BPM 25%
    twin = tracks.loc[[a]].assign(energy=0.5).rename(index={a: "twin".ljust(22, "x")})
    far_num = (int(tracks.loc[a, "camelot"][:-1]) + 5) % 12 + 1
    far = twin.assign(camelot=f"{far_num}{tracks.loc[a, 'camelot'][-1]}", bpm=tracks.loc[a, "bpm"] * 1.25).rename(index=lambda _: "far".ljust(22, "x"))
    catalog = pd.concat([tracks, twin, far]).rename_axis("track_id").reset_index()
    scores = LearnedMix(catalog, result.encoders)(a, ["twin".ljust(22, "x"), "far".ljust(22, "x")])
    assert scores[0] > scores[1]


def test_cli_writes_npz_weights_and_a_json_report_that_learned_mix_loads(tmp_path):
    import json

    data, out = tmp_path / "data", tmp_path / "out"
    write_fake(make_fake_dataset(), data)
    report = main(["--data", str(data), "--out", str(out), "--max-epochs", "2"])
    tracks = pd.read_parquet(data / "tracks.parquet")
    mix = LearnedMix.load(tracks, out / "mix_learned.npz")
    ids = tracks["track_id"].tolist()[:5]
    scores = mix(ids[0], ids)
    assert scores.shape == (5,) and ((0 <= scores) & (scores <= 1)).all()
    saved = json.loads((out / "mix_learned.json").read_text(encoding="utf-8"))
    assert saved == report
    assert [h["epoch"] for h in saved["history"]][:1] == [0] and saved["n_train_pairs"] > 0
