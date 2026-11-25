import numpy as np
import pandas as pd
import pytest

from src.rerank.mix_rule import MixWeights, RuleMix, s_bpm, s_energy, s_key

A, B, C = "a" * 22, "b" * 22, "c" * 22


def tracks() -> pd.DataFrame:
    return pd.DataFrame(
        {"track_id": [A, B, C], "bpm": [120.0, 128.0, 60.0], "camelot": ["8A", "8B", "3A"], "energy": [0.7, 0.5, 0.7]}
    )


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (120.0, 120.0, 1.0),
        (120.0, 60.0, 1.0),      # nhịp một nửa
        (120.0, 240.0, 1.0),     # nhịp gấp đôi
        (120.0, 127.2, 1.0),     # lệch đúng 6%
        (120.0, 128.0, 0.8889),  # lệch 6,67% → (12 − 6,67) / 6
        (120.0, 130.8, 0.5),     # lệch 9%
        (120.0, 134.4, 0.0),     # lệch 12%
        (120.0, 150.0, 0.0),     # lệch 25% (×0,5 → 37,5%, ×2 → 150%)
        (128.0, 64.4, 1.0),      # ×2 = 128,8 → lệch 0,6%
    ],
)
def test_s_bpm_edge_cases(a, b, expected):
    assert s_bpm(np.array([a]), np.array([b]))[0] == pytest.approx(expected, abs=1e-4)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        ("8A", "8A", 1.0),
        ("8A", "8B", 0.8),   # cùng số khác chữ (tương đối trưởng/thứ)
        ("12A", "1A", 0.8),  # vòng quay qua 12 → 1
        ("1B", "12B", 0.8),
        ("8A", "9A", 0.8),
        ("8A", "7A", 0.8),
        ("8A", "9B", 0.0),   # bước chéo: spec chấm 0
        ("8A", "10A", 0.0),
        ("8A", "2B", 0.0),
    ],
)
def test_s_key_on_the_camelot_wheel(a, b, expected):
    assert s_key(np.array([a]), np.array([b]))[0] == pytest.approx(expected)


def test_s_energy_is_one_minus_energy_gap():
    np.testing.assert_allclose(s_energy(np.array([0.7]), np.array([0.2, 0.7, 1.0])), [0.5, 1.0, 0.7])
    assert s_energy(np.array([0.0]), np.array([1.0]))[0] == pytest.approx(0.0)


def test_s_bpm_scores_one_current_track_against_many_candidates():
    scores = s_bpm(np.array([120.0]), np.array([120.0, 130.8, 60.0, 150.0]))
    np.testing.assert_allclose(scores, [1.0, 0.5, 1.0, 0.0], atol=1e-9)


def test_rule_mix_weights_the_three_components():
    mix = RuleMix(tracks())
    # A → B: s_bpm 0,8889, s_key 0,8 (8A→8B), s_energy 0,8 → 0,4·0,8889 + 0,4·0,8 + 0,2·0,8
    # A → C: s_bpm 1 (nhịp một nửa), s_key 0 (8A→3A), s_energy 1 → 0,4 + 0 + 0,2
    np.testing.assert_allclose(mix(A, [A, B, C]), [1.0, 0.83556, 0.6], atol=1e-4)


def test_rule_mix_uses_the_weights_it_is_given():
    bpm_only = RuleMix(tracks(), MixWeights(bpm=1.0, key=0.0, energy=0.0))
    np.testing.assert_allclose(bpm_only(A, [B, C]), [0.8889, 1.0], atol=1e-4)


def test_rule_mix_unknown_track_is_an_error():
    with pytest.raises(KeyError, match="z" * 22):
        RuleMix(tracks())(A, [B, "z" * 22])


def test_scoring_one_track_against_100_candidates_takes_a_few_ms():
    import time

    rng = np.random.default_rng(0)
    n = 130_737  # cỡ danh mục thật sau #5
    ids = [f"{i:022d}" for i in range(n)]
    camelot = [f"{k}{l}" for k, l in zip(rng.integers(1, 13, n), rng.choice(["A", "B"], n))]
    catalog = pd.DataFrame({"track_id": ids, "bpm": rng.uniform(70, 180, n), "camelot": camelot, "energy": rng.uniform(0, 1, n)})
    mix = RuleMix(catalog)
    candidates = [ids[i] for i in rng.choice(n, 100, replace=False)]
    mix(ids[0], candidates)  # làm nóng
    t0 = time.perf_counter()
    for _ in range(200):
        mix(ids[0], candidates)
    mean_ms = (time.perf_counter() - t0) / 200 * 1000
    assert mean_ms < 5, f"{mean_ms:.2f} ms mỗi lần chấm"
