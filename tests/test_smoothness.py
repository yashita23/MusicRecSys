import math

import pandas as pd
import pytest

from src.eval.smoothness import smoothness
from src.features.track_features import TrackFeatures

A, B, C, D = "a" * 22, "b" * 22, "c" * 22, "d" * 22


def features(camelot=("8A", "8B", "3A", "9A")) -> TrackFeatures:
    return TrackFeatures.from_tracks(
        pd.DataFrame(
            {
                "track_id": [A, B, C, D],
                "bpm": [120.0, 127.2, 60.0, 130.8],
                "camelot": list(camelot),
                "energy": [0.7, 0.5, 0.9, 0.9],
            }
        )
    )


def test_smoothness_pools_every_adjacent_pair_of_every_chain():
    # Chuỗi 1: A→B lệch đúng 6%, 8A→8B hợp, |Δe| 0,2 → cặp mix được
    #          B→C: 60×2 = 120 so với 127,2 → lệch 5,66%; 8B→3A không hợp; |Δe| 0,4
    # Chuỗi 2: C→D: 130,8×0,5 = 65,4 so với 60 → lệch 9%; 3A→9A không hợp; |Δe| 0
    s = smoothness([[A, B, C], [C, D]], features())
    assert s.n_pairs == 3
    assert s.bpm_delta_pct == pytest.approx((6 + 5.6604 + 9) / 3, abs=1e-3)
    assert s.key_compatible_pct == pytest.approx(100 / 3)
    assert s.energy_gap == pytest.approx(0.2)
    assert s.mixable_pct == pytest.approx(100 / 3)


@pytest.mark.parametrize(
    "camelot_a, camelot_b, compatible",
    [("12A", "1A", True), ("8A", "9A", True), ("8A", "7A", True), ("8A", "9B", False), ("8A", "10A", False)],
)
def test_key_compatibility_through_smoothness(camelot_a, camelot_b, compatible):
    s = smoothness([[A, B]], features(camelot=(camelot_a, camelot_b, "3A", "9A")))
    assert s.key_compatible_pct == (100.0 if compatible else 0.0)
    assert s.mixable_pct == (100.0 if compatible else 0.0)  # A→B lệch đúng 6% nên chỉ còn phụ thuộc tông


def test_chain_of_one_track_has_no_pairs_and_nan_metrics():
    s = smoothness([[A]], features())
    assert s.n_pairs == 0
    assert all(math.isnan(v) for v in (s.bpm_delta_pct, s.key_compatible_pct, s.energy_gap, s.mixable_pct))
