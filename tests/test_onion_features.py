"""Đặc trưng cho bộ last.fm (ticket #56): tông từ HPCP 36 bin gốc La, energy thay thế, bảng tracks."""
import numpy as np
import pandas as pd
import pytest

from src.data.onion import HPCP_MEAN
from src.features.camelot import camelot_code, parse_camelot
from src.data.schema import SchemaError
from src.features.onion import bin_phase, center_offset, energy_proxy, estimate_keys, hpcp_chroma, main, make_lastfm_tracks, validate_lastfm_tracks
from src.rerank.mix_rule import RuleMix

C, D, E, F, G, A, B = 0, 2, 4, 5, 7, 9, 11  # pitch class, 0 = Đô (như cột key của MPD)


def scale_chroma(tonic: int, third: int, fifth: int, others: list[int]) -> np.ndarray:
    """Chroma 12 nốt (0 = Đô): nốt chủ 2, hai nốt còn lại của hợp âm chủ 1,5, các nốt khác trong thang 1."""
    chroma = np.zeros(12)
    chroma[others] = 1.0
    chroma[[third, fifth]] = 1.5
    chroma[tonic] = 2.0
    return chroma


C_MAJOR = scale_chroma(C, E, G, [D, F, A, B])
A_MINOR = scale_chroma(A, C, E, [B, D, F, G])


def to_hpcp36(chroma: np.ndarray, offset: int = 0) -> np.ndarray:
    """Chroma 0 = Đô → HPCP 36 bin của Essentia: bin 0 nằm đúng tâm nốt La, nốt s (tính từ La) ở bin 3s.

    offset ±1 dời năng lượng lệch ⅓ cung (lên dây hơi cao/thấp), vẫn phải ra cùng nốt.
    """
    hpcp = np.zeros(36)
    for pitch, value in enumerate(chroma):
        from_a = (pitch - A) % 12
        hpcp[(3 * from_a + offset) % 36] += value
    return hpcp


def test_a_c_major_profile_is_c_major_8b_and_an_a_minor_profile_is_a_minor_8a():
    keys, modes, _ = estimate_keys(hpcp_chroma(np.stack([to_hpcp36(C_MAJOR), to_hpcp36(A_MINOR)])))
    assert (keys.tolist(), modes.tolist()) == ([C, A], [1, 0])
    assert [camelot_code(k, m) for k, m in zip(keys, modes)] == ["8B", "8A"]


@pytest.mark.parametrize("k", range(12))
def test_rotating_the_chroma_by_k_semitones_moves_the_key_by_k(k):
    keys, modes, _ = estimate_keys(hpcp_chroma(np.stack([to_hpcp36(np.roll(C_MAJOR, k)), to_hpcp36(np.roll(A_MINOR, k))])))
    assert keys.tolist() == [(C + k) % 12, (A + k) % 12] and modes.tolist() == [1, 0]


@pytest.mark.parametrize("offset", [-1, 1])
def test_energy_a_third_of_a_semitone_off_still_counts_for_the_same_note(offset):
    # Gộp bin 3s−1, 3s, 3s+1 (tâm ở nốt); gộp 3s, 3s+1, 3s+2 sẽ đẩy năng lượng lệch thấp sang nốt dưới
    keys, modes, _ = estimate_keys(hpcp_chroma(to_hpcp36(C_MAJOR, offset)[None, :]))
    assert (keys.tolist(), modes.tolist()) == ([C], [1])


def test_key_strength_is_the_correlation_with_the_best_profile():
    # Chroma đúng bằng hồ sơ Krumhansl–Kessler trưởng dịch lên Rê: tương quan 1 với Rê trưởng
    kk_major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
    keys, modes, strength = estimate_keys(hpcp_chroma(to_hpcp36(np.roll(kk_major, D))[None, :]))
    assert (keys[0], modes[0]) == (D, 1) and strength[0] == pytest.approx(1.0)


def test_energy_proxy_is_the_loudness_percentile_within_0_and_1():
    loudness = pd.Series([-30.0, -8.0, -14.0, -14.0, -20.0])  # LUFS: càng gần 0 càng to
    # Hạng (trung bình khi hòa) / 5: −30 → 1, −20 → 2, −14 và −14 → (3 + 4) / 2 = 3,5, −8 → 5
    assert energy_proxy(loudness).tolist() == pytest.approx([0.2, 1.0, 0.7, 0.7, 0.4])


def tid(name: str) -> str:
    return name.ljust(16, "x")


T1, T2, T3, T4 = (tid(c) for c in ("t1", "t2", "t3", "t4"))  # T4 chỉ có trong essentia, không có trong phiên


def essentia_rows() -> pd.DataFrame:
    rows = []
    for track, chroma, bpm, first, second, loud, ks in [
        (T1, C_MAJOR, 120.0, 120.0, 60.0, -9.0, 0.7),   # đỉnh nhịp thứ hai = ½ đỉnh thứ nhất → nghi lỗi nhịp
        (T2, A_MINOR, 98.0, 98.0, 112.0, -15.0, 0.6),
        (T3, np.roll(C_MAJOR, D), 128.0, 128.0, 0.0, -12.0, 0.8),  # không có đỉnh thứ hai
        (T4, C_MAJOR, 90.0, 90.0, 45.0, -30.0, 0.5),
    ]:
        row = {"id": track, "rhythm.bpm": bpm, "rhythm.bpm_histogram_first_peak_bpm": first,
               "rhythm.bpm_histogram_second_peak_bpm": second, "lowlevel.loudness_ebu128.integrated": loud,
               "tonal.key_krumhansl.strength": ks}
        row.update(dict(zip(HPCP_MEAN, to_hpcp36(chroma))))
        rows.append(row)
    return pd.DataFrame(rows)


SESSIONS = pd.DataFrame(
    {
        "session_id": [0, 0, 0, 1, 1, 1],
        "user_id": [5, 5, 5, 6, 6, 6],
        "split": ["train", "train", "train", "val", "val", "val"],
        "pos": [0, 1, 2, 0, 1, 2],
        "track_id": [T1, T2, T1, T3, T1, T2],
    }
)


def test_the_tracks_table_covers_session_tracks_with_rule_features_and_train_counts():
    tracks, stats = make_lastfm_tracks(essentia_rows(), SESSIONS)
    assert tracks.columns.tolist() == ["track_id", "bpm", "key", "mode", "camelot", "energy", "key_strength", "n_occurrences"]
    by_id = tracks.set_index("track_id")
    assert sorted(by_id.index) == [T1, T2, T3]  # T4 không có trong phiên
    assert by_id.loc[[T1, T2, T3], "camelot"].tolist() == ["8B", "8A", "10B"]  # Đô trưởng, La thứ, Rê trưởng
    assert by_id.loc[[T1, T2, T3], "n_occurrences"].tolist() == [2, 1, 0]  # chỉ đếm phiên train
    assert by_id.loc[T1, "energy"] == 1.0 and by_id.loc[T2, "energy"] == by_id["energy"].min()
    parse_camelot(tracks["camelot"])  # mã hợp lệ
    # RuleMix chạy thẳng trên bảng này: T2 (La thứ, 8A) cạnh Đô trưởng 8B → tông mix được; T3 Rê trưởng 10B thì không
    mix = RuleMix(tracks).components(T1, [T2, T3])
    assert mix.key.tolist() == [0.8, 0.0]

    assert stats["tracks"] == 3
    for check in ("key_strength_vs_essentia", "check_offset_grouping", "check_essentia_like_36"):
        assert set(stats[check]) == {"pearson", "median_abs_diff"}


def test_cli_writes_tracks_and_a_log_from_the_onion_and_lastfm_outputs(tmp_path, capsys):
    import json

    onion, lastfm = tmp_path / "onion", tmp_path / "lastfm"
    onion.mkdir()
    lastfm.mkdir()
    essentia_rows().to_parquet(onion / "essentia.parquet", index=False)
    SESSIONS.to_parquet(lastfm / "sessions.parquet", index=False)
    main(["--onion", str(onion), "--sessions", str(lastfm / "sessions.parquet"), "--out", str(lastfm)])

    tracks = pd.read_parquet(lastfm / "tracks.parquet")
    assert len(tracks) == 3 and tracks["camelot"].tolist() == ["8B", "8A", "10B"]
    log = json.loads((lastfm / "tracks_log.json").read_text(encoding="utf-8"))
    assert log == json.loads(capsys.readouterr().out)
    assert log["tracks"] == 3 and log["octave_suspect"]["suspect"] == 1
    validate_lastfm_tracks(tracks)


@pytest.mark.parametrize("offset, phase", [(0, 0), (1, 1), (-1, 2)])
def test_bin_phase_finds_where_note_energy_sits_within_each_third_of_a_semitone(offset, phase):
    # Năng lượng ở bin 3s → pha 0 (tâm nốt, đúng gốc bin); lệch +⅓ cung → pha 1; −⅓ cung → pha 2
    assert bin_phase(to_hpcp36(C_MAJOR, offset)[None, :]).tolist() == [phase]


def test_tracks_log_reports_bin_phase_shares_and_both_octave_denominators():
    _, stats = make_lastfm_tracks(essentia_rows(), SESSIONS)
    assert stats["bin_phase"] == {"center_offset_bins": 0.0, "centered_grouping_ok": True, "phase_0": 1.0, "phase_1": 0.0, "phase_2": 0.0}
    assert stats["octave_suspect"] == {"tracks": 3, "tracks_with_second_peak": 2, "suspect": 1, "rate_of_second_peak": 0.5, "rate_of_all": 1 / 3}


def test_validation_rejects_a_camelot_code_that_does_not_match_key_and_mode():
    tracks, _ = make_lastfm_tracks(essentia_rows(), SESSIONS)
    validate_lastfm_tracks(tracks)
    with pytest.raises(SchemaError, match="camelot"):
        validate_lastfm_tracks(tracks.assign(camelot=["8B", "8B", "8B"]))
    with pytest.raises(SchemaError, match="energy"):
        validate_lastfm_tracks(tracks.assign(energy=[0.0, 0.5, 1.0]))


@pytest.mark.parametrize("offset, expected", [(0, 0.0), (1, 1.0), (-1, -1.0)])
def test_center_offset_measures_how_far_note_energy_sits_from_bin_3s(offset, expected):
    # Năng lượng dồn hết ở một pha → độ lệch đúng bằng pha đó (đơn vị bin); |độ lệch| < 0,5 mới giữ cách gộp có tâm
    assert center_offset(to_hpcp36(C_MAJOR, offset)[None, :]) == pytest.approx(expected)


def test_center_offset_between_two_phases_weighs_their_energy():
    # Pha 0 và pha 1 bằng nhau, pha 2 trống → tâm ở giữa: +0,5 bin
    hpcp = to_hpcp36(C_MAJOR, 0) + to_hpcp36(C_MAJOR, 1)
    assert center_offset(hpcp[None, :]) == pytest.approx(0.5)
