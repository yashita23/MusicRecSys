"""Đặc trưng cho bộ last.fm (Music4All-Onion): BPM, tông ước lượng từ HPCP, energy thay thế (ticket #56).

Music4All-Onion không có nhãn tông (cột key_* chỉ còn độ mạnh) và không có energy kiểu Spotify, nên nhóm tự ước lượng
để RuleMix, smoothness và điểm mix học chạy được mà không sửa.

HPCP của Onion (tonal.hpcp.mean_0..35) do music extractor của Essentia tính: 36 bin, referenceFrequency = tần số lên
dây ước tính của bài (≈ 440 Hz, nốt La). Theo src/algorithms/spectral/hpcp.cpp (MTG/essentia, commit df27f94):
pcpBinF = log2(freq / referenceFrequency) · size, nên bin i nằm đúng tâm tần số ref · 2^(i/36): bin 0 là La, nốt thứ s
tính từ La ở bin 3s, và hai bin 3s−1, 3s+1 lệch ±⅓ cung. Key của Essentia cũng đánh số từ La ("A", "Bb", "B", "C", …).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.onion import HPCP_MEAN
from src.features.camelot import camelot_code

HPCP_BINS = 36
BINS_PER_SEMITONE = HPCP_BINS // 12
A_PITCH_CLASS = 9  # nốt La khi 0 = Đô (quy ước cột key của MPD và camelot_code)
# Hồ sơ Krumhansl–Kessler, nốt chủ ở vị trí 0 (trùng bảng "krumhansl" trong key.cpp của Essentia)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
OCTAVE_TOLERANCE = 0.05  # đỉnh nhịp thứ hai trong ±5% của 2× hoặc ½ đỉnh thứ nhất → nghi lỗi nhịp gấp đôi/một nửa
TRACK_COLUMNS = ["track_id", "bpm", "key", "mode", "camelot", "energy", "key_strength", "n_occurrences"]


def hpcp_chroma(hpcp: np.ndarray) -> np.ndarray:
    """HPCP 36 bin gốc La (n, 36) → chroma 12 nốt với 0 = Đô (n, 12).

    Nốt s (tính từ La) gom ba bin 3s−1, 3s, 3s+1, tâm ở nốt; rồi xoay để chỉ số 0 là Đô.
    """
    hpcp = np.asarray(hpcp, dtype=float)
    if hpcp.ndim != 2 or hpcp.shape[1] != HPCP_BINS:
        raise ValueError(f"HPCP phải có dạng (n, {HPCP_BINS}), đang là {hpcp.shape}")
    centered = np.roll(hpcp, 1, axis=1)  # cột 3s giờ là bin gốc 3s−1, nên nhóm [3s, 3s+3) = bin gốc 3s−1, 3s, 3s+1
    from_a = centered.reshape(len(hpcp), 12, BINS_PER_SEMITONE).sum(axis=2)
    return np.roll(from_a, A_PITCH_CLASS, axis=1)  # cột Đô = nốt thứ 3 tính từ La


def _profiles() -> np.ndarray:
    """24 hồ sơ đã chuẩn hóa z: hàng t là trưởng có nốt chủ t, hàng 12 + t là thứ có nốt chủ t."""
    rows = [np.roll(KK_MAJOR, t) for t in range(12)] + [np.roll(KK_MINOR, t) for t in range(12)]
    profiles = np.array(rows)
    return (profiles - profiles.mean(axis=1, keepdims=True)) / profiles.std(axis=1, keepdims=True)


def estimate_keys(chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Krumhansl–Schmuckler: tương quan Pearson của chroma với 24 hồ sơ → (key 0–11, mode 1 trưởng / 0 thứ, độ mạnh).

    Độ mạnh là tương quan lớn nhất. Chroma phẳng (độ lệch chuẩn 0) không có tông: độ mạnh 0, key 0, mode 1.
    """
    chroma = np.asarray(chroma, dtype=float)
    std = chroma.std(axis=1, keepdims=True)
    flat = std[:, 0] == 0
    z = (chroma - chroma.mean(axis=1, keepdims=True)) / np.where(std == 0, 1.0, std)
    corr = z @ _profiles().T / 12
    best = corr.argmax(axis=1)
    strength = np.where(flat, 0.0, corr[np.arange(len(corr)), best])
    keys = np.where(flat, 0, best % 12)
    modes = np.where(flat, 1, (best < 12).astype(int))
    return keys.astype(int), modes.astype(int), strength


def _essentia_like_strength(hpcp: np.ndarray) -> np.ndarray:
    """Độ mạnh tông kiểu Key của Essentia trên 36 bin (hồ sơ nội suy như Key::resize, dò 36 phép dịch), không cộng hòa âm.

    Chỉ dùng để tự kiểm tra: Essentia còn cộng hòa âm (usePolyphony, useThreeChords) nên không bằng hệt.
    """
    def resize36(profile: np.ndarray) -> np.ndarray:
        nxt = np.roll(profile, -1)
        return np.concatenate([[p, p - (p - q) / 3, p - 2 * (p - q) / 3] for p, q in zip(profile, nxt)])

    profiles = np.array([np.roll(resize36(p), shift) for p in (KK_MAJOR, KK_MINOR) for shift in range(HPCP_BINS)])
    profiles = (profiles - profiles.mean(axis=1, keepdims=True)) / profiles.std(axis=1, keepdims=True)
    std = hpcp.std(axis=1, keepdims=True)
    z = (hpcp - hpcp.mean(axis=1, keepdims=True)) / np.where(std == 0, 1.0, std)
    return (z @ profiles.T / HPCP_BINS).max(axis=1)


def _agreement(ours: np.ndarray, essentia: np.ndarray) -> dict:
    return {"pearson": float(np.corrcoef(ours, essentia)[0, 1]), "median_abs_diff": float(np.median(np.abs(ours - essentia)))}


def energy_proxy(loudness: pd.Series) -> pd.Series:
    """energy thay thế = hạng phần trăm của loudness EBU R128 (LUFS) trong tập bài, trong (0, 1]; bài to nhất = 1.

    Không cùng nghĩa với energy của Spotify (cảm nhận cường độ, nhịp, âm sắc): chỉ giữ thứ tự to/nhỏ, nên |Δenergy|
    giữa hai bài là khoảng cách theo hạng chứ không theo độ ồn tuyệt đối.
    """
    return loudness.rank(pct=True, method="average")


def _octave_suspect(first: pd.Series, second: pd.Series) -> dict:
    has_second = second > 0
    ratio = second[has_second] / first[has_second]
    suspect = ((ratio - 2).abs() <= 2 * OCTAVE_TOLERANCE) | ((ratio - 0.5).abs() <= 0.5 * OCTAVE_TOLERANCE)
    n = int(has_second.sum())
    return {"tracks_with_second_peak": n, "suspect": int(suspect.sum()), "rate": float(suspect.mean()) if n else 0.0}


def make_lastfm_tracks(essentia: pd.DataFrame, sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bảng bài của bộ last.fm (mọi bài có trong sessions) với ba đặc trưng của luật, kèm thống kê tự kiểm tra.

    bpm = rhythm.bpm; key, mode, camelot, key_strength từ HPCP (estimate_keys); energy = energy_proxy(loudness);
    n_occurrences = số lượt trong phiên train. Thống kê: tương quan key_strength với tonal.key_krumhansl.strength của
    Essentia (cùng hồ sơ nhưng Essentia cộng thêm hòa âm, nên chỉ kỳ vọng tương quan cao), tỉ lệ nghi lỗi nhịp.
    """
    ids = np.sort(sessions["track_id"].unique())
    ess = essentia.set_index("id")
    missing = pd.Index(ids).difference(ess.index)
    if len(missing):
        raise ValueError(f"{len(missing)} bài trong phiên không có trong essentia (ví dụ {list(missing[:3])})")
    ess = ess.loc[ids]

    keys, modes, strength = estimate_keys(hpcp_chroma(ess[list(HPCP_MEAN)].to_numpy()))
    train_counts = sessions.loc[sessions["split"] == "train", "track_id"].value_counts()
    tracks = pd.DataFrame(
        {
            "track_id": ids,
            "bpm": ess["rhythm.bpm"].to_numpy(dtype=float),
            "key": keys,
            "mode": modes,
            "camelot": [camelot_code(k, m) for k, m in zip(keys, modes)],
            "energy": energy_proxy(ess["lowlevel.loudness_ebu128.integrated"].reset_index(drop=True)).to_numpy(),
            "key_strength": strength,
            "n_occurrences": pd.Series(ids).map(train_counts).fillna(0).astype("int64").to_numpy(),
        }
    )[TRACK_COLUMNS]
    if not (np.isfinite(tracks["bpm"]).all() and (tracks["bpm"] > 0).all() and tracks["energy"].between(0, 1).all()):
        raise ValueError("bpm phải hữu hạn và > 0, energy trong 0–1")

    essentia_strength = ess["tonal.key_krumhansl.strength"].to_numpy(dtype=float)
    hpcp = ess[list(HPCP_MEAN)].to_numpy(dtype=float)
    offset_chroma = np.roll(hpcp.reshape(len(hpcp), 12, BINS_PER_SEMITONE).sum(axis=2), A_PITCH_CLASS, axis=1)
    stats = {
        "tracks": int(len(tracks)),
        "key_strength_vs_essentia": _agreement(strength, essentia_strength),
        # Đối chứng: gộp lệch ⅓ cung (bin 3s, 3s+1, 3s+2) và độ mạnh kiểu Essentia 36 bin. Gốc bin xác định từ mã nguồn
        # Essentia (docstring đầu file); các số này để báo cáo độ tin cậy của phép tự kiểm tra, không dùng để chọn.
        "check_offset_grouping": _agreement(estimate_keys(offset_chroma)[2], essentia_strength),
        "check_essentia_like_36": _agreement(_essentia_like_strength(hpcp), essentia_strength),
        "octave_suspect": _octave_suspect(ess["rhythm.bpm_histogram_first_peak_bpm"], ess["rhythm.bpm_histogram_second_peak_bpm"]),
        "modes": {"major": int((modes == 1).sum()), "minor": int((modes == 0).sum())},
    }
    return tracks, stats


def main(argv: list[str] | None = None) -> dict:
    """python -m src.features.onion --onion <mixnext-onion> --sessions <sessions.parquet> --out <thư mục ra>

    Ghi tracks.parquet và tracks_log.json (thống kê tự kiểm tra) vào cùng thư mục với sessions (notebook 10).
    """
    ap = argparse.ArgumentParser(description="Đặc trưng cho bộ last.fm: BPM, tông từ HPCP, energy thay thế (#56)")
    ap.add_argument("--onion", required=True, help="Thư mục có essentia.parquet (Dataset mixnext-onion)")
    ap.add_argument("--sessions", required=True, help="sessions.parquet của bộ last.fm (#55)")
    ap.add_argument("--out", required=True, help="Thư mục ra, ví dụ /kaggle/working/out/lastfm")
    args = ap.parse_args(argv)
    essentia = pd.read_parquet(Path(args.onion) / "essentia.parquet")
    sessions = pd.read_parquet(args.sessions, columns=["split", "track_id"])
    tracks, stats = make_lastfm_tracks(essentia, sessions)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tracks.to_parquet(out / "tracks.parquet", index=False)
    (out / "tracks_log.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))
    return stats


if __name__ == "__main__":
    main()
