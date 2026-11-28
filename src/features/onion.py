"""Đặc trưng cho bộ last.fm (Music4All-Onion): BPM, tông ước lượng từ HPCP, energy thay thế (ticket #56).

Music4All-Onion không có nhãn tông (cột key_* chỉ còn độ mạnh) và không có energy kiểu Spotify, nên nhóm tự ước lượng
để RuleMix, smoothness và điểm mix học chạy được mà không sửa.

Gốc bin HPCP, xác minh trong mã nguồn Essentia (MTG/essentia, commit df27f94), không chọn theo số liệu:
- src/essentia/utils/extractor_music/MusicTonalDescriptors.cpp: tonal.hpcp là HPCP size 36, referenceFrequency = tần số
  lên dây ước tính của bài (≈ 440 Hz, nốt La), cũng là đầu vào của key_krumhansl / key_temperley / key_edma.
- src/algorithms/spectral/hpcp.cpp: pcpBinF = log2(freq / referenceFrequency) · size, nên bin i nằm đúng tâm tần số
  ref · 2^(i/36): bin 0 là La, nốt thứ s tính từ La ở bin 3s, hai bin 3s−1 và 3s+1 lệch ±⅓ cung.
- src/algorithms/tonal/key.cpp: tên tông đánh số từ La ("A", "Bb", "B", "C", …); Key::resize đặt nốt i ở bin 3i rồi
  nội suy về phía nốt kế tiếp (bin 3i+1, 3i+2).

Giới hạn của phép tự kiểm tra:
- Độ mạnh tông là tương quan tốt nhất qua 24 hồ sơ; xoay chroma trọn bán cung chỉ đổi thứ tự hồ sơ, nên so độ mạnh với
  Essentia không phát hiện được nhầm gốc La/Đô. Gốc La dựa vào mã nguồn ở trên.
- Độ mạnh của Essentia nội suy hồ sơ một phía (về nốt kế tiếp), dò cả phép dịch ⅓ cung và cộng hòa âm, nên tự nó
  nghiêng về cách gộp 3s..3s+2. Tương quan cao hơn của cách gộp đó không chứng minh gốc bin sai.
- Phép kiểm tra có ngưỡng là center_offset: độ lệch trung bình của tâm nốt so với bin 3s phải trong ±0,5 bin.
  Thử trên 20 bài thật (khi làm #54): +0,31 bin, tức tâm nốt hơi cao hơn bin 3s nhưng vẫn thuộc cách gộp có tâm.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.onion import HPCP_MEAN
from src.data.schema import SchemaError
from src.features.camelot import camelot_code

HPCP_BINS = 36
BINS_PER_SEMITONE = HPCP_BINS // 12
A_PITCH_CLASS = 9  # nốt La khi 0 = Đô (quy ước cột key của MPD và camelot_code)
# Hồ sơ Krumhansl–Kessler, nốt chủ ở vị trí 0 (trùng bảng "krumhansl" trong key.cpp của Essentia)
KK_MAJOR = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
KK_MINOR = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
OCTAVE_TOLERANCE = 0.05  # đỉnh nhịp thứ hai trong ±5% của 2× hoặc ½ đỉnh thứ nhất → nghi lỗi nhịp gấp đôi/một nửa
TRACK_COLUMNS = ["track_id", "bpm", "key", "mode", "camelot", "energy", "key_strength", "n_occurrences"]
ONION_TRACK_ID = r"^[A-Za-z0-9]{16}$"


def _zscore(rows: np.ndarray) -> np.ndarray:
    """Chuẩn hóa z theo hàng; hàng phẳng (độ lệch chuẩn 0) thành 0."""
    std = rows.std(axis=1, keepdims=True)
    return (rows - rows.mean(axis=1, keepdims=True)) / np.where(std == 0, 1.0, std)


def hpcp_chroma(hpcp: np.ndarray, centered: bool = True) -> np.ndarray:
    """HPCP 36 bin gốc La (n, 36) → chroma 12 nốt với 0 = Đô (n, 12).

    centered=True: nốt s (tính từ La) gom ba bin 3s−1, 3s, 3s+1, tâm ở nốt (cách dùng).
    centered=False: gom 3s, 3s+1, 3s+2, lệch ⅓ cung (chỉ để đối chứng trong thống kê tự kiểm tra).
    """
    hpcp = np.asarray(hpcp, dtype=float)
    if hpcp.ndim != 2 or hpcp.shape[1] != HPCP_BINS:
        raise ValueError(f"HPCP phải có dạng (n, {HPCP_BINS}), đang là {hpcp.shape}")
    if centered:
        hpcp = np.roll(hpcp, 1, axis=1)  # cột 3s giờ là bin gốc 3s−1, nên nhóm [3s, 3s+3) = bin gốc 3s−1, 3s, 3s+1
    from_a = hpcp.reshape(len(hpcp), 12, BINS_PER_SEMITONE).sum(axis=2)
    return np.roll(from_a, A_PITCH_CLASS, axis=1)  # cột Đô = nốt thứ 3 tính từ La


def bin_phase(hpcp: np.ndarray) -> np.ndarray:
    """Pha (0, 1, 2) có tổng năng lượng HPCP lớn nhất, theo chỉ số bin chia 3 dư bao nhiêu.

    Bin 3s là tâm nốt thì năng lượng dồn ở pha 0; HPCP lệch +⅓ cung dồn ở pha 1, lệch −⅓ cung ở pha 2.
    """
    hpcp = np.asarray(hpcp, dtype=float)
    return hpcp.reshape(len(hpcp), 12, BINS_PER_SEMITONE).sum(axis=1).argmax(axis=1)


def _profiles() -> np.ndarray:
    """24 hồ sơ đã chuẩn hóa z: hàng t là trưởng có nốt chủ t, hàng 12 + t là thứ có nốt chủ t."""
    return _zscore(np.array([np.roll(KK_MAJOR, t) for t in range(12)] + [np.roll(KK_MINOR, t) for t in range(12)]))


def estimate_keys(chroma: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Krumhansl–Schmuckler: tương quan Pearson của chroma với 24 hồ sơ → (key 0–11, mode 1 trưởng / 0 thứ, độ mạnh).

    Độ mạnh là tương quan lớn nhất, trong [−1, 1]. Chroma phẳng (độ lệch chuẩn 0) không có tông: độ mạnh 0, key 0,
    mode 1 (tức Đô trưởng 8B; đếm trong thống kê).
    """
    chroma = np.asarray(chroma, dtype=float)
    flat = chroma.std(axis=1) == 0
    corr = _zscore(chroma) @ _profiles().T / 12
    best = corr.argmax(axis=1)
    strength = np.where(flat, 0.0, corr[np.arange(len(corr)), best])
    keys = np.where(flat, 0, best % 12)
    modes = np.where(flat, 1, (best < 12).astype(int))
    return keys.astype(int), modes.astype(int), strength


def _essentia_like_strength(hpcp: np.ndarray) -> np.ndarray:
    """Độ mạnh tông kiểu Key của Essentia trên 36 bin (hồ sơ nội suy như Key::resize, dò 36 phép dịch), không cộng hòa âm.

    Chỉ dùng để đối chứng trong thống kê tự kiểm tra.
    """
    def resize36(profile: np.ndarray) -> np.ndarray:
        nxt = np.roll(profile, -1)
        return np.concatenate([[p, p - (p - q) / 3, p - 2 * (p - q) / 3] for p, q in zip(profile, nxt)])

    profiles = _zscore(np.array([np.roll(resize36(p), shift) for p in (KK_MAJOR, KK_MINOR) for shift in range(HPCP_BINS)]))
    return (_zscore(hpcp) @ profiles.T / HPCP_BINS).max(axis=1)


def _agreement(ours: np.ndarray, essentia: np.ndarray) -> dict:
    return {"pearson": float(np.corrcoef(ours, essentia)[0, 1]), "median_abs_diff": float(np.median(np.abs(ours - essentia)))}


def energy_proxy(loudness: pd.Series) -> pd.Series:
    """energy thay thế = hạng phần trăm của loudness EBU R128 (LUFS) trong tập bài, trong (0, 1]; hòa thì lấy hạng trung bình.

    Không cùng nghĩa với energy của Spotify (cảm nhận cường độ, nhịp, âm sắc): chỉ giữ thứ tự to/nhỏ, nên |Δenergy|
    giữa hai bài là khoảng cách theo hạng chứ không theo độ ồn tuyệt đối.
    """
    return loudness.rank(pct=True, method="average")


def _octave_suspect(first: pd.Series, second: pd.Series) -> dict:
    """Bài có đỉnh nhịp thứ hai ≈ 2× hoặc ½ đỉnh thứ nhất; báo tỉ lệ trên số bài có đỉnh thứ hai và trên mọi bài."""
    has_second = second > 0
    ratio = second[has_second] / first[has_second]
    suspect = int((((ratio - 2).abs() <= 2 * OCTAVE_TOLERANCE) | ((ratio - 0.5).abs() <= 0.5 * OCTAVE_TOLERANCE)).sum())
    n_second = int(has_second.sum())
    return {
        "tracks": int(len(first)),
        "tracks_with_second_peak": n_second,
        "suspect": suspect,
        "rate_of_second_peak": suspect / n_second if n_second else 0.0,
        "rate_of_all": suspect / len(first) if len(first) else 0.0,
    }


def center_offset(hpcp: np.ndarray) -> float:
    """Độ lệch trung bình (đơn vị bin, trong (−1,5; 1,5]) của tâm nốt so với bin 3s: trung bình vòng tròn của năng lượng
    HPCP theo ba pha (chỉ số bin chia 3 dư 0, 1, 2 ứng với góc 0°, 120°, 240°).

    Cách gộp có tâm (3s−1, 3s, 3s+1) đúng khi |độ lệch| < 0,5 bin, vì ranh giới với cách gộp lệch (3s..3s+2) nằm ở 0,5.
    """
    energy = np.asarray(hpcp, dtype=float).reshape(len(hpcp), 12, BINS_PER_SEMITONE).sum(axis=(0, 1))
    angles = 2 * np.pi * np.arange(BINS_PER_SEMITONE) / BINS_PER_SEMITONE
    angle = np.arctan2((energy * np.sin(angles)).sum(), (energy * np.cos(angles)).sum())
    return float(angle / (2 * np.pi) * BINS_PER_SEMITONE)


def _phase_check(hpcp: np.ndarray) -> dict:
    """Phép kiểm tra gốc bin có ngưỡng (#56): giữ cách gộp có tâm khi |center_offset| < 0,5 bin; kèm tỉ lệ bài theo pha."""
    shares = np.bincount(bin_phase(hpcp), minlength=BINS_PER_SEMITONE) / max(len(hpcp), 1)
    offset = center_offset(hpcp)
    return {
        "center_offset_bins": offset,
        "centered_grouping_ok": bool(abs(offset) < 0.5),
        "phase_0": float(shares[0]),
        "phase_1": float(shares[1]),
        "phase_2": float(shares[2]),
    }


def validate_lastfm_tracks(tracks: pd.DataFrame) -> None:
    """Kiểm tra lastfm/tracks.parquet theo docs/data-contract.md (mục Bộ phụ last.fm)."""
    if tracks.columns.tolist() != TRACK_COLUMNS:
        raise SchemaError("lastfm_tracks", [f"cột phải là {TRACK_COLUMNS}, đang là {tracks.columns.tolist()}"])
    problems = []
    if not tracks["track_id"].astype(str).str.fullmatch(ONION_TRACK_ID).all() or tracks["track_id"].duplicated().any():
        problems.append("track_id phải là ID 16 ký tự của Music4All-Onion, không trùng")
    if not (np.isfinite(tracks["bpm"]).all() and (tracks["bpm"] > 0).all()):
        problems.append("bpm phải hữu hạn và > 0")
    if not (tracks["key"].between(0, 11).all() and tracks["mode"].isin([0, 1]).all()):
        problems.append("key phải trong 0–11, mode 0 hoặc 1")
    expected = [camelot_code(k, m) for k, m in zip(tracks["key"], tracks["mode"])]
    if tracks["camelot"].tolist() != expected:
        problems.append("camelot phải bằng camelot_code(key, mode)")
    if not ((tracks["energy"] > 0) & (tracks["energy"] <= 1)).all():
        problems.append("energy phải trong (0, 1]")
    if not tracks["key_strength"].between(-1, 1).all():
        problems.append("key_strength phải trong [−1, 1]")
    if (tracks["n_occurrences"] < 0).any():
        problems.append("n_occurrences phải ≥ 0")
    if problems:
        raise SchemaError("lastfm_tracks", problems)


def make_lastfm_tracks(essentia: pd.DataFrame, sessions: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Bảng bài của bộ last.fm (mọi bài có trong sessions) với ba đặc trưng của luật, kèm thống kê tự kiểm tra.

    bpm = rhythm.bpm; key, mode, camelot, key_strength từ HPCP (estimate_keys); energy = energy_proxy(loudness);
    n_occurrences = số lượt trong phiên train. Thống kê: pha năng lượng HPCP (bin_phase, có ngưỡng), tương quan
    key_strength với tonal.key_krumhansl.strength kèm hai đối chứng, tỉ lệ nghi lỗi nhịp, số bài trưởng/thứ/phẳng.
    """
    ids = np.sort(sessions["track_id"].unique())
    ess = essentia.set_index("id")
    missing = pd.Index(ids).difference(ess.index)
    if len(missing):
        raise ValueError(f"{len(missing)} bài trong phiên không có trong essentia (ví dụ {list(missing[:3])})")
    ess = ess.loc[ids]
    hpcp = ess[list(HPCP_MEAN)].to_numpy(dtype=float)

    keys, modes, strength = estimate_keys(hpcp_chroma(hpcp))
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
    validate_lastfm_tracks(tracks)

    essentia_strength = ess["tonal.key_krumhansl.strength"].to_numpy(dtype=float)
    stats = {
        "tracks": int(len(tracks)),
        "bin_phase": _phase_check(hpcp),
        "key_strength_vs_essentia": _agreement(strength, essentia_strength),
        # Đối chứng, không dùng để chọn (xem docstring đầu file): gộp lệch ⅓ cung, và độ mạnh kiểu Essentia 36 bin
        "check_offset_grouping": _agreement(estimate_keys(hpcp_chroma(hpcp, centered=False))[2], essentia_strength),
        "check_essentia_like_36": _agreement(_essentia_like_strength(hpcp), essentia_strength),
        "octave_suspect": _octave_suspect(ess["rhythm.bpm_histogram_first_peak_bpm"], ess["rhythm.bpm_histogram_second_peak_bpm"]),
        "modes": {"major": int((modes == 1).sum()), "minor": int((modes == 0).sum()), "flat": int((hpcp.std(axis=1) == 0).sum())},
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
