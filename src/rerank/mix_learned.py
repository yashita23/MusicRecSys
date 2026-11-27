"""Điểm mix học từ các cặp bài liền nhau trong playlist (ticket #23, spec §6).

Hai bộ mã hóa nhỏ f_out (bài đang phát) và f_in (bài kế tiếp) học từ đặc trưng âm thanh, cùng đầu vào với luật
(BPM, tông Camelot, energy), không dùng ID bài nên chấm được cả bài chưa có trong playlist nào. Đầu ra của mỗi bộ
mã hóa được chuẩn hóa L2, nên s_learn(a, b) = f_out(x_a)ᵀ f_in(x_b) nằm trong [−1, 1] và điểm mix = (1 + s_learn) / 2
nằm trong 0–1, cùng thang với gu_norm (combine không chuẩn hóa lại điểm mix).

Module này chỉ dùng numpy để app chạy không cần torch; phần huấn luyện bằng PyTorch ở mix_learned_train.py,
xuất trọng số ra file .npz mà LearnedMix.load đọc.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.features.track_features import TrackFeatures

# Đặc trưng x của một bài. Tempo dùng log2 để nhịp gấp đôi/một nửa cách nhau đúng 1, kèm (cos, sin) của log2(bpm)
# để nhịp gấp đôi/một nửa trùng nhau như ở luật. Tông là góc trên vòng Camelot (12 ↔ 1 liền nhau) và chữ A/B.
FEATURE_NAMES = ("log_tempo", "tempo_cos", "tempo_sin", "key_cos", "key_sin", "major", "energy")
REFERENCE_BPM = 120.0
_EPS = 1e-12  # như torch.nn.functional.normalize

Layer = tuple[np.ndarray, np.ndarray]  # (weight shape (out, in) như torch.nn.Linear, bias shape (out,))


def consecutive_pairs(sessions: pd.DataFrame, split: str) -> pd.DataFrame:
    """Các cặp bài liền nhau (pos, pos + 1) trong các phiên của split, theo session_id rồi pos.

    Một bài lặp lại ngay sau chính nó không phải là một lần chuyển bài nên bị bỏ.
    """
    s = sessions.loc[sessions["split"] == split, ["session_id", "pos", "track_id"]].sort_values(["session_id", "pos"])
    same_session = s["session_id"].to_numpy()[1:] == s["session_id"].to_numpy()[:-1]
    prev, nxt = s["track_id"].to_numpy()[:-1], s["track_id"].to_numpy()[1:]
    keep = same_session & (prev != nxt)
    return pd.DataFrame({"prev_track_id": prev[keep], "next_track_id": nxt[keep]})


def audio_features(features: TrackFeatures) -> np.ndarray:
    """Ma trận đặc trưng float32 shape (số bài, len(FEATURE_NAMES)), theo thứ tự dòng của features."""
    octave = np.log2(features.bpm / REFERENCE_BPM)
    key_angle = 2 * np.pi * features.key_num / 12
    tempo_angle = 2 * np.pi * octave
    columns = {
        "log_tempo": octave,
        "tempo_cos": np.cos(tempo_angle),
        "tempo_sin": np.sin(tempo_angle),
        "key_cos": np.cos(key_angle),
        "key_sin": np.sin(key_angle),
        "major": (features.key_letter == "B").astype(float),
        "energy": features.energy,
    }
    return np.stack([columns[name] for name in FEATURE_NAMES], axis=1).astype(np.float32)


def _mlp(x: np.ndarray, layers: Sequence[Layer]) -> np.ndarray:
    """Linear → ReLU → … → Linear, rồi chuẩn hóa L2 từng dòng (giống MLP trong mix_learned_train.py)."""
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < len(layers) - 1:
            x = np.maximum(x, 0)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _EPS)


@dataclass(frozen=True)
class MixEncoders:
    """Trọng số của f_out và f_in (float32) và nhiệt độ τ đã dùng khi học."""

    f_out: list[Layer]
    f_in: list[Layer]
    temperature: float

    def save(self, path: str | Path) -> None:
        arrays: dict[str, Any] = {"feature_names": np.array(FEATURE_NAMES), "temperature": np.array(self.temperature)}
        for name, layers in (("f_out", self.f_out), ("f_in", self.f_in)):
            for i, (w, b) in enumerate(layers):
                arrays[f"{name}.{i}.weight"] = np.asarray(w, dtype=np.float32)
                arrays[f"{name}.{i}.bias"] = np.asarray(b, dtype=np.float32)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str | Path) -> MixEncoders:
        with np.load(path) as z:
            names = tuple(z["feature_names"].tolist())
            if names != FEATURE_NAMES:
                raise ValueError(f"{path}: trọng số học trên đặc trưng {names}, code hiện tại dùng {FEATURE_NAMES}")

            def layers(name: str) -> list[Layer]:
                n = sum(1 for key in z.files if key.startswith(f"{name}.") and key.endswith(".weight"))
                return [(z[f"{name}.{i}.weight"], z[f"{name}.{i}.bias"]) for i in range(n)]

            return cls(f_out=layers("f_out"), f_in=layers("f_in"), temperature=float(z["temperature"]))


class LearnedMix:
    """Điểm mix học, cùng giao diện với RuleMix: `LearnedMix(tracks, encoders)(current_track_id, candidate_track_ids)`.

    Mã hóa cả danh mục một lần lúc dựng; mỗi lần gọi chỉ còn một phép nhân ma trận nhỏ.
    """

    def __init__(self, tracks: pd.DataFrame, encoders: MixEncoders):
        self.features = TrackFeatures.from_tracks(tracks)
        self.encoders = encoders
        x = audio_features(self.features)
        self._out = _mlp(x, encoders.f_out).astype(np.float32)
        self._in = _mlp(x, encoders.f_in).astype(np.float32)

    @classmethod
    def load(cls, tracks: pd.DataFrame, path: str | Path) -> LearnedMix:
        """Từ file .npz do mix_learned_train xuất ra (artifacts/mix_learned.npz)."""
        return cls(tracks, MixEncoders.load(path))

    def s_learn(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> np.ndarray:
        """f_out(x_a)ᵀ f_in(x_b) cho từng ứng viên b, trong [−1, 1]."""
        current = self.features.rows([current_track_id])[0]
        candidates = self.features.rows(list(candidate_track_ids))
        return self._in[candidates] @ self._out[current]

    def __call__(self, current_track_id: str, candidate_track_ids: Sequence[str]) -> np.ndarray:
        return np.clip((1.0 + self.s_learn(current_track_id, candidate_track_ids)) / 2.0, 0.0, 1.0)
