"""Mã Camelot từ key (pitch class 0–11) và mode (0 = thứ, 1 = trưởng), và ngược lại."""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from src.data.preview import preview_ids

CAMELOT_PATTERN = r"^(?:[1-9]|1[0-2])[AB]$"
MAX_CAMELOT_DISTANCE = 7  # 6 bước trên vòng (nửa vòng) + đổi chữ A/B
_CAMELOT_RE = re.compile(CAMELOT_PATTERN)


def camelot_code(key: int, mode: int) -> str:
    """C trưởng = 8B, A thứ = 8A; mỗi bước quãng năm (+7 bán cung) tăng 1 số trên vòng."""
    if mode == 1:
        return f"{(7 * key + 7) % 12 + 1}B"
    return f"{(7 * key + 4) % 12 + 1}A"


def parse_camelot(codes: Sequence[str] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """'8A' → (8, 'A'): mảng số 1–12 và mảng chữ A/B. Mã ngoài 1A–12B là lỗi."""
    arr = np.asarray(codes, dtype=str)
    valid = np.array([bool(_CAMELOT_RE.fullmatch(c)) for c in arr.ravel()]).reshape(arr.shape)
    if not valid.all():
        raise ValueError(f"mã Camelot không hợp lệ (phải trong 1A–12B): {preview_ids(arr[~valid].tolist())}")
    return np.char.rstrip(arr, "AB").astype(int), np.char.lstrip(arr, "0123456789")


def _wheel_step(num_a: np.ndarray, num_b: np.ndarray) -> np.ndarray:
    step = np.abs(num_a - num_b) % 12
    return np.minimum(step, 12 - step)  # vòng: 12 ↔ 1 cách nhau 1


def camelot_distance(num_a: np.ndarray, letter_a: np.ndarray, num_b: np.ndarray, letter_b: np.ndarray) -> np.ndarray:
    """Khoảng cách trên vòng Camelot: số bước theo số (0–6, 12 ↔ 1 liền nhau) cộng 1 nếu khác chữ A/B; trong 0–7."""
    return _wheel_step(num_a, num_b) + (letter_a != letter_b).astype(int)


def same_key(num_a: np.ndarray, letter_a: np.ndarray, num_b: np.ndarray, letter_b: np.ndarray) -> np.ndarray:
    """Cùng mã Camelot."""
    return (num_a == num_b) & (letter_a == letter_b)


def compatible_keys(num_a: np.ndarray, letter_a: np.ndarray, num_b: np.ndarray, letter_b: np.ndarray) -> np.ndarray:
    """Tông tương thích: cùng mã, ±1 số cùng chữ, hoặc cùng số khác chữ. Bước chéo (8A→9B) không tính."""
    step = _wheel_step(num_a, num_b)
    same_letter = letter_a == letter_b
    return (same_letter & (step <= 1)) | (~same_letter & (step == 0))

