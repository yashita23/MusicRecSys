"""Mã Camelot từ key (pitch class 0–11) và mode (0 = thứ, 1 = trưởng), và ngược lại."""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np

from src.data.preview import preview_ids

CAMELOT_PATTERN = r"^(?:[1-9]|1[0-2])[AB]$"
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

